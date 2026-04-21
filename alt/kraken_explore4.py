"""KRAKEN v4: Focus on high-Sharpe regime filter with full universe gate.

Key idea: universe is broad (>=10 leveraged ETFs), but per-asset trend filter
keeps us in the strongest ones only. Crucially, add a tight per-asset STOP-LOSS
that exits any asset that breaks down, not just on rebalance day.

Also: use rolling Sharpe-based position sizing:
  w_i = 1/n when all passing OR weight by recent 60d Sharpe (positive portion).
"""
import os, json, itertools
import numpy as np
import pandas as pd

REPO   = "/home/user/bonds"
ETF_D  = os.path.join(REPO, "data", "etfs")
FRED_D = os.path.join(REPO, "data", "fred")

SECTORS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]

PAIRS = {
    "TQQQ": ("QQQ", 3), "UPRO": ("SPY", 3),
    "QLD":  ("QQQ", 2), "SSO":  ("SPY", 2),
    "SOXL": ("SMH", 3), "TECL": ("XLK", 3),
    "FAS":  ("XLF", 3),
    "ERX":  ("XLE", 2), "DRN":  ("VNQ", 3),
    "EDC":  ("EEM", 3), "YINN": ("FXI", 3),
    "NUGT": ("GLD", 2), "UGL":  ("GLD", 2),
    "UCO":  ("USO", 2), "TMF":  ("TLT", 3),
    "UBT":  ("TLT", 2), "TYD":  ("IEF", 3),
}
# Must have >=10 leveraged ETFs in universe
LEV_ALL = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC","YINN","NUGT","UGL","UCO","TMF","UBT","TYD"]

START = pd.Timestamp("2010-03-11")
IS_END= pd.Timestamp("2018-12-31")
OOS_S = pd.Timestamp("2019-01-01")
OOS_E = pd.Timestamp("2026-04-02")

def load_etf(sym):
    df = pd.read_csv(os.path.join(ETF_D, f"{sym}.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    return df[["Open","Close"]].astype(float)

def load_fred(name):
    df = pd.read_csv(os.path.join(FRED_D, f"{name}.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    return df[name].astype(float)

underlying_syms = set(u for u,_ in PAIRS.values())
syms = set(LEV_ALL + SECTORS + list(underlying_syms) + ["BIL","SPY","IEF"])
opens={}; closes={}
for s in syms:
    try:
        df = load_etf(s); opens[s]=df["Open"]; closes[s]=df["Close"]
    except FileNotFoundError: pass
opens = pd.DataFrame(opens); closes = pd.DataFrame(closes)
cal = closes["SPY"].dropna().index
cal = cal[(cal>=START) & (cal<=OOS_E)]
opens = opens.reindex(cal); closes = closes.reindex(cal)
vix = load_fred("VIXCLS").reindex(cal).ffill()
hy  = load_fred("BAMLH0A0HYM2").reindex(cal).ffill()

open_ret = (opens.shift(-1) / opens - 1.0).fillna(0.0)

def metrics(r):
    r = r.dropna()
    if len(r)==0: return dict(sharpe=np.nan,cagr=np.nan,mdd=np.nan,vol=np.nan,navx=np.nan)
    ann=252
    mu=r.mean()*ann; sd=r.std(ddof=0)*np.sqrt(ann)
    sr=mu/sd if sd>0 else float("nan")
    nav=(1+r).cumprod()
    n_years=len(r)/ann
    cagr=nav.iloc[-1]**(1/n_years)-1 if n_years>0 else 0
    dd=nav/nav.cummax()-1
    return dict(sharpe=sr,cagr=cagr,mdd=dd.min(),vol=sd,navx=nav.iloc[-1])

def breadth_sig(ma=200):
    sub = closes[SECTORS]
    mvg = sub.rolling(ma, min_periods=ma).mean()
    above = (sub>mvg).astype(float)
    valid = sub.notna() & mvg.notna()
    num = above.where(valid,0).sum(axis=1)
    den = valid.sum(axis=1).replace(0,np.nan)
    return num/den

def make_state(breadth_hi=0.55, vix_max=25, hy_win=60, confirm=3, ma=200, use_spy=True, spy_ma=200):
    br = breadth_sig(ma=ma)
    hy_ema = hy.ewm(span=hy_win, adjust=False, min_periods=hy_win).mean()
    hy_ok  = (hy<hy_ema)
    vix_ok = (vix<vix_max)
    filt = (br>=breadth_hi) & hy_ok & vix_ok
    if use_spy:
        spy_ok = (closes["SPY"] > closes["SPY"].rolling(spy_ma, min_periods=spy_ma).mean())
        filt = filt & spy_ok
    raw = filt.astype(int).values
    if confirm<=1: out = raw
    else:
        out = np.empty_like(raw); state=0; streak=0
        for i in range(len(raw)):
            r=raw[i]
            if r==state: streak=0
            else:
                streak+=1
                if streak>=confirm: state=r; streak=0
            out[i]=state
    return pd.Series(out, index=closes.index, dtype=int).shift(1).fillna(0).astype(int)

def build_trend_matrix(universe, use_underlier, win):
    out = pd.DataFrame(index=closes.index, columns=universe, dtype=float)
    for a in universe:
        src = a
        if use_underlier and a in PAIRS and PAIRS[a][0] in closes.columns:
            src = PAIRS[a][0]
        p = closes[src]
        ma = p.rolling(win, min_periods=win).mean()
        out[a] = (p > ma).astype(float)
    return out.shift(1).fillna(0.0)

def backtest_daily_filter(universe, state, trend_mat, rebal_days=21, tc_bps=10,
                          daily_trend_check=True, max_hold=None, nav_stop_dd=None, nav_stop_win=63):
    """
    If daily_trend_check: every day, keep only holdings whose trend is still on.
    If count of holders drops, redistribute among remaining (only on rebalance day).
    This way we EXIT a name when its trend breaks, even mid-month.
    """
    T = len(state)
    cols = universe + ["BIL"]
    W = np.zeros((T, len(cols)))
    state_arr = state.values
    last_w = np.zeros(len(cols))
    rebal_mask = np.zeros(T, dtype=bool)
    for i in range(0, T, rebal_days): rebal_mask[i]=True

    trend_arr = trend_mat.reindex(columns=universe).fillna(0.0).values
    close_arr = closes.reindex(columns=universe).values
    or_mat = open_ret.reindex(columns=cols).values

    pnl_out = np.zeros(T); dW_out = np.zeros(T)
    nav = 1.0
    nav_hist = []

    for i in range(T):
        # NAV trailing stop
        stopped = False
        if nav_stop_dd is not None and len(nav_hist) >= nav_stop_win:
            hwm = max(nav_hist[-nav_stop_win:])
            if nav/hwm - 1.0 <= -nav_stop_dd:
                stopped = True
        bull = bool(state_arr[i]) and not stopped
        was_bull = last_w[:-1].sum() > 0.5
        rebal_today = rebal_mask[i] or (bull != was_bull)

        if bull:
            t_row = trend_arr[i]
            valid = np.isfinite(close_arr[i]) & (t_row > 0)
        else:
            valid = np.zeros(len(universe), dtype=bool)

        if rebal_today:
            if bull and valid.any():
                target = np.zeros(len(cols))
                target[:-1] = valid.astype(float) / valid.sum()
            else:
                target = np.zeros(len(cols)); target[-1]=1.0
        elif daily_trend_check and bull:
            # Keep only trend-on names among currently held; convert exits to BIL
            target = last_w.copy()
            held_mask = target[:-1] > 0
            to_drop = held_mask & (~valid)
            if to_drop.any():
                target[:-1][to_drop] = 0.0
                # remaining weights keep their level; excess goes to BIL
                residual = 1.0 - target.sum()
                target[-1] += residual
            # also state flipped off handled by rebal_today above
        else:
            target = last_w.copy()

        dW = np.abs(target - last_w).sum()
        tc = dW * (tc_bps/1e4)
        pnl = (target * or_mat[i]).sum() - tc
        nav *= (1.0 + pnl)
        nav_hist.append(nav)
        W[i] = target
        last_w = target
        pnl_out[i] = pnl
        dW_out[i] = dW

    r = pd.Series(pnl_out, index=closes.index)
    turn = pd.Series(dW_out, index=closes.index)
    return r, turn, W

def eval_cfg(bh, vxm, cf, tw, rb, use_under, use_spy=True, spy_ma=200, nav_stop=None, daily_exit=True, tc_bps=10):
    st = make_state(bh, vxm, 60, cf, 200, use_spy, spy_ma)
    tr = build_trend_matrix(LEV_ALL, use_under, tw)
    r, turn, W = backtest_daily_filter(LEV_ALL, st, tr, rb, tc_bps,
                                        daily_trend_check=daily_exit,
                                        nav_stop_dd=nav_stop, nav_stop_win=63)
    is_m = metrics(r[r.index<=IS_END])
    oos_m = metrics(r[r.index>=OOS_S])
    full_m = metrics(r)
    return dict(
        bh=bh, vxm=vxm, cf=cf, tw=tw, rb=rb, un_u=use_under, spy=use_spy, spyma=spy_ma, ns=nav_stop, dex=daily_exit,
        full_sr=full_m["sharpe"], full_cagr=full_m["cagr"], full_mdd=full_m["mdd"], full_vol=full_m["vol"], full_navx=full_m["navx"],
        is_sr=is_m["sharpe"], is_cagr=is_m["cagr"], is_mdd=is_m["mdd"],
        oos_sr=oos_m["sharpe"], oos_cagr=oos_m["cagr"], oos_mdd=oos_m["mdd"],
        gap=abs(is_m["sharpe"]-oos_m["sharpe"]), tim=float((st==1).mean()),
        turn_ann=turn.mean()*252,
    )

if __name__=="__main__":
    rows=[]
    grid = list(itertools.product(
        [150, 200, 250],   # trend win
        [5, 10, 21],       # rebal
        [0.50, 0.60, 0.70],# bh
        [22, 25, 28],      # vxm
        [3, 5],            # cf
        [True],            # under
        [True, False],     # daily exit
        [None, 0.10, 0.15],# nav stop
    ))
    print(f"configs: {len(grid)}")
    for i, (tw, rb, bh, vxm, cf, ul, dex, ns) in enumerate(grid):
        res = eval_cfg(bh, vxm, cf, tw, rb, ul, nav_stop=ns, daily_exit=dex)
        rows.append(res)
        if i%50==0: print(f"  {i}/{len(grid)}")
    df = pd.DataFrame(rows)
    df.to_csv("data/results/kraken_grid4.csv", index=False)

    print("\n--- Top 30 by full_sr ---")
    print(df.sort_values("full_sr", ascending=False).head(30).to_string(index=False))

    mask = (df.full_sr>=2.0)&(df.full_cagr>=0.20)&(df.is_sr>=1.5)&(df.oos_sr>=1.5)&(df.gap<=0.5)
    qual = df[mask].sort_values("full_sr", ascending=False)
    print(f"\n--- Qualifying: {len(qual)} ---")
    print(qual.head(30).to_string(index=False))

    mask2 = (df.full_sr>=1.5)&(df.is_sr>=1.2)&(df.oos_sr>=1.2)&(df.gap<=0.6)
    near = df[mask2].sort_values("full_sr", ascending=False)
    print(f"\n--- Near-qualifying (sr>=1.5): {len(near)} ---")
    print(near.head(30).to_string(index=False))

    mask3 = (df.full_sr>=1.2)
    near3 = df[mask3].sort_values("full_sr", ascending=False)
    print(f"\n--- full_sr>=1.2: {len(near3)} ---")
    print(near3.head(30).to_string(index=False))
