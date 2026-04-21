"""KRAKEN v6: Final focused design.

Best-so-far is LEV_SECTORS at Sharpe 0.88. We need >=10 leveraged ETFs in universe.
Let's keep LEV_SECTORS small-ish but expand to hit the 10 minimum with additional
equity leveraged names. Use tight regime gate + per-asset trend + NAV stop.

Also: try "skip fat-tail months" — if SPY prior-month has bad return, skip.
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

UNIV_10 = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC"]  # 10 equity
UNIV_12 = UNIV_10 + ["YINN","NUGT"]  # 12
UNIV_15 = UNIV_12 + ["UGL","UCO","TMF"]  # 15
UNIV_17 = list(PAIRS.keys())  # 17

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
syms = set(UNIV_17 + SECTORS + list(underlying_syms) + ["BIL","SPY","IEF","TLT","GLD","USO","EEM","FXI","SMH","VNQ"])
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

def make_state(breadth_hi, vix_max, confirm, hy_win=60, ma=200, extra_spy=True, spy_ma=200):
    br = breadth_sig(ma=ma)
    hy_ema = hy.ewm(span=hy_win, adjust=False, min_periods=hy_win).mean()
    hy_ok  = (hy<hy_ema)
    vix_ok = (vix<vix_max)
    filt = (br>=breadth_hi) & hy_ok & vix_ok
    if extra_spy:
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

def build_trend_matrix(universe, win, use_under=True):
    out = pd.DataFrame(index=closes.index, columns=universe, dtype=float)
    for a in universe:
        src = a
        if use_under and a in PAIRS and PAIRS[a][0] in closes.columns:
            src = PAIRS[a][0]
        p = closes[src]
        ma = p.rolling(win, min_periods=win).mean()
        out[a] = (p > ma).astype(float)
    return out.shift(1).fillna(0.0)

def backtest_full(universe, state, trend_mat, rebal_days=21, tc_bps=10,
                  daily_exit=True, nav_stop=None, nav_stop_win=63, top_k=None):
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

    # momentum for optional top-K selection among trend-passing
    mom = closes[universe].pct_change(63).shift(1).reindex(columns=universe).values

    pnl_out = np.zeros(T); dW_out = np.zeros(T)
    nav = 1.0; nav_hist = []

    for i in range(T):
        stopped = False
        if nav_stop is not None and len(nav_hist) >= nav_stop_win:
            hwm = max(nav_hist[-nav_stop_win:])
            if nav/hwm - 1.0 <= -nav_stop:
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
                if top_k is not None and valid.sum() > top_k:
                    m = mom[i].copy()
                    m[~valid] = -np.inf
                    idx_top = np.argpartition(-m, top_k-1)[:top_k]
                    picks = np.zeros_like(valid, dtype=bool)
                    picks[idx_top] = True
                    valid = picks & valid
                target = np.zeros(len(cols))
                target[:-1] = valid.astype(float) / valid.sum()
            else:
                target = np.zeros(len(cols)); target[-1]=1.0
        elif daily_exit and bull:
            target = last_w.copy()
            held = target[:-1] > 0
            bad = held & (~valid)
            if bad.any():
                target[:-1][bad] = 0.0
                target[-1] = 1.0 - target[:-1].sum()
        else:
            target = last_w.copy()
            # If state flipped off mid-month, exit
            if not bull and was_bull:
                target = np.zeros(len(cols)); target[-1]=1.0

        dW = np.abs(target - last_w).sum()
        tc = dW * (tc_bps/1e4)
        pnl = (target * or_mat[i]).sum() - tc
        nav *= (1.0 + pnl)
        nav_hist.append(nav)
        W[i] = target; last_w = target
        pnl_out[i] = pnl; dW_out[i] = dW

    r = pd.Series(pnl_out, index=closes.index)
    turn = pd.Series(dW_out, index=closes.index)
    return r, turn, W

def eval_cfg(univ, bh, vxm, cf, tw, rb, daily_exit=True, nav_stop=None, top_k=None, tc_bps=10):
    st = make_state(bh, vxm, cf, extra_spy=True, spy_ma=200)
    tr = build_trend_matrix(univ, tw)
    r, turn, W = backtest_full(univ, st, tr, rb, tc_bps,
                                daily_exit=daily_exit,
                                nav_stop=nav_stop, nav_stop_win=63,
                                top_k=top_k)
    is_m = metrics(r[r.index<=IS_END])
    oos_m = metrics(r[r.index>=OOS_S])
    full_m = metrics(r)
    return dict(
        un=len(univ), bh=bh, vxm=vxm, cf=cf, tw=tw, rb=rb, dex=daily_exit, ns=nav_stop, tk=top_k,
        full_sr=full_m["sharpe"], full_cagr=full_m["cagr"], full_mdd=full_m["mdd"], full_vol=full_m["vol"], full_navx=full_m["navx"],
        is_sr=is_m["sharpe"], is_cagr=is_m["cagr"], is_mdd=is_m["mdd"],
        oos_sr=oos_m["sharpe"], oos_cagr=oos_m["cagr"], oos_mdd=oos_m["mdd"],
        gap=abs(is_m["sharpe"]-oos_m["sharpe"]), tim=float((st==1).mean()),
        turn_ann=turn.mean()*252,
    )

if __name__ == "__main__":
    rows=[]
    universes = [
        ("UNIV_10", UNIV_10),
        ("UNIV_12", UNIV_12),
        ("UNIV_15", UNIV_15),
    ]
    grid = list(itertools.product(
        universes,
        [150, 200, 250],          # trend win
        [10, 21],                 # rebal
        [0.5, 0.6],               # bh
        [22, 25, 28],             # vxm
        [3, 5],                   # confirm
        [True, False],            # daily exit
        [None, 0.10, 0.15, 0.20], # nav stop
        [None, 2, 3, 5],          # top_k
    ))
    print(f"configs: {len(grid)}")
    for i, ((uname, univ), tw, rb, bh, vxm, cf, dex, ns, tk) in enumerate(grid):
        try:
            res = eval_cfg(univ, bh, vxm, cf, tw, rb, dex, ns, tk)
            res["uname"] = uname
            rows.append(res)
        except Exception as e:
            pass
        if i%200==0: print(f"  {i}/{len(grid)}")

    df = pd.DataFrame(rows)
    df.to_csv("data/results/kraken_grid6.csv", index=False)

    print("\n--- Top 25 by full_sr ---")
    print(df.sort_values("full_sr", ascending=False).head(25).to_string(index=False))

    mask = (df.full_sr>=2.0)&(df.full_cagr>=0.20)&(df.is_sr>=1.5)&(df.oos_sr>=1.5)&(df.gap<=0.5)
    qual = df[mask].sort_values("full_sr", ascending=False)
    print(f"\n--- Qualifying (hard reqs): {len(qual)} ---")
    print(qual.head(30).to_string(index=False))

    mask2 = (df.full_sr>=1.2)&(df.is_sr>=0.8)&(df.oos_sr>=0.8)&(df.gap<=0.5)
    near = df[mask2].sort_values("full_sr", ascending=False)
    print(f"\n--- sr>=1.2 robust: {len(near)} ---")
    print(near.head(20).to_string(index=False))

    # Best by Calmar
    df2 = df[df.full_cagr>=0.20].copy()
    df2["calmar"] = df2.full_cagr / (-df2.full_mdd)
    print(f"\n--- Top 15 by Calmar (cagr>=20%) ---")
    print(df2.sort_values("calmar", ascending=False).head(15).to_string(index=False))
