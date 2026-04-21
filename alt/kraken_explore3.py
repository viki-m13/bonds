"""KRAKEN v3: richer regime + trailing stop + broader set of ideas.

Ideas to test:
- Multi-filter regime: SPY>200dMA, SPY>10mMA, breadth, VIX, HY spread, T10Y2Y, DFF
- Use 2x ETFs (SSO,QLD) preferentially over 3x
- Trailing stop on NAV
- Conditional rebalance: only when regime changes or trend changes
- Apply asset-level trend filter on UNDERLIER
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
LEV_EQ_BULL  = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC","YINN"]
LEV_2X_ONLY  = ["QLD","SSO","ERX","NUGT","UGL","UCO","UBT"]
LEV_US       = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS"]
LEV_SECTORS  = ["SOXL","TECL","FAS","ERX","DRN"]
LEV_ALL      = LEV_EQ_BULL + ["NUGT","UGL","UCO","TMF","UBT","TYD"]

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
    except FileNotFoundError:
        pass
opens = pd.DataFrame(opens); closes = pd.DataFrame(closes)
cal = closes["SPY"].dropna().index
cal = cal[(cal>=START) & (cal<=OOS_E)]
opens = opens.reindex(cal); closes = closes.reindex(cal)
vix = load_fred("VIXCLS").reindex(cal).ffill()
hy  = load_fred("BAMLH0A0HYM2").reindex(cal).ffill()
try: t10y2y = load_fred("T10Y2Y").reindex(cal).ffill()
except: t10y2y = pd.Series(1.0, index=cal)

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

def make_state_multi(breadth_hi=0.55, vix_max=25, hy_win=60, confirm=3, ma=200,
                    use_spy_trend=True, spy_ma=200, use_yc=False):
    br = breadth_sig(ma=ma)
    hy_ema = hy.ewm(span=hy_win, adjust=False, min_periods=hy_win).mean()
    hy_ok  = (hy<hy_ema)
    vix_ok = (vix<vix_max)
    filt = (br>=breadth_hi) & hy_ok & vix_ok
    if use_spy_trend:
        spy_ok = (closes["SPY"] > closes["SPY"].rolling(spy_ma, min_periods=spy_ma).mean())
        filt = filt & spy_ok
    if use_yc:
        filt = filt & (t10y2y > -0.3)
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

def build_trend_matrix(universe, use_underlier=True, win=200, method="ma"):
    out = pd.DataFrame(index=closes.index, columns=universe, dtype=float)
    for a in universe:
        src = a
        if use_underlier and a in PAIRS and PAIRS[a][0] in closes.columns:
            src = PAIRS[a][0]
        p = closes[src]
        if method == "ma":
            ma = p.rolling(win, min_periods=win).mean()
            out[a] = (p > ma).astype(float)
        elif method == "dual":
            ma_f = p.rolling(max(win//4,20), min_periods=max(win//4,20)).mean()
            ma_s = p.rolling(win, min_periods=win).mean()
            out[a] = ((p>ma_s) & (ma_f>ma_s)).astype(float)
    return out.shift(1).fillna(0.0)

def backtest_per_asset(universe, state, trend_mat, rebal_days=21, tc_bps=10,
                       nav_stop_dd=None, nav_stop_win=63):
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

    # Tracking NAV for trailing-stop
    nav = 1.0
    nav_hist = []
    stopped = False

    pnl_out = np.zeros(T)
    dW_out = np.zeros(T)

    for i in range(T):
        # Compute trailing stop on strategy NAV
        if nav_stop_dd is not None and len(nav_hist) >= nav_stop_win:
            recent = nav_hist[-nav_stop_win:]
            hwm = max(recent)
            if nav / hwm - 1.0 <= -nav_stop_dd:
                stopped = True
            else:
                stopped = False

        bull_regime = bool(state_arr[i]) and not stopped
        was_bull = last_w[:-1].sum() > 0.5
        rebal_today = rebal_mask[i] or (bull_regime != was_bull)
        if rebal_today:
            if bull_regime:
                t_row = trend_arr[i]
                valid = np.isfinite(close_arr[i]) & (t_row > 0)
                n = int(valid.sum())
                if n > 0:
                    target = np.zeros(len(cols))
                    target[:-1] = valid.astype(float) / n
                else:
                    target = np.zeros(len(cols)); target[-1]=1.0
            else:
                target = np.zeros(len(cols)); target[-1]=1.0
        else:
            target = last_w.copy()

        dW = np.abs(target - last_w).sum()
        tc = dW * (tc_bps/1e4)
        pnl = (target * or_mat[i]).sum() - tc
        nav = nav * (1.0 + pnl)
        nav_hist.append(nav)
        W[i] = target
        last_w = target
        pnl_out[i] = pnl
        dW_out[i] = dW

    r = pd.Series(pnl_out, index=closes.index)
    turn = pd.Series(dW_out, index=closes.index)
    return r, turn, W

def eval_cfg(universe, bh, vxm, cf, tw, rb, use_under, ma=200, use_spy=True, spy_ma=200, use_yc=False, nav_stop=None, nav_stop_win=63, tc_bps=10):
    st = make_state_multi(bh, vxm, 60, cf, ma, use_spy, spy_ma, use_yc)
    tr = build_trend_matrix(universe, use_under, tw)
    r, turn, W = backtest_per_asset(universe, st, tr, rb, tc_bps,
                                     nav_stop_dd=nav_stop, nav_stop_win=nav_stop_win)
    is_m = metrics(r[r.index<=IS_END])
    oos_m = metrics(r[r.index>=OOS_S])
    full_m = metrics(r)
    return dict(
        un=len(universe), bh=bh, vxm=vxm, cf=cf, tw=tw, rb=rb, un_u=use_under,
        spy=use_spy, spyma=spy_ma, yc=use_yc, ns=nav_stop, nsw=nav_stop_win,
        full_sr=full_m["sharpe"], full_cagr=full_m["cagr"], full_mdd=full_m["mdd"], full_vol=full_m["vol"], full_navx=full_m["navx"],
        is_sr=is_m["sharpe"], is_cagr=is_m["cagr"], is_mdd=is_m["mdd"],
        oos_sr=oos_m["sharpe"], oos_cagr=oos_m["cagr"], oos_mdd=oos_m["mdd"],
        gap=abs(is_m["sharpe"]-oos_m["sharpe"]), tim=float((st==1).mean()),
        turn_ann=turn.mean()*252,
    )

if __name__=="__main__":
    rows=[]
    # Smart grid
    universes = [
        ("LEV_US", LEV_US),
        ("LEV_EQ_BULL", LEV_EQ_BULL),
        ("LEV_SECTORS", LEV_SECTORS),
        ("LEV_ALL", LEV_ALL),
    ]
    grid = list(itertools.product(
        universes,
        [150, 200, 250],       # trend win
        [5, 10, 21],           # rebal
        [0.50, 0.60],          # breadth_hi
        [22, 28],              # vix_max
        [3, 5],                # confirm
        [True],                # use underlier
        [True],                # spy trend gate
        [200],                 # spy ma
        [0.10, 0.15, None],    # nav stop
    ))
    print(f"configs: {len(grid)}")
    for i, ((uname,univ), tw, rb, bh, vxm, cf, ul, spy, spyma, ns) in enumerate(grid):
        res = eval_cfg(univ, bh, vxm, cf, tw, rb, ul, use_spy=spy, spy_ma=spyma, nav_stop=ns)
        res["uname"] = uname
        rows.append(res)
        if i%100==0: print(f"  {i}/{len(grid)}")
    df = pd.DataFrame(rows)
    df.to_csv("data/results/kraken_grid3.csv", index=False)

    print("\n--- Top 25 by full_sr ---")
    print(df.sort_values("full_sr", ascending=False).head(25).to_string(index=False))

    mask = (df.full_sr>=2.0)&(df.full_cagr>=0.20)&(df.is_sr>=1.5)&(df.oos_sr>=1.5)&(df.gap<=0.5)
    qual = df[mask].sort_values("full_sr", ascending=False)
    print(f"\n--- Qualifying: {len(qual)} ---")
    print(qual.head(30).to_string(index=False))

    mask2 = (df.full_sr>=1.3)&(df.is_sr>=1.0)&(df.oos_sr>=1.0)&(df.gap<=0.6)
    near = df[mask2].sort_values("full_sr", ascending=False)
    print(f"\n--- Near-qualifying (sr>=1.3): {len(near)} ---")
    print(near.head(30).to_string(index=False))
