"""KRAKEN v2 exploration: per-asset trend-following.

For each leveraged ETF, hold if its own trend is positive AND regime is on.
Weight equally across holders. No cross-sectional momentum.

Also try: "underlier trend" — use the unleveraged index instead of the
leveraged ETF for trend signal (cleaner signal).
"""
import os, json, itertools
import numpy as np
import pandas as pd

REPO   = "/home/user/bonds"
ETF_D  = os.path.join(REPO, "data", "etfs")
FRED_D = os.path.join(REPO, "data", "fred")

SECTORS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]

# Pairs: leveraged -> (underlying, leverage)
PAIRS = {
    "TQQQ": ("QQQ", 3), "UPRO": ("SPY", 3),
    "QLD":  ("QQQ", 2), "SSO":  ("SPY", 2),
    "SOXL": ("SMH", 3), "TECL": ("XLK", 3),
    "FAS":  ("XLF", 3), "LABU": ("IBB", 3),
    "ERX":  ("XLE", 2), "DRN":  ("VNQ", 3),
    "EDC":  ("EEM", 3), "YINN": ("FXI", 3),
    "NUGT": ("GLD", 2), "UGL":  ("GLD", 2),
    "UCO":  ("USO", 2), "TMF":  ("TLT", 3),
    "UBT":  ("TLT", 2), "TYD":  ("IEF", 3),
}
LEV_EQ  = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC","YINN","LABU"]
LEV_ALL = LEV_EQ + ["NUGT","UGL","UCO","TMF","UBT","TYD"]

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

# Gather all symbols needed
underlying_syms = set(u for u,_ in PAIRS.values())
syms = set(LEV_ALL + SECTORS + list(underlying_syms) + ["BIL","SPY"])
opens={}; closes={}
for s in syms:
    try:
        df = load_etf(s); opens[s]=df["Open"]; closes[s]=df["Close"]
    except FileNotFoundError:
        print(f"missing {s}")
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

def macro_sig(vix_max=25, hy_win=60):
    hy_ema = hy.ewm(span=hy_win, adjust=False, min_periods=hy_win).mean()
    return ((hy<hy_ema)&(vix<vix_max)).astype(float)

def make_state(breadth_hi=0.55, vix_max=25, hy_win=60, confirm=3, ma=200):
    br = breadth_sig(ma=ma)
    mc = macro_sig(vix_max=vix_max, hy_win=hy_win)
    raw = ((br>=breadth_hi) & (mc==1.0)).astype(int).values
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

def trend_filter(df, win):
    """Binary: is close above its win-day MA? Lag 1 bar."""
    ma = df.rolling(win, min_periods=win).mean()
    return (df > ma).astype(float).shift(1).fillna(0.0)

def backtest_per_asset(universe, state, trend_mat, rebal_days=21, tc_bps=10, max_hold=None, vol_cap=None, lookback_vol=63):
    """Each asset held iff trend_mat[asset]==1 AND state==1 on the rebalance day.
    Weights = 1/number_of_holders (capped at max_hold if given)."""
    T = len(state)
    cols = universe + ["BIL"]
    W = np.zeros((T, len(cols)))
    state_arr = state.values
    last_w = np.zeros(len(cols))
    rebal_mask = np.zeros(T, dtype=bool)
    for i in range(0, T, rebal_days): rebal_mask[i]=True

    trend_arr = trend_mat.reindex(columns=universe).fillna(0.0).values
    close_arr = closes.reindex(columns=universe).values

    # Optional vol cap: inverse-vol weight
    if vol_cap is not None:
        ret_univ = closes[universe].pct_change()
        vol = ret_univ.rolling(lookback_vol, min_periods=lookback_vol).std().shift(1).values
    else:
        vol = None

    for i in range(T):
        bull = bool(state_arr[i])
        was_bull = last_w[:-1].sum() > 0.5
        rebal_today = rebal_mask[i] or (bull != was_bull)
        if rebal_today:
            if bull:
                t_row = trend_arr[i]
                valid = np.isfinite(close_arr[i]) & (t_row > 0)
                if max_hold is not None and valid.sum() > max_hold:
                    # keep top by simple prior 21d return among passing
                    pass  # not used
                n = valid.sum()
                if n > 0:
                    target = np.zeros(len(cols))
                    if vol is not None:
                        v = vol[i].copy()
                        v[~valid] = np.nan
                        inv = 1.0 / v
                        inv[~np.isfinite(inv)] = 0
                        s = inv.sum()
                        if s > 0:
                            w = inv/s
                            # cap each at max 1.0 (long-only)
                            target[:-1] = w
                        else:
                            target[-1]=1.0
                    else:
                        target[:-1] = valid.astype(float) / n
                else:
                    target = np.zeros(len(cols)); target[-1]=1.0
            else:
                target = np.zeros(len(cols)); target[-1]=1.0
        else:
            target = last_w.copy()
        W[i] = target
        last_w = target

    or_mat = open_ret.reindex(columns=cols).values
    pnl = (W * or_mat).sum(axis=1)
    dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1,len(cols))))).sum(axis=1)
    tc_cost = dW * (tc_bps/1e4)
    r = pd.Series(pnl - tc_cost, index=closes.index)
    turn = pd.Series(dW, index=closes.index)
    return r, turn, W

def build_trend_matrix(universe, use_underlier=True, win=200):
    """Matrix of trend signals for each asset in universe."""
    out = pd.DataFrame(index=closes.index, columns=universe, dtype=float)
    for a in universe:
        src = a
        if use_underlier and a in PAIRS:
            src = PAIRS[a][0]
            if src not in closes.columns:
                src = a
        p = closes[src]
        ma = p.rolling(win, min_periods=win).mean()
        out[a] = (p > ma).astype(float)
    return out.shift(1).fillna(0.0)

def eval_cfg(universe, bh, vxm, cf, trend_win, rebal, use_under, tc_bps=10, vol_cap=False, ma=200):
    st = make_state(bh, vxm, 60, cf, ma)
    tr = build_trend_matrix(universe, use_under, trend_win)
    r, turn, W = backtest_per_asset(universe, st, tr, rebal, tc_bps, vol_cap=(None if not vol_cap else True))
    is_m = metrics(r[r.index<=IS_END])
    oos_m = metrics(r[r.index>=OOS_S])
    full_m = metrics(r)
    return dict(
        univ=len(universe), bh=bh, vxm=vxm, cf=cf, tw=trend_win, rb=rebal, under=use_under, vc=vol_cap,
        full_sr=full_m["sharpe"], full_cagr=full_m["cagr"], full_mdd=full_m["mdd"], full_vol=full_m["vol"], full_navx=full_m["navx"],
        is_sr=is_m["sharpe"], is_cagr=is_m["cagr"], is_mdd=is_m["mdd"],
        oos_sr=oos_m["sharpe"], oos_cagr=oos_m["cagr"], oos_mdd=oos_m["mdd"],
        gap=abs(is_m["sharpe"]-oos_m["sharpe"]), tim=float((st==1).mean()),
        turn_ann=turn.mean()*252,
    )

rows=[]
grid = list(itertools.product(
    [("LEV_EQ",LEV_EQ), ("LEV_ALL",LEV_ALL)],
    [100, 150, 200, 250],    # trend win
    [5, 10, 21],             # rebal
    [0.50, 0.60, 0.70],      # breadth_hi
    [22, 25, 28],            # vix_max
    [1, 3, 5],               # confirm
    [True, False],           # use underlier
    [False, True],           # inv-vol weight
))
print(f"configs: {len(grid)}")
for i, ((uname,univ), tw, rb, bh, vxm, cf, ul, vc) in enumerate(grid):
    res = eval_cfg(univ, bh, vxm, cf, tw, rb, ul, vol_cap=vc)
    res["uname"] = uname
    rows.append(res)
    if i%100==0: print(f"  {i}/{len(grid)}")
df = pd.DataFrame(rows)
df.to_csv("data/results/kraken_grid2.csv", index=False)

print("\n--- Top 20 by full_sr ---")
print(df.sort_values("full_sr", ascending=False).head(20).to_string(index=False))

mask = (df.full_sr>=2.0)&(df.full_cagr>=0.20)&(df.is_sr>=1.5)&(df.oos_sr>=1.5)&(df.gap<=0.5)
qual = df[mask].sort_values("full_sr", ascending=False)
print(f"\n--- Qualifying: {len(qual)} ---")
print(qual.head(30).to_string(index=False))

mask2 = (df.full_sr>=1.5)&(df.is_sr>=1.2)&(df.oos_sr>=1.2)&(df.gap<=0.6)
near = df[mask2].sort_values("full_sr", ascending=False)
print(f"\n--- Near-qualifying: {len(near)} ---")
print(near.head(30).to_string(index=False))
