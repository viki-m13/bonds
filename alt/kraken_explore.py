"""KRAKEN grid search with VECTORIZED backtest.

Strategy: regime_state (breadth+macro) x top-K momentum on leveraged universe.
Monthly/weekly rebalance, cash when regime off.

All signals are lagged 1 bar to use only info through close[t-1].
PnL realized open[t]->open[t+1].
"""
import os, json
import numpy as np
import pandas as pd

REPO   = "/home/user/bonds"
ETF_D  = os.path.join(REPO, "data", "etfs")
FRED_D = os.path.join(REPO, "data", "fred")

SECTORS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]
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

syms = set(LEV_ALL + SECTORS + ["BIL","SPY"])
opens={}; closes={}
for s in syms:
    df = load_etf(s); opens[s]=df["Open"]; closes[s]=df["Close"]
opens = pd.DataFrame(opens); closes = pd.DataFrame(closes)
cal = closes["SPY"].dropna().index
cal = cal[(cal>=START) & (cal<=OOS_E)]
opens = opens.reindex(cal); closes = closes.reindex(cal)
vix = load_fred("VIXCLS").reindex(cal).ffill()
hy  = load_fred("BAMLH0A0HYM2").reindex(cal).ffill()

# Precompute open-to-open returns once
open_ret = (opens.shift(-1) / opens - 1.0).fillna(0.0)   # shape [T, assets]

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
    if confirm<=1:
        out = raw
    else:
        out = np.empty_like(raw); state=0; streak=0
        for i in range(len(raw)):
            r = raw[i]
            if r == state: streak=0
            else:
                streak+=1
                if streak>=confirm:
                    state=r; streak=0
            out[i]=state
    s = pd.Series(out, index=closes.index, dtype=int)
    return s.shift(1).fillna(0).astype(int)

def momentum(df, lb, skip):
    return ((df.shift(skip)/df.shift(skip+lb))-1.0).shift(1)

def backtest_vec(universe, state, mom, rebal_days, top_k, tc_bps=10):
    """Fully vectorized-ish backtest."""
    T = len(state)
    universe = list(universe)
    # Build target weight matrix (T x (n_univ+1)), last col is BIL
    cols = universe + ["BIL"]
    W = np.zeros((T, len(cols)))
    state_arr = state.values
    last_w = np.zeros(len(cols))
    rebal_mask = np.zeros(T, dtype=bool)
    for i in range(0, T, rebal_days): rebal_mask[i]=True

    # For each day, determine if regime flipped from last weight
    # Track previous bull-ness via last_w sum on universe
    mom_arr = mom.reindex(columns=universe).values   # T x N
    close_arr = closes.reindex(columns=universe).values
    for i in range(T):
        bull = bool(state_arr[i])
        was_bull = last_w[:-1].sum() > 0.5
        rebal_today = rebal_mask[i] or (bull != was_bull)
        if rebal_today:
            if bull:
                m = mom_arr[i].copy()
                # valid = finite m and finite close today
                valid = np.isfinite(m) & np.isfinite(close_arr[i])
                m[~valid] = -np.inf
                # positive momentum only
                m[m<=0] = -np.inf
                if np.any(np.isfinite(m)):
                    k = min(top_k, int(np.sum(np.isfinite(m))))
                    idx_top = np.argpartition(-m, k-1)[:k] if k>0 else []
                    # filter to actually-finite picks
                    idx_top = [j for j in idx_top if np.isfinite(m[j])]
                    target = np.zeros(len(cols))
                    if len(idx_top)>0:
                        for j in idx_top: target[j] = 1.0/len(idx_top)
                    else:
                        target[-1] = 1.0
                else:
                    target = np.zeros(len(cols)); target[-1]=1.0
            else:
                target = np.zeros(len(cols)); target[-1]=1.0
        else:
            target = last_w.copy()
        W[i] = target
        last_w = target

    # PnL: W[i] . open_ret[i]   and   TC on |W[i] - W[i-1]|
    or_mat = open_ret.reindex(columns=cols).values    # T x cols
    pnl = (W * or_mat).sum(axis=1)
    dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1,len(cols))))).sum(axis=1)
    tc_cost = dW * (tc_bps/1e4)
    r = pd.Series(pnl - tc_cost, index=closes.index)
    turn = pd.Series(dW, index=closes.index)
    return r, turn, W

def eval_config(universe, breadth_hi, vix_max, confirm, mom_lb, mom_skip, top_k, rebal, ma=200, hy_win=60, tc_bps=10):
    st = make_state(breadth_hi, vix_max, hy_win, confirm, ma)
    mom = momentum(closes[universe], mom_lb, mom_skip)
    r, turn, W = backtest_vec(universe, st, mom, rebal, top_k, tc_bps=tc_bps)
    is_m = metrics(r[r.index<=IS_END])
    oos_m = metrics(r[r.index>=OOS_S])
    full_m = metrics(r)
    gap = abs(is_m["sharpe"]-oos_m["sharpe"])
    tim = float((st==1).mean())
    return dict(
        univ=len(universe), bh=breadth_hi, vxm=vix_max, cf=confirm,
        lb=mom_lb, sk=mom_skip, k=top_k, rb=rebal, ma=ma, hy=hy_win, tc=tc_bps,
        full_sr=full_m["sharpe"], full_cagr=full_m["cagr"], full_mdd=full_m["mdd"], full_vol=full_m["vol"], full_navx=full_m["navx"],
        is_sr=is_m["sharpe"], is_cagr=is_m["cagr"], is_mdd=is_m["mdd"],
        oos_sr=oos_m["sharpe"], oos_cagr=oos_m["cagr"], oos_mdd=oos_m["mdd"],
        gap=gap, tim=tim, turn_ann=turn.mean()*252,
    )

if __name__=="__main__":
    import itertools, sys
    rows = []
    # main grid
    grids = list(itertools.product(
        [("LEV_EQ",LEV_EQ), ("LEV_ALL",LEV_ALL)],
        [63, 126, 189, 252],     # mom_lb
        [5, 21],                 # mom_skip
        [1, 2, 3],               # top_k
        [5, 10, 21],             # rebal
        [0.50, 0.60, 0.70],      # breadth_hi
        [1, 3],                  # confirm
    ))
    print(f"configs: {len(grids)}")
    for i, ((uname, univ), lb, sk, k, rb, bh, cf) in enumerate(grids):
        res = eval_config(univ, bh, 25, cf, lb, sk, k, rb)
        res["uname"] = uname
        rows.append(res)
        if i%50==0:
            print(f"  {i}/{len(grids)}")
    df = pd.DataFrame(rows)
    df.to_csv("data/results/kraken_grid.csv", index=False)

    print("\n--- Top 15 by full_sr ---")
    print(df.sort_values("full_sr", ascending=False).head(15).to_string(index=False))

    mask = (df.full_sr>=2.0)&(df.full_cagr>=0.20)&(df.is_sr>=1.5)&(df.oos_sr>=1.5)&(df.gap<=0.5)
    qual = df[mask].sort_values("full_sr", ascending=False)
    print(f"\n--- Qualifying (hard reqs): {len(qual)} ---")
    print(qual.head(30).to_string(index=False))

    mask2 = (df.full_sr>=1.5)&(df.is_sr>=1.2)&(df.oos_sr>=1.2)&(df.gap<=0.6)
    near = df[mask2].sort_values("full_sr", ascending=False)
    print(f"\n--- Near-qualifying (softer): {len(near)} ---")
    print(near.head(30).to_string(index=False))
