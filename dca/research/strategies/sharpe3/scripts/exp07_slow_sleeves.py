"""Invention track F: slow, cheap sleeves to diversify the ensemble.
All monthly rebalance (turnover ~0.07/day max), DEV only:
  1. 52-week-high proximity (George-Hwang momentum variant)
  2. same-calendar-month seasonality (Heston-Sadka lite: avg return of this
     calendar month over past 5 years)
  3. long-term reversal (36m past return, skip 1m)
  4. peer-spillover momentum: corr-weighted peer 21d return (Moskowitz-flavor)
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
DEV_END = pd.Timestamp("2014-12-31")
idx = R.index
mo = [d for d in G.month_ends(idx) if pd.Timestamp("1995-01-01") < d <= DEV_END]
E_mo = G.elig_on(mo, ELIG)
L = np.log1p(R)

def run_monthly(name, sigfn, invert=False, q=0.15):
    rows = []
    for d in mo:
        s = sigfn(d)
        if s is None: continue
        e = E_mo.loc[d]
        s = s[e[e].index.intersection(s.index)].dropna()
        if len(s) < 100: continue
        if invert: s = -s
        w = pd.Series(G.normalize_ls(s, q, q)); w.name = d
        rows.append(w)
    W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
    net, gross, tno = G.run(W, R)
    rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], name)
    print(G.fmt(rep), flush=True)
    net[:DEV_END].to_pickle(f"/tmp/sharpe3_work/sleeve_{name.split()[0]}.pkl")
    return rep

# 1) 52w high proximity
roll_max = PX.rolling(252, min_periods=200).max()
prox = PX / roll_max
run_monthly("hi52 proximity", lambda d: prox.loc[:d].iloc[-1])

# 2) same-month seasonality: mean of same calendar month monthly returns, past 5y
Lm = L.resample("ME").sum()
mret = np.expm1(Lm)
def seas(d):
    m = pd.Timestamp(d).month
    hist = mret[(mret.index.month == m) & (mret.index < pd.Timestamp(d) - pd.DateOffset(days=20))].tail(5)
    if len(hist) < 3: return None
    return hist.mean()
run_monthly("seasonal same-month", seas)

# 3) long-term reversal 36m skip 1m
r36 = np.expm1(L.rolling(252*3).sum().shift(21))
run_monthly("ltrev 36m", lambda d: r36.loc[:d].iloc[-1], invert=True)

# 4) peer spillover: corr-weighted peer r21, refreshed quarterly
r21 = np.expm1(L.rolling(21).sum())
qtr = mo[::3]
peer_cache = {}
def peers_for(d):
    qs = [q_ for q_ in qtr if q_ <= d]
    if not qs: return None
    q_ = qs[-1]
    if q_ not in peer_cache:
        i = idx.get_loc(q_)
        e = E_mo.loc[min([m for m in mo if m >= q_], default=mo[-1])] if q_ not in E_mo.index else E_mo.loc[q_]
        names = e[e].index
        Rw = R.iloc[i-252:i+1][names]
        good = Rw.columns[Rw.notna().sum() > 230]
        Rw = Rw[good].fillna(0.0)
        C = np.corrcoef(Rw.values.T)
        np.fill_diagonal(C, -1)
        topk = np.argsort(-C, axis=1)[:, :20]
        peer_cache[q_] = (list(good), topk)
    return peer_cache[q_]
def spill(d):
    pc = peers_for(d)
    if pc is None: return None
    good, topk = pc
    r = r21.loc[:d].iloc[-1][good].fillna(0.0).values
    peer_r = r[topk].mean(axis=1)
    return pd.Series(peer_r - 0.0, index=good)
run_monthly("peerspill 21d qtr", spill)
print(f"exp07 done t={time.time()-t0:.0f}s", flush=True)
