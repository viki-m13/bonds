"""Invention track I: cross-sectional ML on price/volume features.

Weekly walk-forward, expanding window, retrained yearly (Jan 1 boundary uses
only prior data). HistGradientBoostingRegressor -> predicted next-week residual
return. Books: decile L-S, top50/bottom50, and a patient variant.

Honesty: features at week-end t use data through t; label = residual return
over days t+2..t+6 (respecting the execution lag); training set at prediction
date d contains only weeks whose labels completed before d.
DEV only (train from 1996, predict 2001-2014).
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
DEV_END = pd.Timestamp("2014-12-31")
idx = R.index
resid = pd.read_pickle("/tmp/sharpe3_work/_resid_daily.pkl")
E_d = ELIG.reindex(idx).ffill().fillna(False)[R.columns].astype(bool)
rvol = resid.rolling(63, min_periods=40).std().shift(1)
L = np.log1p(R)

F = {}
F["z1"] = (resid / rvol)
F["z5"] = (resid.rolling(5).sum() / (rvol*np.sqrt(5)))
F["z21"] = (resid.rolling(21).sum() / (rvol*np.sqrt(21)))
F["r5"] = np.expm1(L.rolling(5).sum())
F["r21"] = np.expm1(L.rolling(21).sum())
F["r63"] = np.expm1(L.rolling(63).sum())
F["mom"] = np.expm1(L.rolling(231).sum().shift(21))
F["vr"] = R.rolling(21).std() / R.rolling(63, min_periods=40).std()
F["vol63"] = R.rolling(63, min_periods=40).std()
F["dvr"] = DV.rolling(5).mean() / DV.rolling(63, min_periods=40).mean()
F["dvrk"] = DV.rolling(63, min_periods=40).median().rank(axis=1, pct=True)
F["hi52"] = PX / PX.rolling(252, min_periods=200).max()
F["mx21"] = R.rolling(21).max()
F["skw"] = R.rolling(63, min_periods=40).skew()
feat_names = list(F)
print(f"features ready t={time.time()-t0:.0f}s", flush=True)

# label: residual sum t+2..t+6, vol-scaled (scale-free target)
LBL = (resid.shift(-6).rolling(5).sum() / (rvol*np.sqrt(5)))   # aligned so row t = days t+2..t+6

wk = [d for d in G.week_ends(idx) if pd.Timestamp("1996-06-01") < d <= DEV_END]
E_wk = G.elig_on(wk, ELIG)

# assemble per-week frames once
rows_X, rows_y, rows_meta = [], [], []
for d in wk:
    e = E_wk.loc[d]; names = e[e].index.intersection(R.columns)
    X = pd.DataFrame({k: F[k].loc[:d].iloc[-1][names] for k in feat_names})
    y = LBL.loc[d][names] if d in LBL.index else pd.Series(np.nan, index=names)
    ok = X.notna().sum(axis=1) >= 10
    X = X[ok]; y = y[ok.index[ok]]
    rows_X.append(X); rows_y.append(y)
    rows_meta.append(pd.DataFrame({"date": d, "ticker": X.index}))
print(f"frames ready t={time.time()-t0:.0f}s", flush=True)

XA = pd.concat(rows_X); yA = pd.concat(rows_y)
MA = pd.concat(rows_meta); MA.index = XA.index
dates_arr = MA["date"].values

pred = {}
for year in range(2001, 2015):
    tr_end = pd.Timestamp(f"{year-1}-12-15")   # labels complete: signal <= mid-Dec
    te_a, te_b = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    tr = dates_arr <= np.datetime64(tr_end)
    te = (dates_arr >= np.datetime64(te_a)) & (dates_arr <= np.datetime64(te_b))
    ytr = yA.values[tr]
    fin = np.isfinite(ytr)
    m = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.06,
                                      l2_regularization=1.0, random_state=0)
    m.fit(XA.values[tr][fin], np.clip(ytr[fin], -5, 5))
    p = m.predict(XA.values[te])
    sub = MA[te].copy(); sub["p"] = p
    for d, g in sub.groupby("date"):
        pred[d] = pd.Series(g["p"].values, index=g["ticker"].values)
    print(f"{year}: trained on {int(fin.sum())} rows t={time.time()-t0:.0f}s", flush=True)

# rank IC diagnostic
ics = []
for d, p in pred.items():
    y = LBL.loc[d].reindex(p.index)
    if y.notna().sum() > 100:
        ics.append(p.corr(y, method="spearman"))
ics = pd.Series(ics)
print(f"rank IC: mean {ics.mean():.4f}  t={ics.mean()/ics.std()*np.sqrt(len(ics)):.1f}", flush=True)

def book(name, weight_fn):
    rows = []
    for d, p in sorted(pred.items()):
        w = weight_fn(p.dropna())
        if w is None: continue
        w.name = d; rows.append(w)
    W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
    net, gross, tno = G.run(W, R)
    rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], name)
    print(G.fmt(rep), flush=True)
    net[:DEV_END].to_pickle(f"/tmp/sharpe3_work/sleeve_{name.replace(' ','_')}.pkl")
    return rep

book("ml decile", lambda p: pd.Series(G.normalize_ls(p, 0.1, 0.1)) if len(p) > 200 else None)
def topk(p, K=50):
    if len(p) < 200: return None
    s = p.sort_values()
    sel = pd.concat([s.head(K)*0 - 0.5/K, s.tail(K)*0 + 0.5/K])
    return sel
book("ml top50", topk)
print(f"exp10 done t={time.time()-t0:.0f}s", flush=True)
