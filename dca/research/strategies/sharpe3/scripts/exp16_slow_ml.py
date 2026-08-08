"""Last ML attempt: SLOW ML — 21-day-horizon labels, monthly rebalance,
patient execution. If ML skill exists at low-turnover horizons, costs stop
mattering (tno ~0.03/d -> drag <1%/yr). DEV2-oriented: train expanding,
predict 2005-2019; also report 2001-2014 for continuity.
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
idx = R.index
resid = pd.read_pickle("/tmp/sharpe3_work/_resid_daily.pkl")
E_d = ELIG.reindex(idx).ffill().fillna(False)[R.columns].astype(bool)
rvol = resid.rolling(63, min_periods=40).std().shift(1)
L = np.log1p(R)
F = {
 "z5": (resid.rolling(5).sum()/(rvol*np.sqrt(5))),
 "z21": (resid.rolling(21).sum()/(rvol*np.sqrt(21))),
 "z63": (resid.rolling(63).sum()/(rvol*np.sqrt(63))),
 "mom": np.expm1(L.rolling(231).sum().shift(21)),
 "r63": np.expm1(L.rolling(63).sum()),
 "vol63": R.rolling(63, min_periods=40).std(),
 "vr": R.rolling(21).std()/R.rolling(63, min_periods=40).std(),
 "dvr": DV.rolling(21).mean()/DV.rolling(126, min_periods=80).mean(),
 "dvrk": DV.rolling(63, min_periods=40).median().rank(axis=1, pct=True),
 "hi52": PX/PX.rolling(252, min_periods=200).max(),
 "skw": R.rolling(126, min_periods=80).skew(),
}
LBL = (resid.shift(-23).rolling(21).sum()/(rvol*np.sqrt(21)))   # days t+2..t+23
mo = [d for d in G.month_ends(idx) if pd.Timestamp("1997-06-01") < d <= pd.Timestamp("2019-12-31")]
E_mo = G.elig_on(mo, ELIG)
Xs, ys, metas = [], [], []
for d in mo:
    e = E_mo.loc[d]; names = e[e].index.intersection(R.columns)
    X = pd.DataFrame({k: F[k].loc[:d].iloc[-1][names] for k in F})
    y = LBL.loc[d][names] if d in LBL.index else pd.Series(np.nan, index=names)
    ok = X.notna().sum(axis=1) >= 8
    Xs.append(X[ok]); ys.append(y[ok.index[ok]])
    metas.append(pd.DataFrame({"date": d, "ticker": X[ok].index}))
XA = pd.concat(Xs); yA = pd.concat(ys); MA = pd.concat(metas); MA.index = XA.index
dts = MA["date"].values
print(f"frames ready {XA.shape} t={time.time()-t0:.0f}s", flush=True)

pred = {}
for year in range(2001, 2020):
    tr = dts <= np.datetime64(pd.Timestamp(f"{year-1}-11-01"))
    te = (dts >= np.datetime64(pd.Timestamp(f"{year}-01-01"))) & (dts <= np.datetime64(pd.Timestamp(f"{year}-12-31")))
    ytr = yA.values[tr]; fin = np.isfinite(ytr)
    if fin.sum() < 5000: continue
    m = HistGradientBoostingRegressor(max_iter=120, max_depth=5, learning_rate=0.07, l2_regularization=1.0, random_state=0)
    m.fit(XA.values[tr][fin], np.clip(ytr[fin], -5, 5))
    p = m.predict(XA.values[te])
    sub = MA[te].copy(); sub["p"] = p
    for d, g in sub.groupby("date"):
        pred[d] = pd.Series(g["p"].values, index=g["ticker"].values)
print(f"predictions t={time.time()-t0:.0f}s", flush=True)

ics = []
for d, p in pred.items():
    y = LBL.loc[d].reindex(p.index)
    if y.notna().sum() > 100: ics.append((d, p.corr(y, method="spearman")))
ic = pd.Series(dict(ics))
print(f"rank IC: full {ic.mean():.4f} (t={ic.mean()/ic.std()*np.sqrt(len(ic)):.1f}); "
      f"2005-19 {ic['2005':'2019'].mean():.4f} (t={ic['2005':'2019'].mean()/ic['2005':'2019'].std()*np.sqrt(len(ic['2005':'2019'])):.1f})", flush=True)

rows = []
for d, p in sorted(pred.items()):
    s = p.dropna().clip(-3, 3)
    z = (s - s.mean())/(s.std()+1e-12)
    pos = z[z > 0.8]; neg = z[z < -0.8]
    if len(pos) < 20 or len(neg) < 20: continue
    w = pd.concat([0.5*pos/pos.sum(), 0.5*neg/(-neg).sum()]); w.name = d
    rows.append(w)
Wt = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
Tv = Wt.values; Wv = np.zeros_like(Tv); cur = np.zeros(Tv.shape[1])
for k in range(len(Tv)):
    cur = cur + 0.5*(Tv[k]-cur); Wv[k] = cur
Wt = pd.DataFrame(Wv, index=Wt.index, columns=Wt.columns)
for a0, b0 in [("2001", "2014"), ("2005", "2019"), ("2015", "2019")]:
    line = f"slow-ml monthly {a0}-{b0}:"
    for fee in (0, 5, 10):
        net, g_, t_ = G.run(Wt, R, fee_bps=fee)
        line += f"  {fee}bp {G.sharpe(net[a0:b0]):5.2f}"
    line += f"  (tno {t_[a0:b0].mean():.3f}, gross {G.sharpe(g_[a0:b0]):.2f})"
    print(line, flush=True)
print(f"exp16 done t={time.time()-t0:.0f}s", flush=True)
