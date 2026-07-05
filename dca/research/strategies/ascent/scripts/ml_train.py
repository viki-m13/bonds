"""ASCENT ML score — faithful walk-forward HistGBM (exp78 spec):
fwd-12m tercile target (top vs bottom), 36 rank features, yearly refits.
Predictions for year Y trained only on data with COMPLETE fwd-12m labels
strictly before Y (label embargo: training rows need date <= Y-1yr so the
forward window doesn't overlap the test year).
Output: scratchpad/_mlprob.pkl (monthly prob panel).
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, fok, liq, me, cols = D["FEAT"], D["fok"], D["liq"], D["me"], D["cols"]
M = me.index
fnames = list(FEAT.keys())
Z = {nm: FEAT[nm].where(liq).rank(axis=1, pct=True).astype(np.float32) for nm in fnames}
p(f"ranked t={time.time()-t0:.0f}s")

recs = []
for dt in M[(M >= pd.Timestamp("2011-06-01"))]:
    fv = fok.loc[dt].dropna()
    if len(fv) < 60: continue
    q1, q2 = fv.quantile(1/3), fv.quantile(2/3)
    y = pd.Series(np.where(fv >= q2, 1, np.where(fv <= q1, 0, np.nan)), index=fv.index).dropna()
    X = np.column_stack([Z[nm].loc[dt].reindex(y.index).values for nm in fnames])
    for i, tk in enumerate(y.index):
        recs.append((dt, tk, *X[i], int(y.iloc[i])))
DF = pd.DataFrame.from_records(recs, columns=["date", "tk"] + fnames + ["y"])
p(f"samples {len(DF)} t={time.time()-t0:.0f}s")

from sklearn.ensemble import HistGradientBoostingClassifier
pr = []
for ytest in range(2015, 2027):
    # label embargo: training months must have their full 12m forward window
    # complete before the test year starts
    tr = DF[DF.date < pd.Timestamp(f"{ytest-1}-01-01")]
    te_all = M[(M >= pd.Timestamp(f"{ytest}-01-01")) & (M <= pd.Timestamp(f"{ytest}-12-31"))]
    if len(te_all) == 0 or len(tr) < 5000: continue
    clf = HistGradientBoostingClassifier(max_iter=250, max_depth=4, learning_rate=0.05,
                                         l2_regularization=1.0, min_samples_leaf=200)
    clf.fit(tr[fnames].values, tr["y"].values)
    # predict EVERY liquid name each test month (not only labeled rows)
    for dt in te_all:
        ok = liq.loc[dt]
        idx = ok[ok].index
        X = np.column_stack([Z[nm].loc[dt].reindex(idx).values for nm in fnames])
        good = ~np.isnan(X).all(axis=1)
        if good.sum() == 0: continue
        Xg = np.nan_to_num(X[good], nan=0.5)
        probs = clf.predict_proba(Xg)[:, 1]
        pr.append(pd.DataFrame({"date": dt, "tk": idx[good], "p": probs}))
    p(f"  trained <{ytest-1}, predicted {ytest} (tr={len(tr)}) t={time.time()-t0:.0f}s")
PROB = pd.concat(pr).pivot_table(index="date", columns="tk", values="p").reindex(M)
pd.to_pickle(PROB.astype(np.float32), f"{HERE}/_mlprob.pkl")
p(f"saved _mlprob.pkl {PROB.shape} DONE t={time.time()-t0:.0f}s")
