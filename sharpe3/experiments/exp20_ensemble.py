"""Exp20: the ensemble. All economically-motivated sleeves, defined a priori
(no OOS-peeking selection), combined at equal risk (inverse trailing 60d vol),
then vol-targeted to 10% ann. Reports full-sample/IS/OOS, correlations, and an
explicitly-labeled ORACLE upper bound (best-3-by-OOS, in-sample-selected).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import bt, sleeves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = sleeves.build_all()
print("sleeves:", df.columns.tolist())
print("corr:\n", df.corr().round(2).to_string())

out = {"sleeves": {}, "corr": df.corr().round(3).to_dict()}
for c in df.columns:
    m = bt.metrics(df[c])
    m.update(bt.is_oos(df[c]))
    m["OOS22"] = bt.sharpe(df[c][df[c].index >= "2022-01-01"])
    m["yearly"] = bt.yearly_sharpes(df[c])
    out["sleeves"][c] = m
    print(f"{c:16s} SR={m['sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} OOS22={m['OOS22']}")

# equal-risk combo (trailing 60d vol, shifted)
sv = df.rolling(60).std().shift(1)
wts = (1 / sv).div((1 / sv).sum(axis=1), axis=0)
combo = (df * wts).sum(axis=1)
# vol-target 10%
cv = combo.rolling(60).std().shift(1) * np.sqrt(252)
lev = (0.10 / cv).clip(0.2, 5.0)
combo_vt = combo * lev

for nm, r in (("combo_eqrisk", combo), ("combo_voltarget", combo_vt)):
    m = bt.metrics(r)
    m.update(bt.is_oos(r))
    m["OOS22"] = bt.sharpe(r[r.index >= "2022-01-01"])
    m["yearly"] = bt.yearly_sharpes(r)
    out[nm] = m
    print(f"{nm:16s} SR={m['sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} OOS22={m['OOS22']} yearly={m['yearly']}")

# ORACLE (upper bound, selection bias by construction): best 3 sleeves by OOS
oos_sr = {c: bt.sharpe(df[c][df[c].index >= "2019-01-01"]) for c in df.columns}
top3 = sorted(oos_sr, key=oos_sr.get, reverse=True)[:3]
orc = df[top3].mean(axis=1)
m = bt.metrics(orc); m.update(bt.is_oos(orc))
out["oracle_top3_by_oos"] = {"names": top3, **m}
print("ORACLE top3", top3, "SR", m["sharpe"], "OOS", m["OOS"])

json.dump(out, open(os.path.join(ROOT, "results", "exp20_ensemble.json"), "w"), indent=1, default=str)
combo_vt.to_frame("ret").to_parquet(os.path.join(ROOT, "cache", "ensemble_returns.parquet"))
print("saved")
