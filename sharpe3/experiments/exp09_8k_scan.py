"""Exp09: event-study scan over 8-K item types on the S&P 500 PIT panel.

For each item code: mean market-adjusted forward return (1/5/21d) from the
first close AFTER filing (conservative), split IS (<2019) / OOS, with t-stats.
Scan only; no portfolio yet.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, member = P["close"], P["member"]
r1 = close.pct_change(fill_method=None)
mkt = r1.where(member).mean(axis=1)
adj = r1.sub(mkt, axis=0)
dates = close.index

# forward CAR matrices from t+1 (t = first trading day >= filing date)
car = {h: adj.shift(-h).rolling(h).sum().shift(-1) for h in (1, 5, 21)}  # sum of adj over [t+1, t+h]

ek = pd.read_parquet(os.path.join(ROOT, "..", "dca", "research", "data", "sec", "8k_items.parquet"))
ek["date"] = pd.to_datetime(ek["date"])
ek = ek[ek.tk.isin(close.columns)]
ek["items"] = ek["items"].fillna("")

codes = ["1.01","1.02","1.03","2.01","2.02","2.03","2.05","2.06","3.01","4.01","4.02","5.01","5.02","5.03","5.07","7.01","8.01"]
res = {}
tkloc = {t: i for i, t in enumerate(close.columns)}
for code in codes:
    sub = ek[ek["items"].str.contains(code.replace(".", r"\."), regex=True)]
    rows = {"n": 0}
    vals = {h: {"IS": [], "OOS": []} for h in (1, 5, 21)}
    ti = dates.searchsorted(sub["date"].values)
    ok = ti < len(dates) - 25
    ti = ti[ok]; tks = sub["tk"].values[ok]
    for h in (1, 5, 21):
        m = car[h].values
        v = m[ti, [tkloc[t] for t in tks]]
        isv = dates[ti] < pd.Timestamp("2019-01-01")
        vals[h]["IS"] = v[isv & ~np.isnan(v)]
        vals[h]["OOS"] = v[~isv & ~np.isnan(v)]
    rows["n"] = int(len(ti))
    for h in (1, 5, 21):
        for p in ("IS", "OOS"):
            v = vals[h][p]
            if len(v) > 30:
                rows[f"{p}_car{h}_bps"] = round(float(np.mean(v)) * 1e4, 1)
                rows[f"{p}_t{h}"] = round(float(np.mean(v) / np.std(v) * np.sqrt(len(v))), 1)
    res[code] = rows
    print(code, rows, flush=True)

with open(os.path.join(ROOT, "results", "exp09_8k_scan.json"), "w") as f:
    json.dump(res, f, indent=1)
print("saved")
