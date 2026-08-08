"""Exp24: classic distance-method pairs trading (Gatev-Goetzmann-Rouwenhorst)
on S&P 500 PIT members, clean era 2010+.

Monthly formation: 252d normalized prices, top-50 pairs by SSD (within sector).
Trading: open a pair when spread z > 2 (vs formation std), close on
convergence (z crosses 0), stop at 42d. Next-open execution, 5 bps/side.
"""
import os, sys, json
import itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P0 = datalib.load_summit()
P = {k: v.loc["2009":] for k, v in P0.items()}
close, open_, member = P["close"], P["open"], P["member"]
sectors = json.load(open(os.path.join(datalib.PIT, "sectors.json")))
dates = close.index
r_oo = open_.pct_change(fill_method=None)

month_starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
w = pd.DataFrame(0.0, index=dates, columns=close.columns)

for mi in range(13, len(month_starts) - 1):
    t0 = month_starts.iloc[mi]
    i0 = dates.get_loc(t0)
    if i0 < 260:
        continue
    form = close.iloc[i0 - 252:i0]
    elig = member.iloc[i0 - 1] & form.notna().all() & (form.iloc[0] > 0)
    cols = [c for c in close.columns[elig] ]
    norm = form[cols] / form[cols].iloc[0]
    # pair search within sector
    bysec = {}
    for c in cols:
        bysec.setdefault(sectors.get(c, "U"), []).append(c)
    cand = []
    for s, cs in bysec.items():
        if len(cs) < 2:
            continue
        arr = norm[cs].values
        for a_i, b_i in itertools.combinations(range(len(cs)), 2):
            ssd = float(np.mean((arr[:, a_i] - arr[:, b_i]) ** 2))
            cand.append((ssd, cs[a_i], cs[b_i]))
    cand.sort()
    pairs = cand[:50]
    # trade this month: daily z of spread vs formation stats
    i1 = dates.get_loc(month_starts.iloc[mi + 1])
    for ssd, a, b in pairs:
        na = close[a].iloc[i0 - 252:i1] / form[a].iloc[0]
        nb = close[b].iloc[i0 - 252:i1] / form[b].iloc[0]
        sp = (na - nb)
        mu, sd = sp.iloc[:252].mean(), sp.iloc[:252].std()
        if sd == 0 or np.isnan(sd):
            continue
        z = (sp - mu) / sd
        pos = 0
        for j in range(252, len(z)):
            gi = i0 - 252 + j
            zj = z.iloc[j]
            if np.isnan(zj):
                pos = 0
                continue
            if pos == 0 and abs(zj) > 2:
                pos = -np.sign(zj)
                entry = j
            elif pos != 0 and (np.sign(zj) != -pos or j - entry > 42):
                pos = 0
            if pos != 0 and gi < len(dates):
                w.iloc[gi, w.columns.get_loc(a)] += pos * 0.02
                w.iloc[gi, w.columns.get_loc(b)] -= pos * 0.02

g = w.abs().sum(axis=1)
print("avg gross exposure:", round(float(g.mean()), 3), "active frac:", round(float((g > 0).mean()), 3))
res = bt.run(w, P, mode="open", cost_bps=5.0)
net, gross = res["net"], res["gross"]
# scale to unit gross for Sharpe readability (Sharpe invariant anyway)
m = bt.metrics(net)
m["gross_sharpe"] = bt.sharpe(gross)
m["oos19"] = bt.sharpe(net[net.index >= "2019-01-01"])
m["yearly"] = bt.yearly_sharpes(net)
print("pairs:", json.dumps(m, default=str))
json.dump(m, open(os.path.join(ROOT, "results", "exp24_pairs.json"), "w"), indent=1, default=str)
