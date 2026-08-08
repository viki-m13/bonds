"""Exp12: S&P 500 reconstitution events from PIT membership.

Deletions: long deleted stocks for H days after removal (documented rebound).
Additions: short added stocks after inclusion (post-add reversion).
Prices from the broad Tiingo panel (delisting-inclusive — crucial: many
deletes are distressed). Event-study + episodic portfolio.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
close = pd.read_parquet(os.path.join(ROOT, "cache", "broad_close.parquet"))
r1 = close.pct_change(fill_method=None)
r1 = r1.where(r1.abs() < 1.0)

mem = pd.read_csv(os.path.join(ROOT, "..", "data", "pit", "sp500_pit_membership.csv"))
mem["date"] = pd.to_datetime(mem["date"])
mem = mem.sort_values("date")
sets = [set(s.split(",")) for s in mem["tickers"]]
adds, dels = [], []
for i in range(1, len(sets)):
    d = mem["date"].iloc[i]
    for t in sets[i] - sets[i - 1]:
        adds.append((d, t))
    for t in sets[i - 1] - sets[i]:
        dels.append((d, t))
adds = pd.DataFrame(adds, columns=["date", "tk"])
dels = pd.DataFrame(dels, columns=["date", "tk"])
print("adds:", len(adds), "dels:", len(dels))

dates = close.index
mkt = r1.mean(axis=1)  # broad market EW as adjuster

def event_car(events, hmax=126):
    """market-adjusted cumulative return paths after event date."""
    paths = []
    for d, tk in events.itertuples(index=False):
        if tk not in close.columns:
            continue
        i = dates.searchsorted(d)
        if i + 2 >= len(dates):
            continue
        seg = r1[tk].iloc[i + 1:i + 1 + hmax] - mkt.iloc[i + 1:i + 1 + hmax]
        if seg.notna().sum() < 10:
            continue
        v = seg.fillna(0).cumsum().values
        if len(v) < hmax:
            v = np.pad(v, (0, hmax - len(v)), constant_values=np.nan)
        paths.append((d, v))
    return paths

for nm, ev_ in (("dels", dels), ("adds", adds)):
    paths = event_car(ev_)
    arr = np.array([p[1] for p in paths])
    dts = pd.DatetimeIndex([p[0] for p in paths])
    for era, mask in (("pre2012", dts < "2012-01-01"),
                      ("2012-2019", (dts >= "2012-01-01") & (dts < "2019-01-01")),
                      ("2019+", dts >= "2019-01-01")):
        sub = arr[mask]
        if len(sub) < 10:
            continue
        for h in (21, 63, 126):
            v = sub[:, h - 1]
            v = v[~np.isnan(v)]
            print(f"{nm} {era:9s} n={len(v):4d} CAR{h}: {np.mean(v)*100:6.2f}% t={np.mean(v)/np.std(v)*np.sqrt(len(v)):5.1f}")

# episodic portfolio: long dels H=63, short adds H=63, eq-weight live events, market-hedged
H = 63
w = pd.DataFrame(0.0, index=dates, columns=close.columns)
for d, tk in dels.itertuples(index=False):
    if tk not in close.columns: continue
    i = dates.searchsorted(d)
    w.iloc[i + 1:i + 1 + H, w.columns.get_loc(tk)] += 1.0
for d, tk in adds.itertuples(index=False):
    if tk not in close.columns: continue
    i = dates.searchsorted(d)
    w.iloc[i + 1:i + 1 + H, w.columns.get_loc(tk)] -= 1.0
# normalize gross to 1 when any positions live
g = w.abs().sum(axis=1)
w = w.div(g.replace(0, np.nan), axis=0).fillna(0.0)
panel = {"close": close, "open": close}
res = bt.run(w, panel, mode="close", cost_bps=10.0)
net = res["net"]
m = bt.metrics(net); m.update(bt.is_oos(net))
m["yearly"] = bt.yearly_sharpes(net)
print("episodic LS:", json.dumps(m, default=str))
json.dump(m, open(os.path.join(ROOT, "results", "exp12_index_events.json"), "w"), indent=1)
