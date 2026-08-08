"""Exp07: lead-lag effects on the S&P 500 PIT panel.

a) Sector 1d move -> next-day laggard members (buy members that lagged their
   sector's move today, short those that overshot) = intra-sector reversal-to-peers.
b) Big-brother/little-brother: return of the 5 largest-DV names in a sector
   today -> tomorrow's return of the smaller members (follow the leaders).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, open_, volp, member = P["close"], P["open"], P["volume"], P["member"]
r1 = close.pct_change(fill_method=None)
vol20 = r1.rolling(20).std()
dv = (close * volp).rolling(20, min_periods=5).median()
sectors = json.load(open(os.path.join(datalib.PIT, "sectors.json")))
sec = pd.Series({t: sectors.get(t, "Unknown") for t in close.columns})

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

# sector mean returns (member-weighted equal)
secret = {}
secret_big = {}
for s in sec.unique():
    cols = [c for c in sec.index[sec == s] if c in r1.columns]
    if len(cols) < 8:
        continue
    rs = r1[cols].where(member[cols])
    secret[s] = rs.mean(axis=1)
    # big-brother: top-5 DV names' mean return that day
    dvs = dv[cols].where(member[cols])
    rk = dvs.rank(axis=1, ascending=False)
    big = rs.where(rk <= 5)
    secret_big[s] = big.mean(axis=1)

# a) gap-to-sector: stock ret today minus sector ret today -> revert
gap = pd.DataFrame(np.nan, index=r1.index, columns=r1.columns, dtype=float)
follow = pd.DataFrame(np.nan, index=r1.index, columns=r1.columns, dtype=float)
for s, sr in secret.items():
    cols = [c for c in sec.index[sec == s] if c in r1.columns]
    gap[cols] = r1[cols].sub(sr, axis=0)
    # b) small members get the big-brother sector signal
    dvs = dv[cols].where(member[cols])
    rk = dvs.rank(axis=1, ascending=False)
    smallmask = rk > 5
    f = pd.DataFrame(np.tile(secret_big[s].values[:, None], (1, len(cols))),
                     index=r1.index, columns=cols)
    follow[cols] = f.where(smallmask)

out = {}
def ev(name, w, cost=5.0):
    res = bt.run(w, P, mode="open", cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    out[name] = m
    print(f"{name:26s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}")

ev("gap2sector_rev1d", bt.norm_ls(-zs(gap / vol20), member, 0.1, 0.1, 2.0))
ev("gap2sector_rev3d", bt.norm_ls(-zs(gap.rolling(3).sum() / vol20), member, 0.1, 0.1, 2.0))
ev("gap2sector_rev5d", bt.norm_ls(-zs(gap.rolling(5).sum() / vol20), member, 0.1, 0.1, 2.0))
ev("bigbrother_follow1d", bt.norm_ls(zs(follow), member, 0.1, 0.1, 2.0))
ev("bigbrother_follow5d", bt.norm_ls(zs(follow.rolling(5).sum()), member, 0.1, 0.1, 2.0))

with open(os.path.join(ROOT, "results", "exp07_leadlag.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
