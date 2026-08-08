"""Exp10: calendar/flow effects on the S&P 500 PIT panel (as sleeves).

- Turn-of-month long tilt (last 4 + first 3 trading days) on equal-weight members
- Same with a high-beta/low-beta cross-sectional LS (flows hit high-beta harder)
- Pre-weekend / post-weekend overnight cross-sections
All next-open execution, 2 bps (index-like liquidity, MOC baskets).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, open_, member = P["close"], P["open"], P["member"]
r1 = close.pct_change(fill_method=None)
dates = close.index

# trading-day-of-month index
month = dates.to_period("M")
tdom = pd.Series(1, index=dates).groupby(month).cumsum()
tdom_rev = pd.Series(1, index=dates).iloc[::-1].groupby(month[::-1]).cumsum().iloc[::-1]
tom = ((tdom <= 3) | (tdom_rev <= 4)).astype(float)  # turn-of-month flag

mkt = r1.where(member).mean(axis=1)
cov = r1.rolling(60).cov(mkt); var = mkt.rolling(60).var()
beta = cov.div(var, axis=0)

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

ew = member.div(member.sum(axis=1), axis=0)

out = {}
def ev(name, w, cost=2.0):
    res = bt.run(w, P, mode="open", cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    out[name] = m
    print(f"{name:26s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}")

ev("tom_long", ew.mul(tom, axis=0))
ev("nontom_long", ew.mul(1 - tom, axis=0))
bz = zs(beta.where(member))
ev("tom_highbeta_ls", bt.norm_ls(bz, member, 0.2, 0.2, 2.0).mul(tom, axis=0))
dow = pd.Series(dates.dayofweek, index=dates)
for d, nm in ((0, "mon"), (4, "fri")):
    ev(f"{nm}_long", ew.mul((dow == d).astype(float), axis=0))

with open(os.path.join(ROOT, "results", "exp10_calendar.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
