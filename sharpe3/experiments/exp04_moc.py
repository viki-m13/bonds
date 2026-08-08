"""Exp04: same-close (MOC) execution for short-horizon reversal + vol regime
conditioning + vol targeting. Compare against next-open to size the timing cost.
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
intraday = close / open_ - 1

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

sigs = {
    "rev1d": -zs(r1 / vol20),
    "rev_intraday1d": -zs(intraday / vol20),
    "rev_intraday5d": -zs(intraday.rolling(5).sum()),
    "rev3d": -zs(close.pct_change(3, fill_method=None)),
    "rev5d": -zs(close.pct_change(5, fill_method=None)),
}

out = {}
def ev(name, w, mode, cost=5.0, extra=None):
    res = bt.run(w, P, mode=mode, cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    if extra: m.update(extra)
    out[name] = m
    print(f"{name:30s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}")
    return res

for name, raw in sigs.items():
    w = bt.norm_ls(raw, member, 0.1, 0.1, 2.0)
    ev(name + "_open", w, "open")
    ev(name + "_moc", w, "sameclose")

# vol-regime conditioning on best signal: scale by market vol state
best = -zs(intraday.rolling(5).sum())
w = bt.norm_ls(best, member, 0.1, 0.1, 2.0)
mkt = r1.where(member).mean(axis=1)
mv = mkt.rolling(20).std() * np.sqrt(252)
state_hi = (mv > mv.rolling(252, min_periods=60).median()).astype(float)
ev("rev_i5_moc_hivol", w.mul(state_hi, axis=0), "sameclose")
ev("rev_i5_moc_lovol", w.mul(1 - state_hi, axis=0), "sameclose")

# vol targeting: scale to constant 10% ann vol using trailing 60d strat vol
res = bt.run(w, P, mode="sameclose", cost_bps=5.0)
sv = res["net"].rolling(60).std() * np.sqrt(252)
lev = (0.10 / sv).clip(0.2, 3.0).shift(1)
ev("rev_i5_moc_voltarget", w.mul(lev, axis=0), "sameclose")

with open(os.path.join(ROOT, "results", "exp04_moc.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
