"""Exp11: the overnight cross-section in depth.

The persistent overnight-return factor (Lou-Polk-Skouras): stocks with high
past overnight returns tend to keep earning overnight; intraday reverses.
Variants: different formations, holding sessions (full day vs overnight-only),
beta-neutral, yearly diagnostics.

Overnight-only execution (buy close, sell next open) is costed at 2x trades/day.
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
on = open_ / close.shift(1) - 1
ind = close / open_ - 1
dates = close.index

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

out = {}
def ev(name, w, mode="open", cost=5.0):
    res = bt.run(w, P, mode=mode, cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    m["yearly"] = bt.yearly_sharpes(res["net"])
    out[name] = m
    print(f"{name:28s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f}")
    print("   yearly:", m["yearly"])
    return res

for k in (21, 60, 126, 252):
    sig = zs(on.rolling(k).mean() / on.rolling(k).std())
    ev(f"onmom{k}", bt.norm_ls(sig, member, 0.1, 0.1, 2.0))

# overnight-session-only portfolio returns (manual): hold w overnight only
sig = zs(on.rolling(60).mean() / on.rolling(60).std())
w = bt.norm_ls(sig, member, 0.1, 0.1, 2.0)
# overnight ret of session d+1: on(d+1) = open(d+1)/close(d)-1; weights decided close d,
# entered at close d... needs sameclose-style entry: pos = w.shift(1) on 'on' returns
pos = w.shift(1).fillna(0.0)
gross = (pos * on).sum(axis=1)
dwn = w.diff().abs().sum(axis=1).fillna(0.0) * 2  # in and out each day
for cost in (1.0, 3.0, 5.0):
    net = gross - dwn * cost / 1e4
    m = bt.metrics(net)
    m.update(bt.is_oos(net))
    m["yearly"] = bt.yearly_sharpes(net)
    out[f"onmom60_overnightonly_c{cost}"] = m
    print(f"onmom60_ononly_c{cost}: net {m['sharpe']} IS {m['IS']} OOS {m['OOS']}")
    print("   yearly:", m["yearly"])

with open(os.path.join(ROOT, "results", "exp11_overnight.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
