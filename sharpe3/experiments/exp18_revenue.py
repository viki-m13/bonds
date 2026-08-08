"""Exp18: quarterly revenue growth/acceleration factor (SEC XBRL data).

sec_revenue_quarterly is keyed by calendar quarter WITHOUT filing dates, so we
lag availability conservatively: quarter Q data usable only from 70 days after
quarter end. Signals: YoY growth, growth acceleration, revenue surprise vs
4q trend. Monthly rebalance, S&P 500 members, LS quintiles, 5 bps.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, member = P["close"], P["member"]
rev = pd.read_parquet(os.path.join(ROOT, "..", "dca", "research", "data", "sec", "sec_revenue_quarterly.parquet"))
qend = pd.PeriodIndex(rev.index.str.replace("CY", ""), freq="Q").to_timestamp(how="end").normalize()
rev.index = qend
rev = rev[[c for c in rev.columns if c in close.columns]]
print("revenue panel:", rev.shape)

yoy = rev / rev.shift(4) - 1
acc = yoy - yoy.shift(1)
trend = rev.rolling(4).mean()
surp = rev / trend.shift(1) - 1

def to_daily(q):
    d = q.copy()
    d.index = d.index + pd.Timedelta(days=70)   # availability lag
    d = d.reindex(close.index.union(d.index)).ffill().reindex(close.index)
    return d

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

sigs = {"rev_yoy": to_daily(yoy).clip(-1, 3), "rev_acc": to_daily(acc).clip(-2, 2), "rev_surp": to_daily(surp).clip(-1, 1)}
out = {}
for nm, sig in sigs.items():
    z = zs(sig.where(member))
    # monthly rebalance: hold signal constant within month
    z = z.groupby(close.index.to_period("M")).transform("first")
    w = bt.norm_ls(z, member, 0.2, 0.2, 2.0)
    res = bt.run(w, P, mode="open", cost_bps=5.0)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m.update(bt.is_oos(res["net"]))
    out[nm] = m
    print(f"{nm:10s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f}")
json.dump(out, open(os.path.join(ROOT, "results", "exp18_revenue.json"), "w"), indent=1)
print("saved")
