"""Exp19: (a) Heston-Sadka same-month seasonality; (b) gross profitability
(Novy-Marx GP/Assets) from SEC XBRL fundamentals with 70d lag. Monthly rebal.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, member = P["close"], P["member"]
r1 = close.pct_change(fill_method=None)

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

out = {}
def ev(name, w, cost=5.0):
    res = bt.run(w, P, mode="open", cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m.update(bt.is_oos(res["net"]))
    out[name] = m
    print(f"{name:22s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f}")

# (a) same-month seasonality: avg return in same calendar month over past 5y
mret = close.resample("ME").last().pct_change(fill_method=None)
sea = sum(mret.shift(12 * k) for k in range(1, 6)) / 5
sead = sea.reindex(close.index.union(mret.index)).ffill(limit=25).reindex(close.index)
z = zs(sead.where(member)).groupby(close.index.to_period("M")).transform("first")
ev("seasonality_5y", bt.norm_ls(z, member, 0.2, 0.2, 2.0))

# (b) gross profitability
f = pd.read_pickle(os.path.join(ROOT, "..", "dca", "research", "data", "sec", "sec_fundamentals.pkl"))
gp, assets = f.get("GrossProfit"), f.get("Assets")
print("GP type:", type(gp))
if isinstance(gp, pd.DataFrame):
    for df_ in (gp, assets):
        print(df_.shape, df_.index[:2])
    qend = pd.PeriodIndex(gp.index.str.replace("CY", ""), freq="Q").to_timestamp(how="end").normalize()
    gp2 = gp.copy(); gp2.index = qend
    a2 = assets.copy()
    a2.index = pd.PeriodIndex(assets.index.str.replace("CY", ""), freq="Q").to_timestamp(how="end").normalize()
    gpa = (gp2.rolling(4).sum() / a2).replace([np.inf, -np.inf], np.nan)
    gpa = gpa[[c for c in gpa.columns if c in close.columns]]
    gpa.index = gpa.index + pd.Timedelta(days=70)
    gpad = gpa.reindex(close.index.union(gpa.index)).ffill().reindex(close.index)
    z = zs(gpad.where(member)).groupby(close.index.to_period("M")).transform("first")
    ev("gross_profitability", bt.norm_ls(z, member, 0.2, 0.2, 2.0))

json.dump(out, open(os.path.join(ROOT, "results", "exp19_seasonality_quality.json"), "w"), indent=1)
print("saved")
