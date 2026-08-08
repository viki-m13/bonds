"""Exp23: definitive clean-era scorecard. Re-runs the headline signals on the
2010+ panel only (post artifact-era opens), full costs, next-open execution.
This is the table that summarizes what the modern market actually offers.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P0 = datalib.load_summit()
P = {k: v.loc["2010":] for k, v in P0.items()}
close, open_, volp, member = P["close"], P["open"], P["volume"], P["member"]
r1 = close.pct_change(fill_method=None)
intraday = close / open_ - 1
overnight = open_ / close.shift(1) - 1
vol20 = r1.rolling(20).std()

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

sigs = {
    "rev5d": -zs(close.pct_change(5, fill_method=None)),
    "rev_intraday5d_sm3": -zs(intraday.rolling(5).sum()).rolling(3).mean(),
    "onmom60": zs(overnight.rolling(60).mean() / overnight.rolling(60).std()),
    "onmom252": zs(overnight.rolling(252).mean() / overnight.rolling(252).std()),
    "mom12_1": zs(close.shift(21).pct_change(231, fill_method=None)),
    "lowvol60": -zs(r1.rolling(60).std()),
}

out = {}
for nm, sig in sigs.items():
    w = bt.norm_ls(sig, member, 0.1, 0.1, 2.0)
    res = bt.run(w, P, mode="open", cost_bps=5.0)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["2010_2015"] = bt.sharpe(res["net"].loc["2010":"2015"])
    m["2016_2020"] = bt.sharpe(res["net"].loc["2016":"2020"])
    m["2021_2026"] = bt.sharpe(res["net"].loc["2021":])
    out[nm] = m
    print(f"{nm:20s} net={m['sharpe']:6.2f} gross={m['gross_sharpe']:6.2f} "
          f"10-15={m['2010_2015']:6.2f} 16-20={m['2016_2020']:6.2f} 21-26={m['2021_2026']:6.2f}")

json.dump(out, open(os.path.join(ROOT, "results", "exp23_clean_era.json"), "w"), indent=1)
print("saved")
