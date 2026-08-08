"""Exp01: baseline signal families on the S&P 500 PIT panel.

Long-short decile portfolios, next-open execution, 5 bps/side.
Writes results to sharpe3/results/exp01_baselines.json.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

P = datalib.load_summit()
close, open_, vol, member = P["close"], P["open"], P["volume"], P["member"]
r1 = close.pct_change(fill_method=None)
overnight = open_ / close.shift(1) - 1
intraday = close / open_ - 1

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

vol20 = r1.rolling(20).std()

signals = {
    "rev1d": -zs(r1),
    "rev1d_volscaled": -zs(r1 / vol20),
    "rev3d": -zs(close.pct_change(3, fill_method=None)),
    "rev5d": -zs(close.pct_change(5, fill_method=None)),
    "rev10d": -zs(close.pct_change(10, fill_method=None)),
    "rev21d": -zs(close.pct_change(21, fill_method=None)),
    "mom12_1": zs(close.shift(21).pct_change(231, fill_method=None)),
    "mom6_1": zs(close.shift(21).pct_change(105, fill_method=None)),
    "rev_intraday1d": -zs(intraday),
    "rev_overnight1d": -zs(overnight),
    "mom_overnight21d": zs(overnight.rolling(21).sum()),
    "mom_overnight60d": zs(overnight.rolling(60).sum()),
    "rev_intraday5d": -zs(intraday.rolling(5).sum()),
    "lowvol60": -zs(r1.rolling(60).std()),
    "volume_spike": -zs((vol / vol.rolling(20).median() - 1)),
    "amihud21": zs((r1.abs() / (close * vol)).rolling(21).mean()),
}

out = {}
for name, raw in signals.items():
    for frac in (0.1,):
        w = bt.norm_ls(raw, member, long_frac=frac, short_frac=frac, gross=2.0)
        res = bt.run(w, P, mode="open", cost_bps=5.0)
        m = bt.metrics(res["net"])
        m["gross_sharpe"] = bt.sharpe(res["gross"])
        m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
        m.update(bt.is_oos(res["net"]))
        out[name] = m
        print(f"{name:22s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} "
              f"IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f} dd={m['maxdd']:.2f}")

os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp01_baselines.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
