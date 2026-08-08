"""Exp16: conditional liquidity provision — reversal only when it should pay.

Hypothesis: cross-sectional reversal = compensation for providing liquidity;
concentrated after market stress days. Activate the reversal book only on
trigger states; flat otherwise (episodic, low turnover when off).
Triggers: mkt 1d ret < -1%/-2%; mkt vol high; cross-sectional dispersion high;
combinations. Signal: 5d intraday-cum reversal (best family). Next-open exec.
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
intraday = close / open_ - 1
vol20 = r1.rolling(20).std()
mkt = r1.where(member).mean(axis=1)

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

sig = -zs(intraday.rolling(5).sum())
w_base = bt.norm_ls(sig, member, 0.1, 0.1, 2.0)

mv = mkt.rolling(20).std() * np.sqrt(252)
mv_med = mv.rolling(252, min_periods=60).median()
disp = r1.where(member).std(axis=1)
disp_med = disp.rolling(252, min_periods=60).median()

triggers = {
    "always": pd.Series(1.0, index=close.index),
    "mkt_dn1": (mkt < -0.01).astype(float),
    "mkt_dn2": (mkt < -0.02).astype(float),
    "mkt_dn1_2d": (mkt.rolling(2).sum() < -0.015).astype(float),
    "hivol": (mv > mv_med).astype(float),
    "hidisp": (disp > disp_med * 1.25).astype(float),
    "dn1_or_hidisp": ((mkt < -0.01) | (disp > disp_med * 1.25)).astype(float),
    "dn1_and_hivol": ((mkt < -0.01) & (mv > mv_med)).astype(float),
}

out = {}
for nm, trig in triggers.items():
    w = w_base.mul(trig, axis=0)
    res = bt.run(w, P, mode="open", cost_bps=5.0)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["active_frac"] = round(float((trig > 0).mean()), 3)
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    m["OOS22"] = bt.sharpe(res["net"][res["net"].index >= "2022-01-01"])
    m["yearly"] = bt.yearly_sharpes(res["net"])
    out[nm] = m
    print(f"{nm:16s} netSR={m['sharpe']:6.2f} IS={m['IS']:6.2f} OOS19={m['OOS']:6.2f} OOS22={m['OOS22']:6.2f} active={m['active_frac']:.2f}")

json.dump(out, open(os.path.join(ROOT, "results", "exp16_liquidity_provision.json"), "w"), indent=1, default=str)
print("saved")
