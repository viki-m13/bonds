"""Exp02: residualized / sector-neutral short-term reversal + smoothing.

Refinements over exp01:
  - residual returns vs rolling-beta market exposure
  - sector-neutral cross-sectional z-scores
  - signal smoothing (rolling mean of the signal) to cut turnover
  - vol-scaled weighting
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, open_, vol, member = P["close"], P["open"], P["volume"], P["member"]
r1 = close.pct_change(fill_method=None)
sectors = json.load(open(os.path.join(datalib.PIT, "sectors.json")))
sec = pd.Series({t: sectors.get(t, "Unknown") for t in close.columns})

mkt = r1.where(member).mean(axis=1)  # equal-weight member return

# rolling beta (60d) via cov/var
cov = r1.sub(0).rolling(60).cov(mkt)  # per-column rolling cov with mkt
var = mkt.rolling(60).var()
beta = cov.div(var, axis=0)
resid = r1.sub(beta.mul(mkt, axis=0))

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

def zs_sector(x):
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for s in sec.unique():
        cols = sec.index[sec == s]
        cols = [c for c in cols if c in x.columns]
        if len(cols) < 5:
            continue
        sub = x[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1), axis=0)
    return out

vol20 = r1.rolling(20).std()
intraday = close / open_ - 1

sigs = {
    "resid_rev3d": -zs(resid.rolling(3).sum()),
    "resid_rev5d": -zs(resid.rolling(5).sum()),
    "resid_rev10d": -zs(resid.rolling(10).sum()),
    "resid_rev5d_sect": -zs_sector(resid.rolling(5).sum()),
    "resid_rev5d_volscaled": -zs(resid.rolling(5).sum() / (vol20 * np.sqrt(5))),
    "rev_intraday5d_sect": -zs_sector(intraday.rolling(5).sum()),
    "resid_rev5d_sm3": -zs(resid.rolling(5).sum()).rolling(3).mean(),
    "resid_rev10d_sm5": -zs(resid.rolling(10).sum()).rolling(5).mean(),
    "rev_intraday5d_sm3": -zs(intraday.rolling(5).sum()).rolling(3).mean(),
}

out = {}
for name, raw in sigs.items():
    w = bt.norm_ls(raw, member, 0.1, 0.1, 2.0)
    res = bt.run(w, P, mode="open", cost_bps=5.0)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    m["yearly"] = bt.yearly_sharpes(res["net"])
    out[name] = m
    print(f"{name:24s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} "
          f"IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}")

with open(os.path.join(ROOT, "results", "exp02_residual_reversal.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
