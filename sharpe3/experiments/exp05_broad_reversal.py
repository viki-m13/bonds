"""Exp05: short-term reversal on the broad (delisting-inclusive) universe,
by liquidity tier. Execution: NEXT-CLOSE (signal close d -> trade close d+1),
fully causal, no MOC approximation. Cost sweep 10/20/40 bps.

Bid-ask-bounce guard: this repo's crypto work showed 1d-reversal "alpha" in
noisy prices is largely spread bounce. Tiers + vol-scaling + a no-1d variant
(skip the most recent day) let us see how much survives.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
close = pd.read_parquet(os.path.join(CACHE, "broad_close.parquet"))
dv = pd.read_parquet(os.path.join(CACHE, "broad_dollarvol.parquet"))
r1 = close.pct_change(fill_method=None)
# data-error guard: impossible daily moves -> NaN (delisted junk ticks)
r1 = r1.where(r1.abs() < 1.0)
vol20 = r1.rolling(20).std()

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

tiers = {
    "mid":   (dv >= 2e7) & (dv < 2e8),
    "small": (dv >= 5e6) & (dv < 2e7),
    "large": (dv >= 2e8),
}

rev5 = -zs(r1.rolling(5).sum() / (vol20 * np.sqrt(5)))
rev5_skip1 = -zs(r1.shift(1).rolling(4).sum() / (vol20 * 2))
rev1 = -zs(r1 / vol20)

panel = {"close": close, "open": close}  # open unused in close mode

out = {}
def ev(name, w, cost):
    res = bt.run(w, panel, mode="close", cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    out[name] = m
    print(f"{name:30s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}")

for tname, mask in tiers.items():
    member = mask & r1.notna()
    for sname, sig in (("rev5", rev5), ("rev5skip1", rev5_skip1), ("rev1", rev1)):
        w = bt.norm_ls(sig, member, 0.1, 0.1, 2.0)
        cost = {"large": 5, "mid": 10, "small": 20}[tname]
        ev(f"{tname}_{sname}_c{cost}", w, cost)

with open(os.path.join(ROOT, "results", "exp05_broad_reversal.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
