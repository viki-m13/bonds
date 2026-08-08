"""Exp17: NDX-100 panel OHLC signals (the only stock panel with high/low).

Range-based: close position in daily range (5d), range compression breakout,
high-low vol (Parkinson) vs close-close vol ratio (informed trading proxy),
gap-fill tendencies. Next-open exec, 5bps, NDX members only (~100 names).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_n100()
close, open_, high, low, member = P["close"], P["open"], P["high"], P["low"], P["member"]
r1 = close.pct_change(fill_method=None)
vol20 = r1.rolling(20).std()

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

rng = (high - low) / close
rngpos = ((close - low) / (high - low).replace(0, np.nan) - 0.5)
park = (np.log(high / low) ** 2).rolling(21).mean()
cc = (r1 ** 2).rolling(21).mean()
pkr = park / cc.replace(0, np.nan)
gap = open_ / close.shift(1) - 1

sigs = {
    "rngpos5_rev": -zs(rngpos.rolling(5).mean()),
    "rngpos5_cont": zs(rngpos.rolling(5).mean()),
    "range_compress": -zs(rng.rolling(10).mean() / rng.rolling(63).mean()),
    "parkinson_ratio": zs(pkr),
    "parkinson_ratio_neg": -zs(pkr),
    "gapfade5": -zs(gap.rolling(5).sum()),
    "hl_rev5": -zs(((close - (high + low) / 2) / close).rolling(5).mean()),
}

out = {}
for nm, sig in sigs.items():
    w = bt.norm_ls(sig, member, 0.15, 0.15, 2.0)
    res = bt.run(w, P, mode="open", cost_bps=5.0)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m.update(bt.is_oos(res["net"]))
    out[nm] = m
    print(f"{nm:20s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f}")

json.dump(out, open(os.path.join(ROOT, "results", "exp17_n100_range.json"), "w"), indent=1)
print("saved")
