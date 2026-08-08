"""Exp25: factor-neutral stat-arb — the strongest untested lever.

Compares, for each headline signal: naive decile LS vs PCA-factor-neutral
optimized weights (5/10/20 statistical factors). If uncontrolled factor
exposure was hiding the alpha, this is where it shows up.

Clean era only (2010+), next-open execution, 5 bps.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt, riskmodel

t0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P0 = datalib.load_summit()
P = {k: v.loc["2009":] for k, v in P0.items()}
close, open_, member = P["close"], P["open"], P["member"]
r1 = close.pct_change(fill_method=None)
intraday = close / open_ - 1
overnight = open_ / close.shift(1) - 1

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

sigs = {
    "rev5d": -zs(close.pct_change(5, fill_method=None)),
    "rev_i5_sm3": -zs(intraday.rolling(5).sum()).rolling(3).mean(),
    "rev21d": -zs(close.pct_change(21, fill_method=None)),
    "onmom252": zs(overnight.rolling(252).mean() / overnight.rolling(252).std()),
    "mom12_1": zs(close.shift(21).pct_change(231, fill_method=None)),
}

out = {}
def ev(name, w, cost=5.0):
    res = bt.run(w, P, mode="open", cost_bps=cost)
    net = res["net"]
    m = bt.metrics(net)
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m["2010_2017"] = bt.sharpe(net.loc["2010":"2017"])
    m["2018_2026"] = bt.sharpe(net.loc["2018":])
    out[name] = m
    print(f"{name:34s} net={m['sharpe']:6.2f} gross={m['gross_sharpe']:6.2f} "
          f"10-17={m['2010_2017']:6.2f} 18-26={m['2018_2026']:6.2f} to={m['avg_turnover']:5.2f}", flush=True)

for nm, sig in sigs.items():
    ev(f"{nm}_naive_decile", bt.norm_ls(sig, member, 0.1, 0.1, 2.0))
    for K in (5, 10, 20):
        w = riskmodel.optimized_weights(sig, r1, member, n_factors=K)
        ev(f"{nm}_fneutral_K{K}", w)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

json.dump(out, open(os.path.join(ROOT, "results", "exp25_factor_neutral.json"), "w"), indent=1)
print("DONE", time.time() - t0, flush=True)
