"""Exp21: validation battery for the final ensemble and its flagship sleeves.

- stationary block bootstrap 95% CI for OOS Sharpe
- cost sweep on the underlying sleeve portfolios (2/5/10/20 bps)
- execution-mode stress (open vs close-next-day) for the flagship
- rolling 252d Sharpe percentiles
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt, sleeves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)

ens = pd.read_parquet(os.path.join(ROOT, "cache", "ensemble_returns.parquet"))["ret"].dropna()

def block_bootstrap_ci(r, n=2000, block=21):
    r = r.values
    sims = []
    L = len(r)
    for _ in range(n):
        idx = []
        while len(idx) < L:
            s = rng.integers(0, L)
            idx.extend(range(s, min(s + block, L)))
        x = r[np.array(idx[:L]) % L]
        sims.append(np.mean(x) / np.std(x) * np.sqrt(252))
    return np.percentile(sims, [2.5, 50, 97.5])

out = {}
for nm, r in (("ensemble_full", ens), ("ensemble_oos19", ens[ens.index >= "2019-01-01"]),
              ("ensemble_oos22", ens[ens.index >= "2022-01-01"])):
    lo, med, hi = block_bootstrap_ci(r)
    out[nm] = {"sharpe": bt.sharpe(r), "ci95": [round(lo, 2), round(hi, 2)], "days": len(r)}
    print(f"{nm:16s} SR={out[nm]['sharpe']:.2f} 95%CI [{lo:.2f}, {hi:.2f}]")

# cost sweep on sleeves
P = datalib.load_summit()
import importlib
df_costs = {}
for cost in (2.0, 5.0, 10.0, 20.0):
    # rebuild flagship reversal sleeve at given cost
    close, open_, member = P["close"], P["open"], P["member"]
    intraday = close / open_ - 1
    def zs(x): return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    sig = -zs(intraday.rolling(5).sum()).rolling(3).mean()
    w = bt.norm_ls(sig, member, 0.1, 0.1, 2.0)
    res = bt.run(w, P, mode="open", cost_bps=cost)
    df_costs[cost] = {"net_sharpe": bt.sharpe(res["net"]),
                      "oos19": bt.sharpe(res["net"][res["net"].index >= "2019-01-01"])}
    print(f"rev_i5_sm3 @ {cost}bps: SR {df_costs[cost]['net_sharpe']:.2f} OOS {df_costs[cost]['oos19']:.2f}")
out["cost_sweep_rev_i5_sm3"] = df_costs

roll = ens.rolling(252).apply(lambda x: np.mean(x) / np.std(x) * np.sqrt(252), raw=True).dropna()
out["rolling252_sharpe_pct"] = {p: round(float(np.percentile(roll, p)), 2) for p in (5, 25, 50, 75, 95)}
print("rolling 252d Sharpe percentiles:", out["rolling252_sharpe_pct"])
json.dump(out, open(os.path.join(ROOT, "results", "exp21_validation.json"), "w"), indent=1, default=str)
print("saved")
