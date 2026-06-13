"""Concentration TRIM vs never-sell and vs full-liquidation.

On each rebalance boundary (quarter/year), sell only the EXCESS of any holding
above a weight cap and redeploy it into the current top-2. Most of the book —
and its deferred gains — is left alone, so turnover and tax are a fraction of a
full liquidation. We measure: does it cap concentration without hurting the
robust metrics, and is it more phase-stable than full liquidation?

Same 244-window grid, biweekly, 5 bps. `python research/trim_study.py`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import fast
import protocol
import strategy_dca

P = data.build_panel()
S = strategy_dca.build_scores(P)
fd = protocol.get_shared()["fd"]
Snp = S.reindex(index=fd.index, columns=fd.columns).to_numpy(float)
wins, bench = protocol._bench_grid(10, 0, 1000.0, 5.0)


def grid(trim_cap=None, trim_period=None):
    vs_qqq, vs_spy, mults = [], [], []
    full = None
    for wname, s, e in wins:
        if wname in protocol.REGIMES:
            continue
        _, vals, inv = fast.run_fast(fd, Snp, k=2, every=10, start=s, end=e,
                                     cost_bps=5.0, trim_cap=trim_cap,
                                     trim_period=trim_period)
        if inv[0] <= 0:
            continue
        m = vals[0] / inv[0]
        vs_qqq.append(m / bench["qqq"][wname] - 1)
        vs_spy.append(m / bench["spy"][wname] - 1)
        mults.append(m)
        if wname.endswith("_end") and full is None:  # earliest start = 2006 ITD
            full = m
    vq, vs = np.array(vs_qqq), np.array(vs_spy)
    return {"win_qqq": float((vq > 0).mean()), "win_spy": float((vs > 0).mean()),
            "med_vs_qqq": float(np.median(vq)), "p10": float(np.quantile(vq, .1)),
            "worst": float(vq.min()), "full_mult": float(full)}


def top_weight(trim_cap, trim_period):
    _, v, inv, hold = fast.run_fast(fd, Snp, k=2, every=10, start="2006-01-03",
                                    cost_bps=5.0, trim_cap=trim_cap,
                                    trim_period=trim_period, return_holdings=True)
    tot = sum(x for x in hold.values() if x == x and x > 0)
    top = sorted((x / tot for x in hold.values() if x == x and x > 0),
                 reverse=True)
    return top[0], len([x for x in hold.values() if x == x and x > 0])


configs = [("never sell", None, None),
           ("trim 33% / yr", 0.33, "annual"),
           ("trim 25% / yr", 0.25, "annual"),
           ("trim 20% / yr", 0.20, "annual"),
           ("trim 33% / qtr", 0.33, "quarterly"),
           ("trim 25% / qtr", 0.25, "quarterly"),
           ("trim 20% / qtr", 0.20, "quarterly")]

print(f"{'config':16} {'winQQQ':>7} {'winSPY':>7} {'medQQQ':>8} {'p10':>7} "
      f"{'worst':>8} {'fullx':>7} {'topwt':>7}")
out = {}
for name, cap, per in configs:
    g = grid(cap, per)
    tw = top_weight(cap, per) if cap else (None, None)
    g["top_weight"] = tw[0]
    out[name] = g
    tws = f"{tw[0]*100:5.0f}%" if tw[0] else "  36%"
    print(f"{name:16} {g['win_qqq']*100:6.0f}% {g['win_spy']*100:6.0f}% "
          f"{g['med_vs_qqq']*100:+7.1f}% {g['p10']*100:+6.1f}% "
          f"{g['worst']*100:+7.1f}% {g['full_mult']:6.1f} {tws:>7}")

# phase robustness: trim 25%/qtr full multiple across the 10 schedule offsets
print("\nPhase robustness of full multiple (10 schedule offsets):")
for name, cap, per in [("never sell", None, None), ("trim 25% / qtr", 0.25, "quarterly"),
                       ("trim 25% / yr", 0.25, "annual")]:
    ms = []
    for off in range(10):
        _, v, inv = fast.run_fast(fd, Snp, k=2, every=10, offset=off,
                                  start="2006-01-03", cost_bps=5.0,
                                  trim_cap=cap, trim_period=per)
        ms.append(v[0] / inv[0])
    print(f"  {name:16} median {np.median(ms):5.1f}x  range [{min(ms):.1f}, {max(ms):.1f}]")

json.dump(out, open(os.path.join(os.path.dirname(__file__), "trim_study.json"),
                    "w"), indent=1, default=str)
