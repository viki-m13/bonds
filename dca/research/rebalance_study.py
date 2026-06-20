"""What if SUMMIT periodically SOLD EVERYTHING and redeployed into the current
picks, instead of never selling?

Contributions still arrive biweekly and buy the top-2. Additionally, on each
rebalance boundary (first biweekly date of a new month / quarter / year) we
liquidate the ENTIRE portfolio at the next open and the proceeds + that
period's contribution buy the current top-2. Compared head-to-head with the
never-sell baseline on the same 244-window grid, 5 bps/trade.

Note: the engine models trading cost on every sell+rebuy but NOT taxes; a real
taxable account would owe short-term gains on every liquidation, so the
monthly/quarterly variants are flattered here.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import protocol
import strategy_dca
from engine import schedule_dates

P = data.build_panel()
S = strategy_dca.build_scores(P)
idx = P["close"].index
cols = P["close"].columns
sig = schedule_dates(idx, every=10, start="2006-01-03")


def rebalance_sell(freq):
    """Boolean (date x ticker) — True on every ticker on the first biweekly
    signal date of each new period, so the engine liquidates all holdings."""
    keyfn = {
        "monthly": lambda d: (d.year, d.month),
        "quarterly": lambda d: (d.year, (d.month - 1) // 3),
        "annual": lambda d: d.year,
        "biweekly": lambda d: (d.year, d.month, d.day),  # every contribution
    }[freq]
    flags = np.zeros(len(idx), dtype=bool)
    prev = None
    pos = idx.get_indexer(sig)
    for d, p in zip(sig, pos):
        k = keyfn(d)
        if k != prev:
            flags[p] = True
        prev = k
    sell = pd.DataFrame(np.repeat(flags[:, None], len(cols), axis=1),
                        index=idx, columns=cols)
    return sell


def metrics(name, sell):
    c = protocol.evaluate_signal(S, name, k=2, every=10, cost_bps=5.0,
                                 sell=sell, save=False, quiet=True)
    return c


rows = []
configs = [("never-sell (SUMMIT)", None),
           ("annual rebalance", rebalance_sell("annual")),
           ("quarterly rebalance", rebalance_sell("quarterly")),
           ("monthly rebalance", rebalance_sell("monthly")),
           ("every-buy rebalance", rebalance_sell("biweekly"))]

print(f"{'config':22} {'winQQQ':>7} {'winSPY':>7} {'medQQQ':>8} "
      f"{'p10QQQ':>8} {'worstQQQ':>9} {'fullx':>7}")
out = {}
for name, sell in configs:
    c = metrics(name, sell)
    out[name] = c
    print(f"{name:22} {c['win_qqq']*100:6.0f}% {c['win_spy']*100:6.0f}% "
          f"{c['med_vs_qqq']*100:+7.1f}% {c['p10_vs_qqq']*100:+7.1f}% "
          f"{c['worst_vs_qqq']*100:+8.1f}% {c['full_mult']:6.1f}")

print("\n=== regime vs_qqq ===")
regs = list(out["never-sell (SUMMIT)"]["regimes"].keys())
print(f"{'regime':22} " + "".join(f"{n.split()[0][:8]:>9}" for n, _ in configs))
for rg in regs:
    line = f"{rg:22} "
    for name, _ in configs:
        line += f"{out[name]['regimes'][rg]['vs_qqq']*100:+8.0f}%"
    print(line)

json.dump(out, open(os.path.join(os.path.dirname(__file__),
                                 "rebalance_study.json"), "w"),
          indent=1, default=str)
