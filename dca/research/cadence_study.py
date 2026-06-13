"""Cadence study: SUMMIT and ROTATOR at daily / weekly / biweekly / monthly.

Same score & sell matrices, sampled at different contribution cadences. Each
strategy keeps its native cost (SUMMIT 5 bps, ROTATOR 10 bps) and concentration
(SUMMIT k=2, ROTATOR k=3). Benchmarks (QQQ/SPY DCA) use the SAME cadence, so
win-rates are apples-to-apples within each cadence.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import protocol
import strategy_dca
import strategy_rotator as R

CADENCES = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}

P = data.build_panel()
S = strategy_dca.build_scores(P)
RS, RSell = R.build_scores(P), R.build_sell(P)

CONFIGS = [
    ("SUMMIT", S, None, 2, 5.0),
    ("ROTATOR", RS, RSell, 3, 10.0),
]

results = {}
print(f"{'strategy':9} {'cadence':9} {'winQQQ':>7} {'winSPY':>7} "
      f"{'medQQQ':>8} {'p10QQQ':>8} {'worstQQQ':>9} {'fullx':>7}")
for name, sc, sell, k, cb in CONFIGS:
    for cname, ev in CADENCES.items():
        c = protocol.evaluate_signal(sc, f"{name}_{cname}", k=k, every=ev,
                                     cost_bps=cb, sell=sell, save=False,
                                     quiet=True)
        results[f"{name}|{cname}"] = c
        print(f"{name:9} {cname:9} {c['win_qqq']*100:6.0f}% "
              f"{c['win_spy']*100:6.0f}% {c['med_vs_qqq']*100:+7.1f}% "
              f"{c['p10_vs_qqq']*100:+7.1f}% {c['worst_vs_qqq']*100:+8.1f}% "
              f"{c['full_mult']:6.1f}")
    print()

json.dump(results, open(os.path.join(os.path.dirname(__file__),
                                     "cadence_study.json"), "w"),
          indent=1, default=str)

# regime detail at each cadence for the strategy series vs QQQ
print("\n=== full-period multiple by cadence ===")
print(f"{'cadence':9} {'SUMMIT':>8} {'ROTATOR':>8}")
for cname in CADENCES:
    print(f"{cname:9} {results['SUMMIT|'+cname]['full_mult']:7.1f}x "
          f"{results['ROTATOR|'+cname]['full_mult']:7.1f}x")
