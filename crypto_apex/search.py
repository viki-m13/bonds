"""Search over sleeve combinations and params."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from itertools import combinations
import numpy as np
import pandas as pd
import util
from util import metrics, regime_slice, load_prices, load_macro, SURVIVORS, DEAD
import sleeves as SV
from strategy import build_portfolio

OOS_START = "2022-07-01"
IS_END = "2022-06-30"


def main():
    cp = load_prices()
    macro = load_macro(cp.index)
    all_sw = SV.build_all(cp, macro)

    names = list(all_sw.keys())
    rows = []
    # Try all combos of 2-6 sleeves
    for k in range(2, len(names) + 1):
        for combo in combinations(names, k):
            sw = {n: all_sw[n] for n in combo}
            for tv in [0.25, 0.30, 0.40]:
                net = build_portfolio(cp, sw, target_vol=tv, dd_floor=-0.25)
                net = net.fillna(0.0)
                m_full = metrics(net)
                m_oos = metrics(regime_slice(net, OOS_START, "2027-12-31"))
                m_is = metrics(regime_slice(net, "2014-09-17", IS_END))
                m_22 = metrics(regime_slice(net, "2022-01-01", "2022-12-31"))
                rows.append({
                    "combo": "+".join(combo),
                    "k": k, "tv": tv,
                    "full_sr": m_full["sharpe"],
                    "full_cagr": m_full["cagr"],
                    "is_sr": m_is["sharpe"],
                    "oos_sr": m_oos["sharpe"],
                    "oos_cagr": m_oos["cagr"],
                    "mdd": m_full["mdd"],
                    "y22_sr": m_22["sharpe"],
                })

    df = pd.DataFrame(rows).sort_values("oos_sr", ascending=False)
    print(f"\nTop 20 by OOS SR (post-2022-07):")
    print(f"{'combo':50s} {'tv':>5} {'full':>5} {'oos':>5} {'is':>5} {'cagr_f':>7} {'cagr_o':>7} {'mdd':>6} {'y22':>6}")
    for _, r in df.head(20).iterrows():
        print(f"{r['combo']:50s} {r['tv']:.2f} {r['full_sr']:>5.2f} {r['oos_sr']:>5.2f} "
              f"{r['is_sr']:>5.2f} {r['full_cagr']*100:>6.1f}% {r['oos_cagr']*100:>6.1f}% "
              f"{r['mdd']*100:>5.1f}% {r['y22_sr']:>6.2f}")

    print(f"\nTop 20 by FULL SR:")
    df2 = df.sort_values("full_sr", ascending=False)
    for _, r in df2.head(20).iterrows():
        print(f"{r['combo']:50s} {r['tv']:.2f} {r['full_sr']:>5.2f} {r['oos_sr']:>5.2f} "
              f"{r['is_sr']:>5.2f} {r['full_cagr']*100:>6.1f}% {r['oos_cagr']*100:>6.1f}% "
              f"{r['mdd']*100:>5.1f}% {r['y22_sr']:>6.2f}")


if __name__ == "__main__":
    main()
