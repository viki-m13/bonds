"""Full head-to-head: IC and arm scorecard for baselines, Chronos, composite.

This is the decision document — it answers, on identical dates/eligibility/k,
which predictor best satisfies "after I buy, it rarely goes below my price".
"""
import argparse

import numpy as np
import pandas as pd

import baselines
import chronos_signal as cs
import composite
from objective import Arrays, evaluate_arm
from evaluate import ic_table


def extra_scores():
    e = {
        "chronos_safety": cs.safety_score(),
        "chronos_q10margin": cs.q10_margin_score(),
    }
    e.update(composite.build_named())
    return e


ARMS = ["composite_lv_qm", "low_vol", "chronos_q10margin", "chronos_safety",
        "composite_lv_qm_safe", "trend_quality", "mom_12_1", "random"]

SCORE_COLS = ["n_buys", "underwater_frac", "ever_underwater", "mean_max_dip",
              "p10_max_dip", "mean_days_uw", "hit_rate_end", "mean_end_ret",
              "median_end_ret"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=126)
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()

    arr = Arrays()
    extra = extra_scores()

    # arm scorecard
    rows = []
    for name in ARMS:
        sc = extra[name] if name in extra else baselines.build(name)
        m = evaluate_arm(arr, sc, k=args.k, horizon=args.horizon,
                         start=args.start)
        m["arm"] = name
        rows.append(m)
    board = pd.DataFrame(rows).set_index("arm")[SCORE_COLS]
    board = board.sort_values("underwater_frac")

    # IC table on the same predictors
    ic = ic_table(arr, ARMS, extra, args.horizon, start=args.start)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n=== Arm scorecard (k={args.k}, horizon={args.horizon}td, "
          f"{args.start}+, monthly, sorted by underwater_frac) ===\n")
    print(board.round(4).to_string())
    print(f"\n=== Rank-IC vs realized underwater frac (lower=better) ===\n")
    print(ic.sort_values("IC_uw").round(4).to_string())


if __name__ == "__main__":
    main()
