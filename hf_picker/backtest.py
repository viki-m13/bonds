"""Backtest driver: score every arm on the underwater-avoidance objective.

Runs a monthly, next-open, top-k DCA-style selection and reports the
purchase-weighted underwater metrics (the objective) plus end-of-horizon P&L.
All arms share dates, eligibility, k, horizon and costs, so differences are
the signal, not the harness.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import baselines
from objective import Arrays, evaluate_arm

_HERE = os.path.dirname(os.path.abspath(__file__))

HEADLINE = ["n_buys", "underwater_frac", "ever_underwater", "never_underwater",
            "mean_max_dip", "p10_max_dip", "mean_days_uw", "hit_rate_end",
            "mean_end_ret", "median_end_ret"]


def run(arms, k=3, horizon=126, start="2010-01-01", end=None,
        extra_scores=None):
    arr = Arrays()
    rows = []
    for name in arms:
        if extra_scores and name in extra_scores:
            sc = extra_scores[name]
        else:
            sc = baselines.build(name)
        m = evaluate_arm(arr, sc, k=k, horizon=horizon, start=start, end=end)
        m["arm"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("arm")
    return df[[c for c in HEADLINE if c in df.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=126)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--arms", nargs="*", default=list(baselines.REGISTRY))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = run(args.arms, k=args.k, horizon=args.horizon,
             start=args.start, end=args.end)
    df = df.sort_values("underwater_frac")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\nk={args.k}  horizon={args.horizon}td  start={args.start}  "
          f"(sorted by underwater_frac, lower = better)\n")
    print(df.round(4).to_string())
    if args.out:
        df.round(6).to_json(os.path.join(_HERE, args.out), orient="index",
                            indent=2)


if __name__ == "__main__":
    main()
