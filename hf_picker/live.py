"""Live picks: the safest names to buy right now under the model.

Computes the composite safety score (trailing low-volatility rank + Chronos-Bolt
q10 downside-margin rank) on the most recent available close for the current
index members, and prints the top-k — the names the model judges least likely
to trade below today's price after purchase.

Run: python live.py --k 10
"""
import argparse

import numpy as np
import pandas as pd

from data import load_panel, eligibility
from baselines import low_vol
from chronos_signal import (_pu_qm, _pipeline, CONTEXT, PRED_LEN, QLEVELS)


def chronos_live(pos: int):
    """q10 margin + predicted underwater frac for all eligible names at row
    `pos`, computed fresh from the model (not the cache)."""
    import torch
    p = load_panel()
    cl = p["close"].to_numpy(float)
    cols = p["close"].columns
    elig = eligibility(min_history=CONTEXT).to_numpy(bool)
    cand = np.where(elig[pos])[0]
    ctxs, keep = [], []
    for t in cand:
        w = cl[pos - CONTEXT + 1:pos + 1, t]
        if not np.isnan(w).any():
            ctxs.append(torch.tensor(w, dtype=torch.float32))
            keep.append(t)
    pipe = _pipeline()
    q, _ = pipe.predict_quantiles(ctxs, prediction_length=PRED_LEN,
                                  quantile_levels=QLEVELS)
    pu, qm = _pu_qm(q.numpy(), cl[pos, keep], np.array(QLEVELS))
    keep = np.array(keep)
    return cols, keep, pu, qm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    p = load_panel()
    close = p["close"]
    pos = len(close) - 1
    asof = close.index[pos].date()

    cols, keep, pu, qm = chronos_live(pos)
    lv = low_vol().to_numpy(float)[pos, keep]      # higher = lower vol

    # cross-sectional rank-average of low-vol and Chronos q10 margin
    def rank01(x):
        r = pd.Series(x).rank().to_numpy()
        return (r - 1) / max(len(x) - 1, 1)
    score = 0.5 * rank01(lv) + 0.5 * rank01(qm)

    df = pd.DataFrame({
        "ticker": cols[keep],
        "safety_score": score,
        "chronos_q10_margin_63d": qm,
        "chronos_pred_underwater": pu,
        "trailing_vol_126d": -lv,
    }).sort_values("safety_score", ascending=False).head(args.k)
    df = df.reset_index(drop=True)

    pd.set_option("display.width", 160)
    print(f"\nSafest entries as of {asof}  (top {args.k} of {len(keep)} "
          f"S&P 500 members)\n")
    print(df.round(4).to_string(index=False))
    print("\nchronos_q10_margin_63d: model's 10th-pctile 63d-ahead return "
          "(higher = shallower worst-case dip)")
    print("chronos_pred_underwater: model's predicted fraction of the next 63d "
          "spent below today's price (lower = better)")


if __name__ == "__main__":
    main()
