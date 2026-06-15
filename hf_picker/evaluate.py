"""Does each predictor actually forecast the underwater objective?

Two views:
  1. Rank-IC: at every monthly date, across all eligible candidates, the
     Spearman correlation between a predictor and the REALIZED underwater
     fraction / max dip over the forward horizon. Averaged across dates with a
     t-stat. This is the clean, selection-free measure of skill. Chronos is
     pretrained (never fit on this data), so its IC here is genuinely
     out-of-sample.
  2. Arm scorecard: top-k selection per predictor, purchase-weighted underwater
     metrics (in backtest.py).

Sign convention: predictors are "higher = safer". We report IC vs realized
underwater fraction, so a skillful safe-picker has NEGATIVE IC (high score ->
low realized underwater).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import baselines
import chronos_signal as cs
from objective import Arrays, signal_positions, underwater_metrics


def realized_panel(arr: Arrays, horizon: int, start="2010-01-01", end=None,
                   every=21):
    """Per (date, eligible-candidate) realized underwater outcomes. Returns a
    long DataFrame with sig_pos, ticker, uw_frac, max_dip, end_ret."""
    sp = signal_positions(arr.index, every, 0, start, end)
    frames = []
    for p in sp:
        cand = np.where(arr.elig[p] & ~np.isnan(arr.open[p + 1]))[0]
        if len(cand) == 0:
            continue
        ep = np.full(len(cand), p + 1)
        tbl = underwater_metrics(arr, ep, cand, horizon)
        tbl["sig_pos"] = p
        frames.append(tbl[["sig_pos", "ticker", "uw_frac", "max_dip",
                           "end_ret"]])
    return pd.concat(frames, ignore_index=True)


def predictor_matrix(name, extra):
    if name in extra:
        return extra[name]
    return baselines.build(name)


def ic_table(arr, predictors, extra, horizon, start="2010-01-01", end=None):
    rp = realized_panel(arr, horizon, start, end)
    rp = rp.dropna(subset=["uw_frac"])
    out = []
    for name in predictors:
        M = predictor_matrix(name, extra)
        sc = M[rp["sig_pos"].to_numpy(), rp["ticker"].to_numpy()]
        d = rp.assign(score=sc).dropna(subset=["score"])
        # per-date Spearman of score vs realized underwater frac & max dip
        ic_uw, ic_dip = [], []
        for _, g in d.groupby("sig_pos"):
            if len(g) < 10 or g["score"].nunique() < 3:
                continue
            ic_uw.append(spearmanr(g["score"], g["uw_frac"]).statistic)
            ic_dip.append(spearmanr(g["score"], g["max_dip"]).statistic)
        ic_uw = np.array(ic_uw)
        ic_dip = np.array(ic_dip)
        t = ic_uw.mean() / ic_uw.std() * np.sqrt(len(ic_uw)) if len(ic_uw) else np.nan
        out.append({
            "predictor": name, "n_dates": len(ic_uw),
            "IC_uw": ic_uw.mean(), "t_uw": t,
            "IC_maxdip": ic_dip.mean(),
        })
    return pd.DataFrame(out).set_index("predictor")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()

    arr = Arrays()
    extra = {
        "chronos_safety": cs.safety_score(),      # -predicted underwater frac
        "chronos_q10margin": cs.q10_margin_score(),
    }
    predictors = ["chronos_safety", "chronos_q10margin", "low_vol",
                  "trend_quality", "downside_trend", "trend_smoothness",
                  "low_maxdd", "mom_12_1", "self_underwater", "random"]
    tab = ic_table(arr, predictors, extra, args.horizon, start=args.start)
    tab = tab.sort_values("IC_uw")     # most negative (best safe-picker) first
    pd.set_option("display.width", 160)
    print(f"\nRank-IC vs realized underwater (horizon {args.horizon}td, "
          f"{args.start}+). Lower IC_uw = better safe-picker.\n")
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
