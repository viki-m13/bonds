"""Paired comparison: chronos_rerank vs chronos_control_mom scorecards.

Both arms share dates, candidate sets, cadence, and costs, so the per-window
difference in final multiple is the cleanest estimate of what Chronos adds.
Grid windows starting before 2016 are excluded (scores begin 2016-01; both
arms hold cash before that, so those windows are uninformative).
"""
import json
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(_HERE, "scorecards")

REGIMES = {"GFC_2007_2009", "recovery_2009_2012", "bull_2013_2017",
           "sideways_2015_2016", "vol_2018", "covid_2020", "bear_2022",
           "ai_bull_2023_2026"}


def load(name):
    with open(os.path.join(SC, f"{name}.json")) as f:
        d = json.load(f)
    w = pd.DataFrame(d["windows"])
    w["start"] = pd.to_datetime(w["start"])
    return d["card"], w


def main():
    print(f"{'k':>2} {'arm':<8} {'n':>3} {'win_qqq':>8} {'win_spy':>8} "
          f"{'med_vs_qqq':>11} {'worst_vs_qqq':>13} {'med_mult':>9}")
    for k in (1, 2, 3):
        _, wc = load(f"chronos_rerank_k{k}")
        _, wm = load(f"chronos_control_mom_k{k}")
        m = wc.merge(wm, on="window", suffixes=("_c", "_m"))
        grid = m[(~m["window"].isin(REGIMES))
                 & (m["start_c"] >= "2016-01-01")].copy()
        for arm, suf in (("chronos", "_c"), ("control", "_m")):
            g = grid
            print(f"{k:>2} {arm:<8} {len(g):>3} "
                  f"{(g['vs_qqq' + suf] > 0).mean():>8.0%} "
                  f"{(g['vs_spy' + suf] > 0).mean():>8.0%} "
                  f"{g['vs_qqq' + suf].median():>+11.1%} "
                  f"{g['vs_qqq' + suf].min():>+13.1%} "
                  f"{g['mult' + suf].median():>9.3f}")
        d = grid["mult_c"] / grid["mult_m"] - 1
        t = d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 1 else np.nan
        print(f"   paired delta (chronos/control-1): med {d.median():+.2%}  "
              f"mean {d.mean():+.2%}  win {(d > 0).mean():.0%}  "
              f"t={t:.2f}  n={len(d)}")
        reg = m[m["window"].isin(REGIMES) & (m["start_c"] >= "2016-01-01")]
        for _, r in reg.iterrows():
            print(f"   regime {r['window']:<22} chronos {r['mult_c']:.3f} "
                  f"control {r['mult_m']:.3f} "
                  f"delta {r['mult_c'] / r['mult_m'] - 1:+.2%}")
        print()


if __name__ == "__main__":
    main()
