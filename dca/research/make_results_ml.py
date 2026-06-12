"""Generate research/results_ml.md from cached ML scores + scorecards."""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

meta = json.load(open(os.path.join(_HERE, "ml_meta.json")))
cards = {}
for k in (1, 2, 3, 5):
    p = os.path.join(_HERE, "scorecards", f"ml_lgbm_rank_k{k}.json")
    cards[k] = json.load(open(p))["card"]
base = {n: json.load(open(os.path.join(_HERE, "scorecards", f"{n}.json")))["card"]
        for n in ("mom_12_1", "mom_ret126")}

imp = pd.DataFrame(meta["importances_gain_share"], columns=meta["features"],
                   index=meta["fit_dates"])
imp_mean = imp.mean().sort_values(ascending=False)
imp_std = imp.std()
top5_share = (imp.rank(axis=1, ascending=False) <= 5).mean()

L = []
L.append("# ML cross-sectional ranker (LightGBM, walk-forward)\n")
L.append("Signal: `ml_lgbm_rank` — LightGBM regressor on 20 trailing-only "
         "OHLCV features, each cross-sectionally rank-transformed within "
         "date (members only). Label: forward 126d return, rank-transformed "
         "within date. Walk-forward: training dates sampled every 10 td, "
         "refit every 126 td, fit at T trains only on feature dates <= "
         "T-127 td (labels fully realized before T; no leakage). "
         f"First prediction: **{meta['first_prediction']}** "
         f"({len(meta['fit_dates'])} refits). Scores are NaN before that — "
         "grid windows starting 2006-2007 hold cash on NaN rows early, which "
         "drags those windows' multiples.\n")
L.append(f"Model: one spec, no hyperparameter search: "
         f"`{meta['lgbm_params']}`. Max {300000} training rows "
         "(random subsample, fixed seed).\n")

L.append("## Scorecards (biweekly DCA, 5 bps, full grid)\n")
L.append("| signal | k | win_qqq | win_spy | med_vs_qqq | med_vs_spy | "
         "worst_vs_qqq | p10_vs_qqq | full_mult |")
L.append("|---|---|---|---|---|---|---|---|---|")
for k, c in cards.items():
    L.append(f"| ml_lgbm_rank | {k} | {c['win_qqq']:.0%} | {c['win_spy']:.0%} "
             f"| {c['med_vs_qqq']:+.1%} | {c['med_vs_spy']:+.1%} "
             f"| {c['worst_vs_qqq']:+.1%} | {c['p10_vs_qqq']:+.1%} "
             f"| {c['full_mult']:.2f} |")
for n, c in base.items():
    L.append(f"| {n} (baseline) | {c['k']} | {c['win_qqq']:.0%} "
             f"| {c['win_spy']:.0%} | {c['med_vs_qqq']:+.1%} "
             f"| {c['med_vs_spy']:+.1%} | {c['worst_vs_qqq']:+.1%} "
             f"| {c['p10_vs_qqq']:+.1%} | {c['full_mult']:.2f} |")

L.append("\n## Regime windows (k=3)\n")
L.append("| regime | ml mult | ml vs_qqq | ml vs_spy | mom_12_1 vs_qqq |")
L.append("|---|---|---|---|---|")
for r, v in cards[3]["regimes"].items():
    b = base["mom_12_1"]["regimes"].get(r, {})
    L.append(f"| {r} | {v['mult']:.2f} | {v['vs_qqq']:+.1%} | "
             f"{v['vs_spy']:+.1%} | {b.get('vs_qqq', float('nan')):+.1%} |")

L.append("\n## Rank-IC (Spearman, score vs realized fwd 126d return) by year\n")
L.append("| year | mean IC |")
L.append("|---|---|")
for yr, v in meta["ic_by_year"].items():
    L.append(f"| {yr} | {v:+.3f} |")
L.append(f"\nOverall mean IC {meta['ic_overall_mean']:+.3f}, "
         f"t-stat {meta['ic_overall_t']:.1f} (per-date ICs sampled every "
         "10 td; last ~6 months unlabeled, excluded).\n")

L.append("## Feature importances across refits (gain share)\n")
L.append("| feature | mean | std | in top-5 (% of refits) |")
L.append("|---|---|---|---|")
for f in imp_mean.index:
    L.append(f"| {f} | {imp_mean[f]:.3f} | {imp_std[f]:.3f} "
             f"| {top5_share[f]:.0%} |")

with open(os.path.join(_HERE, "results_ml.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("wrote results_ml.md")
print(imp_mean.head(8))
