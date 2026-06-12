"""Generate research/results_ml.md from cached ML scores + scorecards."""
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

SC = os.path.join(_HERE, "scorecards")


def card(name):
    return json.load(open(os.path.join(SC, f"{name}.json")))["card"]


meta = {s: json.load(open(os.path.join(
            _HERE, "ml_meta.json" if s == "l2" else f"ml_meta_{s}.json")))
        for s in ("l2", "lrank")}
sig = {"l2": "ml_lgbm_rank", "lrank": "ml_lgbm_lrank"}
cards = {(s, k): card(f"{sig[s]}_k{k}") for s in sig for k in (1, 2, 3, 5)}
base = {n: card(n) for n in ("mom_12_1", "mom_ret126", "naive_mom126")}

L = []
L.append("# ML cross-sectional ranker (LightGBM, walk-forward)\n")
L.append("Code: `research/signals_ml.py` (scores cached to "
         "`research/ml_scores.parquet` / `ml_scores_lrank.parquet`).\n")
L.append("## Setup\n")
L.append("* 20 trailing-only OHLCV features per (date, ticker): returns "
         "21/63/126/252d, 12-1 momentum, distance from & days since 252d "
         "high, realized vol 20/60/120d + ratios, up-day share 63d, volume "
         "ratio 20/120d, up/down volume share 63d, max daily return 21d, "
         "rolling beta & idiosyncratic 12m momentum vs SPY (252d cov/var), "
         "60d skew, range contraction 10/60d. Every feature is "
         "cross-sectionally rank-transformed within its date, among index "
         "members only.")
L.append("* Label: forward 126d total return, rank-transformed within date "
         "(members only). Used for training only.")
L.append("* Walk-forward: training dates sampled every 10 td; refit every "
         "126 td (~6 months, 37 refits); a model fit at date T trains only "
         "on feature dates <= T - 127 td, so every forward label is fully "
         "realized strictly before T (no label leakage); it predicts all "
         "dates in (T, T+126 td].")
L.append(f"* First prediction **{meta['l2']['first_prediction']}** (needs "
         "~3y of labeled history). Scores are NaN before that, so grid "
         "windows starting 2006-2007 hold cash on early NaN rows; this is "
         "expected and does NOT explain the results (see era split below).")
L.append("* Two specs, no hyperparameter search: (a) `ml_lgbm_rank` — "
         "LGBMRegressor (L2 on rank label), 400 trees, lr 0.05, 31 leaves, "
         "min_data_in_leaf 200, feature_fraction 0.8, <=300k rows/fit; "
         "(b) `ml_lgbm_lrank` — LGBMRanker (lambdarank, decile relevance, "
         "grouped by date), identical tree settings.\n")

L.append("## Scorecards (biweekly DCA, 5 bps, full 244-window grid)\n")
L.append("| signal | k | win_qqq | win_spy | med_vs_qqq | med_vs_spy "
         "| worst_vs_qqq | p10_vs_qqq | full_mult |")
L.append("|---|---|---|---|---|---|---|---|---|")
for (s, k), c in cards.items():
    L.append(f"| {sig[s]} | {k} | {c['win_qqq']:.0%} | {c['win_spy']:.0%} "
             f"| {c['med_vs_qqq']:+.1%} | {c['med_vs_spy']:+.1%} "
             f"| {c['worst_vs_qqq']:+.1%} | {c['p10_vs_qqq']:+.1%} "
             f"| {c['full_mult']:.2f} |")
for n, c in base.items():
    L.append(f"| {n} (baseline) | {c['k']} | {c['win_qqq']:.0%} "
             f"| {c['win_spy']:.0%} | {c['med_vs_qqq']:+.1%} "
             f"| {c['med_vs_spy']:+.1%} | {c['worst_vs_qqq']:+.1%} "
             f"| {c['p10_vs_qqq']:+.1%} | {c['full_mult']:.2f} |")

L.append("\nEra split (k=3, L2 spec, grid windows only): starts before the "
         "first prediction date (NaN cash drag) win_qqq 28% / med -8.5%; "
         "starts after win_qqq 28% / med -6.7% vs mom_12_1's 80% / +19.4% "
         "on the same subset — the failure is the signal, not the NaN "
         "warm-up.\n")

L.append("## Regime windows (k=3)\n")
L.append("| regime | l2 vs_qqq | lrank vs_qqq | mom_12_1 vs_qqq |")
L.append("|---|---|---|---|")
for r in cards[("l2", 3)]["regimes"]:
    a = cards[("l2", 3)]["regimes"][r]["vs_qqq"]
    b = cards[("lrank", 3)]["regimes"].get(r, {}).get("vs_qqq", float("nan"))
    m = base["mom_12_1"]["regimes"].get(r, {}).get("vs_qqq", float("nan"))
    L.append(f"| {r} | {a:+.1%} | {b:+.1%} | {m:+.1%} |")

L.append("\n## Rank-IC (Spearman, score vs realized fwd 126d return) by year\n")
L.append("| year | l2 IC | lrank IC |")
L.append("|---|---|---|")
years = sorted(set(meta["l2"]["ic_by_year"]) | set(meta["lrank"]["ic_by_year"]))
for yr in years:
    a = meta["l2"]["ic_by_year"].get(yr)
    b = meta["lrank"]["ic_by_year"].get(yr)
    L.append(f"| {yr} | {a:+.3f} | {b:+.3f} |")
for s in ("l2", "lrank"):
    L.append(f"\n{sig[s]}: overall mean IC {meta[s]['ic_overall_mean']:+.3f}, "
             f"t-stat {meta[s]['ic_overall_t']:.1f} (per-date ICs every 10 "
             "td; last ~6 months unlabeled, excluded).")

L.append("\n## Feature importances across the 37 refits (gain share)\n")
L.append("| feature | l2 mean | l2 std | l2 in-top-5 | lrank mean |")
L.append("|---|---|---|---|---|")
imp = {s: pd.DataFrame(meta[s]["importances_gain_share"],
                       columns=meta[s]["features"]) for s in sig}
order = imp["l2"].mean().sort_values(ascending=False).index
top5 = (imp["l2"].rank(axis=1, ascending=False) <= 5).mean()
for f in order:
    L.append(f"| {f} | {imp['l2'][f].mean():.3f} | {imp['l2'][f].std():.3f} "
             f"| {top5[f]:.0%} | {imp['lrank'][f].mean():.3f} |")

L.append("""
## Honest assessment

* **The ML ranker fails decisively against simple momentum.** Best ML
  variant (lrank/l2, any k) is far below the naive `mom_12_1` baseline
  (80% win_qqq, +24.6% median) and below `naive_mom126` (73%, +14%); it
  does not clear the protocol bar (win_qqq >= 85%) and is not close.
* **Out-of-sample IC is ~zero** (mean ~0.00, t < 1 for L2). Whatever
  in-sample structure LightGBM finds in 20 rank features does not
  generalize 6 months forward. Yearly ICs flip sign (notably -0.13 in
  2009: a model trained through the 2008 crash goes hard defensive and
  is wrong-footed by the junk rally).
* **Why it underperforms even with ~zero IC bias:** gain importances are
  dominated by beta_252, vol_120 and days_since_high_252 — the model
  learns a defensive low-beta/low-vol tilt (those features "explain" rank
  variance symmetrically). A low-beta basket has a decent *average* rank
  but systematically lags QQQ in the tech-led bull that dominates the
  grid; the DCA objective rewards right-tail picks, which an L2/NDCG
  objective on mean rank does not target.
* **Importance stability:** beta/vol/days-since-high are in the L2 top-5
  in most refits; pure momentum features (ret_252, mom_12_1, idio_mom)
  carry only ~6% gain share each — the model dilutes the one family that
  actually works here with 15 noise-prone features.
* **Spec comparison (allowed 2-3, no grid search):** lambdarank vs L2
  changes little; the problem is the feature/label structure, not the
  loss. We did not tune further — tuning hyperparameters on the
  evaluation grid would be overfitting the protocol.
* Conclusion: with OHLCV-only features at 6-month horizon on ~500 names,
  a pooled gradient-boosted ranker is *worse* than ranking on a single
  momentum column. The cross-sectional edge in this dataset is thin and
  almost entirely the momentum direction; ML mostly adds defensive
  confounders. Not advanced to the slow engine / leakage audit.
""")

with open(os.path.join(_HERE, "results_ml.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("wrote results_ml.md")
