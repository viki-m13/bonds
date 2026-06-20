# ML cross-sectional ranker (LightGBM, walk-forward)

Code: `research/signals_ml.py` (scores cached to `research/ml_scores.parquet` / `ml_scores_lrank.parquet`).

## Setup

* 20 trailing-only OHLCV features per (date, ticker): returns 21/63/126/252d, 12-1 momentum, distance from & days since 252d high, realized vol 20/60/120d + ratios, up-day share 63d, volume ratio 20/120d, up/down volume share 63d, max daily return 21d, rolling beta & idiosyncratic 12m momentum vs SPY (252d cov/var), 60d skew, range contraction 10/60d. Every feature is cross-sectionally rank-transformed within its date, among index members only.
* Label: forward 126d total return, rank-transformed within date (members only). Used for training only.
* Walk-forward: training dates sampled every 10 td; refit every 126 td (~6 months, 37 refits); a model fit at date T trains only on feature dates <= T - 127 td, so every forward label is fully realized strictly before T (no label leakage); it predicts all dates in (T, T+126 td].
* First prediction **2008-01-14** (needs ~3y of labeled history). Scores are NaN before that, so grid windows starting 2006-2007 hold cash on early NaN rows; this is expected and does NOT explain the results (see era split below).
* Two specs, no hyperparameter search: (a) `ml_lgbm_rank` — LGBMRegressor (L2 on rank label), 400 trees, lr 0.05, 31 leaves, min_data_in_leaf 200, feature_fraction 0.8, <=300k rows/fit; (b) `ml_lgbm_lrank` — LGBMRanker (lambdarank, decile relevance, grouped by date), identical tree settings.

## Scorecards (biweekly DCA, 5 bps, full 244-window grid)

| signal | k | win_qqq | win_spy | med_vs_qqq | med_vs_spy | worst_vs_qqq | p10_vs_qqq | full_mult |
|---|---|---|---|---|---|---|---|---|
| ml_lgbm_rank | 1 | 30% | 80% | -6.2% | +9.7% | -41.9% | -24.8% | 5.96 |
| ml_lgbm_rank | 2 | 32% | 84% | -5.4% | +10.7% | -36.2% | -20.9% | 5.89 |
| ml_lgbm_rank | 3 | 28% | 83% | -7.0% | +10.5% | -39.2% | -20.2% | 5.54 |
| ml_lgbm_rank | 5 | 22% | 82% | -7.8% | +9.5% | -38.7% | -21.1% | 5.58 |
| ml_lgbm_lrank | 1 | 45% | 73% | -4.6% | +12.0% | -41.5% | -21.1% | 10.65 |
| ml_lgbm_lrank | 2 | 43% | 80% | -4.0% | +12.7% | -36.1% | -20.2% | 8.64 |
| ml_lgbm_lrank | 3 | 38% | 81% | -3.1% | +12.6% | -37.0% | -19.5% | 7.42 |
| ml_lgbm_lrank | 5 | 28% | 81% | -5.1% | +11.5% | -36.7% | -18.0% | 7.92 |
| mom_12_1 (baseline) | 3 | 80% | 91% | +24.6% | +55.8% | -30.9% | -6.8% | 12.09 |
| mom_ret126 (baseline) | 3 | 73% | 88% | +14.0% | +34.7% | -39.7% | -15.6% | 13.63 |
| naive_mom126 (baseline) | 3 | 73% | 88% | +14.0% | +34.7% | -39.7% | -15.6% | 13.63 |

Era split (k=3, L2 spec, grid windows only): starts before the first prediction date (NaN cash drag) win_qqq 28% / med -8.5%; starts after win_qqq 28% / med -6.7% vs mom_12_1's 80% / +19.4% on the same subset — the failure is the signal, not the NaN warm-up.

## Regime windows (k=3)

| regime | l2 vs_qqq | lrank vs_qqq | mom_12_1 vs_qqq |
|---|---|---|---|
| GFC_2007_2009 | -5.3% | -15.9% | -14.2% |
| recovery_2009_2012 | -3.9% | -3.5% | -26.5% |
| bull_2013_2017 | -12.1% | -10.0% | +24.8% |
| sideways_2015_2016 | +4.3% | +6.8% | +12.2% |
| vol_2018 | -2.2% | -4.9% | -12.3% |
| covid_2020 | +14.8% | +15.9% | +9.9% |
| bear_2022 | -5.6% | -5.4% | -18.4% |
| ai_bull_2023_2026 | +1.9% | -11.0% | +7.1% |

## Rank-IC (Spearman, score vs realized fwd 126d return) by year

| year | l2 IC | lrank IC |
|---|---|---|
| 2008 | +0.039 | -0.100 |
| 2009 | -0.133 | +0.068 |
| 2010 | +0.000 | +0.024 |
| 2011 | +0.088 | -0.137 |
| 2012 | -0.061 | -0.008 |
| 2013 | -0.072 | +0.035 |
| 2014 | +0.031 | -0.024 |
| 2015 | +0.039 | -0.160 |
| 2016 | -0.013 | +0.074 |
| 2017 | -0.019 | -0.040 |
| 2018 | +0.065 | -0.059 |
| 2019 | +0.041 | -0.095 |
| 2020 | +0.008 | +0.321 |
| 2021 | -0.010 | -0.074 |
| 2022 | +0.032 | -0.007 |
| 2023 | -0.007 | +0.027 |
| 2024 | +0.019 | -0.122 |
| 2025 | +0.003 | +0.111 |

ml_lgbm_rank: overall mean IC +0.002, t-stat 0.5 (per-date ICs every 10 td; last ~6 months unlabeled, excluded).

ml_lgbm_lrank: overall mean IC -0.010, t-stat -1.2 (per-date ICs every 10 td; last ~6 months unlabeled, excluded).

## Feature importances across the 37 refits (gain share)

| feature | l2 mean | l2 std | l2 in-top-5 | lrank mean |
|---|---|---|---|---|
| beta_252 | 0.127 | 0.013 | 100% | 0.099 |
| days_since_high_252 | 0.112 | 0.012 | 100% | 0.084 |
| vol_120 | 0.090 | 0.005 | 100% | 0.220 |
| idio_mom_252 | 0.078 | 0.011 | 100% | 0.059 |
| ret_252 | 0.061 | 0.004 | 76% | 0.039 |
| mom_12_1 | 0.056 | 0.004 | 11% | 0.038 |
| skew_60 | 0.054 | 0.005 | 0% | 0.047 |
| dist_high_252 | 0.053 | 0.006 | 3% | 0.043 |
| vol_60 | 0.052 | 0.006 | 11% | 0.058 |
| ret_126 | 0.049 | 0.004 | 0% | 0.043 |
| up_share_63 | 0.039 | 0.003 | 0% | 0.031 |
| ret_63 | 0.039 | 0.003 | 0% | 0.034 |
| updown_volm_63 | 0.036 | 0.003 | 0% | 0.035 |
| volm_ratio_20_120 | 0.030 | 0.004 | 0% | 0.031 |
| max_ret_21 | 0.025 | 0.002 | 0% | 0.023 |
| vol_20 | 0.024 | 0.003 | 0% | 0.037 |
| vol_ratio_20_120 | 0.022 | 0.002 | 0% | 0.025 |
| ret_21 | 0.021 | 0.002 | 0% | 0.019 |
| vol_ratio_20_60 | 0.016 | 0.001 | 0% | 0.017 |
| range_contr_10_60 | 0.015 | 0.002 | 0% | 0.016 |

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

