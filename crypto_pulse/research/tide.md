# TIDE — Trend-Intensity-Dependent Exposure breakout (generalization battery)

TIDE: x-sectional market-neutral 20d breakout, gross scaled by causal market trend-intensity. Net 4.5bps+funding, vol-targeted. Below: the anti-overfit tests, full distributions (not cherry-picked).

**Headline:** full-period Sharpe +1.28, HL-era +2.01, HL-OOS +1.98, pre-HL (independent) +1.11.

## 1. Parameter-grid robustness (HL-era Sharpe)

Is 20/50/3 a plateau or a lucky spike? Each cell = full HL-era Sharpe.

| breakout win \ regime | reg30 | reg50 | reg80 |
|---|---|---|---|
| win10 | +1.54 | +1.42 | +1.25 |
| win15 | +1.74 | +1.80 | +1.53 |
| win20 | +1.86 | +2.01 | +1.78 |
| win30 | +1.96 | +2.21 | +2.03 |
| win40 | +1.64 | +1.90 | +1.79 |

Grid: 100% of 15 cells > 1.0 Sharpe, min +1.25, median +1.79, max +2.21. Broad plateau -> not overfit.

## 2. Year-by-year Sharpe (HL-era + pre)

| year | Sharpe |
|---|---|
| 2019 | +0.39 |
| 2020 | +1.34 |
| 2021 | +1.17 |
| 2022 | +1.43 |
| 2023 | +1.44 |
| 2024 | +1.62 |
| 2025 | +2.33 |
| 2026 | +0.93 |

## 3. Cost sensitivity (HL-era Sharpe)

| taker mult | Sharpe |
|---|---|
| 1x (4.5bps) | +2.01 |
| 2x (9.0bps) | +1.78 |
| 3x (13.5bps) | +1.54 |
| 4x (18.0bps) | +1.31 |

## 4. Coin-subsample bootstrap (20 draws, 70% of coins)

- HL-era Sharpe across random 70% coin subsets: mean +1.59, 5th pct +1.00, min +0.99. Not driven by a few coins.

## 5. Shuffle null (signal permuted across coins)

- Null Sharpe (10 shuffles): mean -0.63, max +0.50 vs real +2.01. Edge vanishes under shuffle -> no look-ahead leak.

## 6. Rolling walk-forward (4 disjoint HL OOS folds)

| fold | Sharpe |
|---|---|
| fold1 (2023-05-12..2024-02-05) | +2.31 |
| fold2 (2024-02-06..2024-11-01) | +0.92 |
| fold3 (2024-11-02..2025-07-29) | +2.41 |
| fold4 (2025-07-30..2026-04-24) | +2.43 |

## Verdict — does TIDE generalize?

- Parameter plateau: YES (100% of cells >1.0). Bootstrap 5th-pct +1.00. Null max +0.50. All 4 WF folds positive: YES.
- **TIDE GENERALIZES — robust across params, coins, costs, time, and walk-forward folds, with a clean null.**
- Honest level: a ~2.0 Sharpe tradeable book (full-period ~1.3). Robust, not overfit — but ~2, not 3. Reported standalone, as requested.
