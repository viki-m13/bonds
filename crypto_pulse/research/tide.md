# TIDE — Trend-Intensity-Dependent Exposure breakout (generalization battery)

TIDE: x-sectional market-neutral 20d breakout, gross scaled by causal market trend-intensity. Net 4.5bps+funding, vol-targeted. Below: the anti-overfit tests, full distributions (not cherry-picked).

**Headline:** full-period Sharpe +1.47, HL-era +2.20, HL-OOS +2.19, pre-HL (independent) +1.25.

## 1. Parameter-grid robustness (HL-era Sharpe)

Is 20/50/3 a plateau or a lucky spike? Each cell = full HL-era Sharpe.

| breakout win \ regime | reg30 | reg50 | reg80 |
|---|---|---|---|
| win10 | +1.97 | +1.94 | +1.70 |
| win15 | +2.22 | +2.30 | +2.01 |
| win20 | +2.10 | +2.20 | +1.92 |
| win30 | +2.17 | +2.27 | +2.01 |
| win40 | +2.06 | +2.15 | +1.94 |

Grid: 100% of 15 cells > 1.0 Sharpe, min +1.70, median +2.06, max +2.30. Broad plateau -> not overfit.

## 2. Year-by-year Sharpe (HL-era + pre)

| year | Sharpe |
|---|---|
| 2019 | +0.52 |
| 2020 | +1.87 |
| 2021 | +2.23 |
| 2022 | +2.07 |
| 2023 | +1.57 |
| 2024 | +1.85 |
| 2025 | +2.40 |
| 2026 | +1.46 |

## 3. Cost sensitivity (HL-era Sharpe)

| taker mult | Sharpe |
|---|---|
| 1x (4.5bps) | +2.20 |
| 2x (9.0bps) | +1.97 |
| 3x (13.5bps) | +1.73 |
| 4x (18.0bps) | +1.49 |

## 4. Coin-subsample bootstrap (20 draws, 70% of coins)

- HL-era Sharpe across random 70% coin subsets: mean +1.76, 5th pct +1.23, min +1.14. Not driven by a few coins.

## 5. Shuffle null (signal permuted across coins)

- Null Sharpe (10 shuffles): mean -0.68, max +0.34 vs real +2.20. Edge vanishes under shuffle -> no look-ahead leak.

## 6. Rolling walk-forward (4 disjoint HL OOS folds)

| fold | Sharpe |
|---|---|
| fold1 (2023-05-12..2024-02-05) | +2.45 |
| fold2 (2024-02-06..2024-11-01) | +1.15 |
| fold3 (2024-11-02..2025-07-29) | +2.16 |
| fold4 (2025-07-30..2026-04-24) | +3.08 |

## Verdict — does TIDE generalize?

- Parameter plateau: YES (100% of cells >1.0). Bootstrap 5th-pct +1.23. Null max +0.34. All 4 WF folds positive: YES.
- **TIDE GENERALIZES — robust across params, coins, costs, time, and walk-forward folds, with a clean null.**
- Honest level: a ~2.2 Sharpe tradeable book (full-period ~1.5). Robust, not overfit — but ~2, not 3. Reported standalone, as requested.
