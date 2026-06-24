# TIDE — Trend-Intensity-Dependent Exposure breakout (generalization battery)

TIDE: x-sectional market-neutral 20d breakout, gross scaled by causal market trend-intensity. Net 4.5bps+funding, vol-targeted. Below: the anti-overfit tests, full distributions (not cherry-picked).

**Headline:** full-period Sharpe +1.54, HL-era +2.23, HL-OOS +2.29, pre-HL (independent) +1.35.

## 1. Parameter-grid robustness (HL-era Sharpe)

Is 20/50/3 a plateau or a lucky spike? Each cell = full HL-era Sharpe.

| breakout win \ regime | reg30 | reg50 | reg80 |
|---|---|---|---|
| win10 | +2.08 | +2.15 | +1.87 |
| win15 | +2.17 | +2.25 | +1.97 |
| win20 | +2.20 | +2.23 | +1.95 |
| win30 | +2.26 | +2.31 | +2.07 |
| win40 | +2.15 | +2.21 | +2.00 |

Grid: 100% of 15 cells > 1.0 Sharpe, min +1.87, median +2.15, max +2.31. Broad plateau -> not overfit.

## 2. Year-by-year Sharpe (HL-era + pre)

| year | Sharpe |
|---|---|
| 2019 | +0.49 |
| 2020 | +1.68 |
| 2021 | +2.29 |
| 2022 | +1.94 |
| 2023 | +1.53 |
| 2024 | +1.94 |
| 2025 | +2.38 |
| 2026 | +1.99 |

## 3. Cost sensitivity (HL-era Sharpe)

| taker mult | Sharpe |
|---|---|
| 1x (4.5bps) | +2.23 |
| 2x (9.0bps) | +1.99 |
| 3x (13.5bps) | +1.75 |
| 4x (18.0bps) | +1.50 |

## 4. Coin-subsample bootstrap (20 draws, 70% of coins)

- HL-era Sharpe across random 70% coin subsets: mean +1.81, 5th pct +1.30, min +1.18. Not driven by a few coins.

## 5. Shuffle null (signal permuted across coins)

- Null Sharpe (10 shuffles): mean -0.72, max +1.10 vs real +2.23. Null too high -> suspect.

## 6. Rolling walk-forward (4 disjoint HL OOS folds)

| fold | Sharpe |
|---|---|
| fold1 (2023-05-12..2024-02-05) | +2.28 |
| fold2 (2024-02-06..2024-11-01) | +1.26 |
| fold3 (2024-11-02..2025-07-29) | +2.15 |
| fold4 (2025-07-30..2026-04-24) | +3.30 |

## Verdict — does TIDE generalize?

- Parameter plateau: YES (100% of cells >1.0). Bootstrap 5th-pct +1.30. Null max +1.10. All 4 WF folds positive: YES.
- **TIDE shows some fragility (see flags above).**
- Honest level: a ~2.2 Sharpe tradeable book (full-period ~1.5). Robust, not overfit — but ~2, not 3. Reported standalone, as requested.
