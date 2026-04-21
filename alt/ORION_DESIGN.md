# ORION — Orthogonal Signal Ensemble

## Honest Headline Result
| Window | Sharpe | CAGR | Vol | MaxDD |
|--------|-------:|-----:|----:|------:|
| IS  (2010-03-11 → 2018-12-31) | **0.68** | 12.1% | 19.5% | -31.7% |
| OOS (2019-01-01 → 2026-04-02) | **1.02** | 29.3% | 29.4% | -24.4% |
| Full | **0.85** | 19.5% | 24.4% | -31.7% |

Hard-requirement compliance:
- Sharpe >= 2.0 full : **FAIL (0.85)**
- CAGR >= 20% full  : **FAIL (19.5%)**
- IS Sharpe >= 1.5  : **FAIL (0.68)**
- OOS Sharpe >= 1.5 : **FAIL (1.02)**
- |IS - OOS| <= 0.5 : **PASS (0.34)**

Sharpe >= 2 on a pure leveraged-ETF book is not realistic for this study
(NOVA, the best prior-art model here, honestly achieves ~0.92). The ORION
construction sits in the same ballpark, with a notably smaller max drawdown
and a favourable OOS > IS pattern.

## Architecture

Two sleeves, four signals, 50/50 blend, weekly rebalance, next-day open fill,
5 bps one-way cost.

### Universe (16 leveraged ETFs)
- **Risk (12):** TQQQ, UPRO, QLD, SSO, SOXL, TECL, FAS, ERX, EDC, YINN, DRN, UCO
- **Safe (4):** TMF, UBT, TYD, UGL

### Signals
- **S1** — 12-month momentum (`log-return, 252d`), lagged 1 day. Cross-sectional rank.
- **S2** — 200-day trend filter (`close > 200d MA`), lagged 1 day. Eligibility mask.
- **S3** — Rolling 60-day realised volatility (low-vol tilt, SAFE sleeve).
- **S4** — Macro regime gate (`VIX < 30 AND HY OAS < 7.0`), lagged 1 day. When OFF,
  the risk sleeve is fully de-risked to cash; the safe sleeve stays on.

### Construction
```
w_risk_t = topK_equal_weight( S1 | S2==1, RISK universe, K=4 ) * S4_gate
w_safe_t = topK_equal_weight( 0.7*zS1 + 0.3*z(-60d_vol) | S2==1, SAFE universe, K=2 )
W_t      = 0.5 * w_risk_t + 0.5 * w_safe_t
```
Rebalance on Wednesdays; freeze positions between rebalances. Execute at
next-day open. Transaction cost = 5 bps * sum|Δw| per day.

## Why "orthogonal"

The most important orthogonality in this study is **between sleeves**, not between
signals. Realised correlation:

| | RISK | SAFE |
|---|---:|---:|
| RISK | 1.000 | -0.067 |
| SAFE | -0.067 | 1.000 |

Two decorrelated sleeves at equal book weight cut the portfolio vol roughly
in half versus the risk sleeve alone, without sacrificing much return - that
is the diversification lift the ensemble buys you.

Individual signal return correlation (diagnostic standalone portfolios):

```
                S1_momentum  S2_trend  S3_lowvol_safe  S4_regime
S1_momentum           1.000     0.712          -0.145      0.551
S2_trend              0.712     1.000           0.074      0.667
S3_lowvol_safe       -0.145     0.074           1.000      0.020
S4_regime             0.551     0.667           0.020      1.000
```

Standalone signal Sharpes: S1 0.74, S2 0.77, S3 0.37, S4 0.43. All four
belong to the same market beta family and therefore can never be truly
orthogonal inside a long-only book; the empirical average pair-correlation
is 0.31, and that cap is what ultimately limits the combined Sharpe.

## What we tried and why it did not reach 2.0

- Equal-weighted long-only portfolios of LETFs have a realised Sharpe ceiling
  of roughly 1.0 in this universe. A passive EW of all 16 names is 0.67.
  A TQQQ-only passive buy-and-hold is 0.83.
- Single-signal trend-filtered momentum across the risk universe peaks at
  around Sharpe 0.8.
- A macro regime gate (VIX + HY OAS) adds ~0.1-0.2 Sharpe by removing the
  worst drawdown periods, but only because these two inputs co-move with the
  underlying equity book.
- Combining risk-sleeve + safe-sleeve portfolios gives the cleanest uncorrelated
  lift and is the core of ORION.
- Inverse-LETF shorts, cross-asset short-horizon reversal and breadth gates
  were all explored. They add noise and do not improve OOS.
- No daily vol-targeting (banned) and no survivorship / lookahead cheats.

## Files
- Code: `alt/orion_strategy.py`
- Metrics: `data/results/orion_metrics.json`
- Returns: `data/results/orion_returns.csv`
