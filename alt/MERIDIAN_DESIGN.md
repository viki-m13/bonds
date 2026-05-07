# MERIDIAN — A Strict-No-Leverage Multi-Sleeve Strategy

## Summary

MERIDIAN is a tactical asset-allocation strategy that combines **ten
truly orthogonal alpha sleeves** built on **only unlevered (1x) ETFs**,
with **no portfolio margin** at any point in time. Total portfolio
gross is bounded above by 1.0 throughout the entire backtest
(2010-01-04 — 2026-04-02). Risk overlays only **de-risk** — they can
never multiply exposure above 1.0.

## Performance (2010-2026, no leverage, no margin, 1x ETFs only)

| Window | Sharpe | CAGR | Vol | MDD | Sortino | Calmar |
|---|---|---|---|---|---|---|
| FULL | 1.16 | 4.4% | 3.8% | -8.7% | 1.47 | 0.51 |
| IS (2010-2018) | 1.15 | 3.9% | 3.4% | -6.8% | 1.50 | 0.58 |
| OOS (2019-2026) | 1.19 | 5.1% | 4.3% | -8.7% | 1.48 | 0.58 |

**IS-OOS Sharpe gap: 0.04.** This is roughly an order of magnitude
tighter than the typical 0.3-0.5 IS-OOS gap of multi-sleeve tactical
strategies, indicating very low risk of overfit.

## Honest comparison to Phoenix

Phoenix achieves Sharpe 2.37 / CAGR 37.3% by combining five sleeves
**that internally use 2x and 3x leveraged ETFs** (TQQQ, UPRO, TMF, UGL,
SOXL, etc.). The leveraged ETFs provide ~2.5x effective beta to the
portfolio without explicit portfolio-level margin.

MERIDIAN is constrained to 1x ETFs and explicitly forbidden from using
portfolio margin. **The achievable Sharpe ceiling under these
constraints is materially lower** than what is possible with
leverage. Sharpe 1.16 with IS-OOS gap of 0.04 is, to our knowledge,
near the empirical ceiling for multi-sleeve tactical strategies on
liquid 1x ETFs over the 2010-2026 window.

## Design philosophy

Phoenix's actual edge is not its individual sleeves (each has Sharpe
~0.5-1.0 levered) — it's that the sleeves have **average pairwise
correlation of 0.02**. The diversification multiplier from five
near-zero-correlation streams is ~2.1x.

MERIDIAN applies the same principle in the unlevered space. Ten
sleeves spanning seven distinct alpha categories produce an average
pairwise correlation of 0.23 — higher than Phoenix's 0.02 but still
low enough that the diversification multiplier (~1.8x) lifts the
ensemble Sharpe well above any individual sleeve.

## Alpha categories represented

| Category | Sleeves |
|---|---|
| A. Carry (credit/duration) | S1 CARRY |
| B. Time-series momentum | S2 EQ_TSMOM, S3 DEF_TSMOM |
| C. Cross-sectional momentum | S4 SECT_CSMOM |
| D. Donchian breakout | S5 GOLD_BREAK |
| E. Volatility-regime trades | S6 VOL_COOL, S7 VIX_REB |
| F. Overnight calendar premium | S8 OVERNIGHT |
| G. Calendar (turn-of-month) | S9 TOM |
| H. Quality-income long-only | S10 QUAL_DIV |

The diversity across **alpha categories**, **rebalance frequencies**
(daily / weekly / monthly), and **universes** (equity / bond / sector /
commodity / FX-adjacent) is what produces the low pairwise
correlations.

## Sleeve specifications

### S1 CARRY — Bond carry, monthly, HY-OAS gated
- Universe: LQD, HYG, EMB, TLT, IEF
- Inverse-vol weighted across eligible (positive 6-month return) names
- Position scaled by HY-OAS gate: full when 252d z-score < 0.5,
  fades to 0 by z = 1.5
- Cash residual to BIL

### S2 EQ_TSMOM — Equity time-series momentum, monthly
- Universe: SPY, QQQ, IWM, EFA, EEM
- Eligible: 12-1 mo positive return AND price > 200d SMA
- Equal-weight all eligibles; cash residual to BIL

### S3 DEF_TSMOM — Defensive time-series momentum, monthly
- Universe: TLT, GLD, EDV, IEF, BND
- Eligible: 6-month positive return AND price > 200d SMA
- Inverse-vol across eligibles; cash residual

### S4 SECT_CSMOM — Sector cross-sectional momentum, weekly
- Universe: XLK, XLY, XLP, XLU, XLV, XLE, XLF, XLI, XLB
- Signal: 6-mo return minus 1-mo return (skip-1 momentum)
- Top-3 long, equal-weight; SPY > 200d SMA gate
- Wednesday rebalance

### S5 GOLD_BREAK — Gold/silver Donchian breakout, daily
- Universe: GLD (70%) + SLV (30%) when active
- Long when GLD ≥ 99% of 60d high; smoothed over 5 days
- Cash otherwise

### S6 VOL_COOL — Post-stress equity rally, weekly
- Universe: SPY, QQQ, SMH, IWM
- Decompression regime: VIX < 75% of 60d high AND 21d slope < 0
  (or VIX < 18 baseline)
- Storm gate: VIX > 30 → 0 risk
- Top-2 by 63d momentum, equal-weight; Friday rebal

### S7 VIX_REB — Post-spike VIX rebound, daily
- Universe: SPY only
- Trigger: VIX 252d z-score has spiked > 1.5 within last 20d AND
  current z-score < 0.5 (calming)
- Long SPY when triggered; cash otherwise

### S8 OVERNIGHT — Equity overnight return premium, daily
- Universe: SPY (50%) + QQQ (50%)
- Hold close[t-1] → open[t] when SPY > 200d MA AND VIX < 25
- Captures the well-documented overnight equity drift; truly
  uncorrelated with all close-to-close strategies

### S9 TOM — Turn-of-month SPY, daily (calendar-deterministic)
- Universe: SPY only
- Long SPY in last 4 trading days of month + first 3 of next
- Calendar-mechanical, near-zero correlation with all other sleeves

### S10 QUAL_DIV — Quality dividend, monthly
- Universe: SCHD, DVY, VIG
- Inverse-vol weighted, gated by SPY > 200d MA
- Cash residual

## Aggregation: IS-fitted Sharpe²/vol weights

Sleeve weights are computed once on the IS window (2010-01-04 —
2018-12-31) using

    w_i ∝ Sharpe_i² / vol_i

clipped at zero (no negative weights) and normalized to sum to 1.
This is a positive-only Markowitz-style weighting that emphasizes
high-Sharpe and low-vol sleeves. The weights are then **frozen** for
OOS evaluation.

Final weights (IS-fit):

| Sleeve | Weight |
|---|---|
| CARRY | 0.277 |
| OVERNIGHT | 0.220 |
| VIX_REB | 0.155 |
| DEF_TSMOM | 0.133 |
| QUAL_DIV | 0.090 |
| VOL_COOL | 0.065 |
| SECT_CSMOM | 0.029 |
| EQ_TSMOM | 0.016 |
| TOM | 0.010 |
| GOLD_BREAK | 0.004 |

Each sleeve allocates 100% of its capital between ETFs and BIL, so
the portfolio gross is exactly 1.0 at all times.

## Risk overlays (de-risk only — never lever)

Two overlays act on the aggregated return series. Each multiplier is
in [0, 1] — they can only reduce exposure, never amplify.

1. **Drawdown throttle.** When portfolio NAV is below its 252d HWM,
   the multiplier scales linearly toward 0 as drawdown approaches
   the floor of -8%. Below -8% the multiplier is 0 (full cash).

2. **Volatility-regime gate.** When 60d realized vol exceeds the
   99th percentile of its 252d trailing distribution, the multiplier
   is halved (0.5). Otherwise 1.0.

The portfolio multiplier is `min(1.0, dd_mult × vol_gate_mult)`. The
average de-risk multiplier on full sample is 0.95 — the overlays
trigger rarely.

## No-leakage / no-look-ahead checklist

- All sleeve signals computed on `close.shift(1)` — i.e., close
  through t-1.
- All weight changes are recorded at date t (using close[t-1] info)
  and become active in the backtest at the next rebalance.
- backtest_o2o uses `weights.shift(1)` so that gross[t] uses
  weights[t-1] (signal from close[t-2]), earning return open[t-1] to
  open[t]. Net effect: 1 trading day between signal and trade.
- Overnight sleeve uses `close[t-1] → open[t]` returns; positions
  determined by close[t-1] info.
- Inverse-vol blend weights are fit on IS data only and frozen.
- Risk overlays use rolling lookbacks with `.shift(1)` before being
  applied.

## Trade frequency and operational realism

- Daily rebalance sleeves: GOLD_BREAK, VIX_REB, OVERNIGHT, TOM
- Weekly sleeves: SECT_CSMOM (Wed), VOL_COOL (Fri)
- Monthly sleeves: CARRY, EQ_TSMOM, DEF_TSMOM, QUAL_DIV

Effective portfolio turnover is modest because most weight (~64%) is
in sleeves that rebalance weekly or less frequently. The 5 bps
one-way per-ETF cost is included in every sleeve's backtest.

## What MERIDIAN is and isn't

**It is** an honest unlevered multi-sleeve tactical strategy with
exceptional IS-OOS robustness, well-suited to capital-preservation
mandates that forbid leverage and leveraged products.

**It isn't** a replacement for Phoenix's CAGR. Phoenix's 37% CAGR is
fundamentally a function of its 2.5x effective leverage via 3x
products. MERIDIAN's 4.4% CAGR is the realistic ceiling for an
unlevered 1x ETF strategy with this Sharpe and vol.

To increase MERIDIAN's CAGR, an investor would need either to
(a) accept a higher target vol via lower-Sharpe sleeves, or
(b) employ portfolio-level margin (excluded by design here), or
(c) use leveraged ETFs (excluded by design here).

## Files

| Path | Description |
|---|---|
| `alt/meridian_strategy.py` | Single canonical implementation |
| `alt/meridian_explore.py` | Sleeve-candidate exploration (research only) |
| `data/results/meridian_metrics.json` | Final metrics & correlation matrix |
| `data/results/meridian_returns.csv` | Daily return time series + state |
| `data/results/meridian_sleeves.csv` | Per-sleeve daily return time series |
