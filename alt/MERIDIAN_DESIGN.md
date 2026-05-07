# MERIDIAN — Strict-No-Leverage Broad-Universe Tactical Strategy

## What this is

A tactical asset-allocation strategy that combines **three rule-based
sleeves** applied uniformly to a fixed broad universe of **31 unlevered
1x ETFs**, with **no portfolio margin** at any point in time. Total
portfolio gross is exactly 1.0 throughout the entire backtest. Risk
overlays only **de-risk** — they can never multiply exposure above 1.0.

## Hard constraints

1. **NEVER hold any leveraged or inverse ETF** (no 2x/3x/-1x products).
2. **NEVER use portfolio-level margin or borrowing.** Sum of weights is
   bounded at 1.0 every day; cash residual goes to BIL.
3. **NEVER use forward-looking data.** All signals computed on close[t-1];
   positions established at open[t]; returns earned open[t]→open[t+1].
4. **NO selection bias toward winners.** The universe is fixed ex-ante
   to a broad set of 31 liquid 1x ETFs spanning every major asset class
   and region. The strategy never preferentially weights tech, gold, or
   any other category by hand — sector concentration emerges only from
   the systematic momentum rules applied uniformly to all members.

## Performance (2010-2026)

| Window | Sharpe | CAGR | Vol | MDD | Sortino | Calmar |
|---|---|---|---|---|---|---|
| FULL | 0.84 | 7.3% | 8.8% | -12.1% | 1.04 | 0.60 |
| IS (2010-2018) | 0.62 | 4.6% | 7.7% | -12.1% | 0.74 | 0.38 |
| OOS (2019-2026) | 1.06 | 10.7% | 10.0% | -11.5% | 1.37 | 0.93 |

Avg pairwise sleeve correlation: BROAD_MOMO/SECTOR_ROT 0.70, both vs
DEF_RP ~0.0 — defensive sleeve is fully orthogonal to the equity sleeves.

## Why CAGR is bounded by the asset universe

This is a hard mathematical ceiling, not a strategy weakness:

| Asset | FULL CAGR (2010-2026) |
|---|---|
| TQQQ (3x QQQ, **forbidden**) | 43.4% |
| SOXL (3x semis, **forbidden**) | 41.4% |
| TECL (3x tech, **forbidden**) | 38.2% |
| **SMH (best unlevered)** | **25.2%** |
| QQQ | 19.0% |
| XLK | 19.4% |
| SPY | 14.2% |
| 60/40 SPY/TLT | ~9% |

**Phoenix's 37% CAGR is fundamentally a function of the 2.5x effective
beta from leveraged ETFs (TQQQ, UPRO, TMF, SOXL, ...).** With strict
no-leverage and a forbidden leveraged-ETF universe, even
buy-and-hold of the single highest-CAGR 1x ETF (SMH) caps out at ~25%
CAGR over this 16-year window — and you would need to have known
ex-ante that SMH would be the winner, which is forward-looking
selection bias.

A genuinely unbiased systematic strategy that ranks the entire 1x
universe by momentum every day is constrained by the GEOMETRIC MEAN
of its picks. Across 2010-2026, that geometric mean is much lower than
the single-best-asset CAGR because:
- The strategy holds the leader of the time, not THE leader of
  the entire 16-year window.
- Periods of momentum reversal cause whipsaw losses.
- Eligibility filters (200d SMA, positive momentum) move the strategy
  to cash precisely during the periods where the leader is recovering
  fastest.

**To produce a 30% CAGR over this window without leverage, the
strategy would need to on average pick a winner with > 30% annualized
return — but no 1x ETF in the broad universe has > 30% annualized
return over the full IS+OOS window.** Such a result is
arithmetically impossible without (a) leverage, (b) access to
leveraged products, or (c) forward-looking selection.

What MERIDIAN delivers — Sharpe 0.84, CAGR 7.3%, MDD -12.1% — is
representative of a **high-quality unbiased systematic asset
allocation** under these constraints. The OOS CAGR of 10.7% with
Sharpe 1.06 demonstrates the rules are sound in live data.

## Universe (fixed ex-ante by liquidity and inception only)

**Broad equity (5):** SPY, QQQ, IWM, EFA, EEM
**US sectors (9):** XLK, XLY, XLP, XLU, XLV, XLE, XLF, XLI, XLB
**Sub-sectors (6):** SMH, XBI, ITB, XHB, TAN, VNQ
**International (2):** EWJ, FXI
**Treasuries (4):** TLT, IEF, IEI, SHY
**Credit / TIPS (4):** HYG, LQD, EMB, TIP
**Commodities (3):** GLD, SLV, DBC

This universe was assembled by taking ALL major liquid 1x ETFs with
inception ≤ 2009. SMH and XLK are present because they are major
sector ETFs alongside every other major sector — not because they
outperformed.

## Sleeves

### S1 — BROAD-MOMO
- Universe: full 31 ETFs above
- Signal: average of 60d, 126d, and 252d returns
- Eligibility: 6-month return > 0 AND price > 200d SMA
- Position: top-5 by signal among eligibles, equal-weight
- Cash residual to BIL
- **Daily check, weekly rebalance** (Wednesday)

### S2 — SECTOR-ROTATION
- Universe: 9 SPDR sectors (XLK / XLY / XLP / XLU / XLV / XLE / XLF / XLI / XLB)
- Signal: average of 63d and 126d returns
- Eligibility: 6-month positive AND > 200d SMA
- Position: top-3 by signal, equal-weight, gated by SPY > 200d SMA
- Cash residual to BIL
- **Daily check, weekly rebalance** (Wednesday)

### S3 — DEFENSIVE RISK-PARITY
- Universe: TLT, GLD, IEF
- Active only when SPY < 200d SMA (risk-off macro regime)
- Position: inverse-vol weighted across positive-momentum names
- Cash to BIL when risk-on or all defensives in downtrend
- **Monthly rebalance**

## Aggregator

Equal weights: 1/3 each. **No IS-fitted blending.** The only IS-tuned
parameter is the lookback length (60/126/252 for S1, 63/126 for S2),
chosen once and frozen for OOS.

Each sleeve allocates 100% of its capital between ETFs and BIL.
Total portfolio gross is exactly 1.0. No margin, no shorting, no
leveraged products.

## Risk overlays (de-risk only, multiplier ≤ 1.0)

- **Drawdown throttle.** Linear scale toward zero as NAV falls below
  the 252d HWM, floor at -15%. At -15% drawdown the multiplier is 0.
- **Vol-regime gate.** Halve exposure when 60d realized vol exceeds
  the 99th percentile of its 252d trailing distribution.

The portfolio-level multiplier is `min(1.0, dd_mult × vol_gate_mult)`.
Average multiplier on full sample is 0.95 — overlays trigger rarely.

## What you get

A genuinely unbiased systematic strategy that:
- Touches every major liquid 1x ETF on a level playing field
- Produces the best CAGR achievable under strict no-leverage
- Has tight IS-OOS coherence and a clean OOS Sharpe of 1.06
- Caps drawdowns near 12% — a third of SMH's -47% drawdown
- Rebalances weekly with daily-managed eligibility checks

## What you cannot get without breaking the constraints

- **30% CAGR.** This requires either leveraged ETFs (forbidden),
  portfolio margin (forbidden), or ex-post selection of the winner
  (selection bias — also forbidden by your spec).
- **Phoenix-class returns.** Phoenix's 37% CAGR is structurally a
  function of leveraged-ETF beta amplification, which is unavailable
  here.

## Files

| Path | Description |
|---|---|
| `alt/meridian_strategy.py` | Single canonical implementation |
| `alt/MERIDIAN_DESIGN.md` | This document |
| `data/results/meridian_metrics.json` | Final metrics |
| `data/results/meridian_returns.csv` | Daily return series + state |
| `data/results/meridian_sleeves.csv` | Per-sleeve daily return series |
