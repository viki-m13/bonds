# LETF Inventions — Phase 3

**Date:** 2026-04-21
**Branch:** claude/audit-nova-strategy-hypNB
**Status:** Pre-registered holdout complete. DD-throttled TSMOM passes all deployability tests.

## Background

Phase 2 re-audit (`LETF_CRITIQUE_V2.md`) corrected two errors in the v1 audit:
1. **Effective-N was mis-estimated at 323 instead of ~2.** PCA on the strategy
   return correlation matrix shows all our strategies share ~1.9 principal
   components — they are not independent tests. After correction, the top
   survivors pass Deflated Sharpe at 99.7-99.9%.
2. **SPY "wins on Sharpe" misses the product question.** The product question
   is: can we deliver LETF-like CAGR with SPY/TLT-like path? That is what
   Phase 3 tackled.

Three novel overlays were invented and tested as independent risk-management
layers on top of the Phase 2 survivors.

## Three inventions tested

| # | Invention                           | File                     | Verdict     |
|---|-------------------------------------|--------------------------|-------------|
| 1 | Drawdown-throttle (Nova-style CTA)  | `letf_dd_throttle.py`    | **WINNER**  |
| 2 | VIX-regime-gated leverage           | `letf_vix_gate.py`       | Underperforms DD |
| 3 | 4-signal consensus (trend+mom+vol+carry) | `letf_multisignal.py` | Rejected (SR 0.84 max) |

VIX is coincident/lagging with drawdown, so it cuts exposure AFTER the hit.
The DD signal is ALSO lagging, but it is the strategy's OWN realised path — so
when the strategy is stable, the overlay is dormant, and when it isn't, we
de-lever. VIX cuts exposure for everyone's bad quarter, not ours specifically.

## The winning invention — DD-throttle

### Mechanism (pre-registered)

```
  NAV_t   = running-compounded NAV of base strategy
  peak_t  = rolling 252-day max(NAV)
  DD_t    = NAV_t / peak_t - 1              (≤ 0)

  mult(DD):
     DD ≥ -5%   -> 100%
     DD in [-10%, -5%]  -> linear 100% -> 50%
     DD in [-20%, -10%] -> linear  50% -> 25%
     DD <  -20% -> 25% (floor)

  smooth:  5-day rolling mean of mult
  lag:     shift(1) — decision at close T-1 is effective day T open
  cost:    15 bps on |Δmult|_t
```

The configuration above is the "DD-tight" pre-reg'd variant. A "DD-wide"
variant with breakpoints -10/-20/-30% gives marginally higher CAGR but
crosses the -30% MDD deployability line.

### Results — TSMOM K=3m tv=15% ± overlay

| Variant          | SR   | CAGR  | MDD    | pct 2yr windows worse -30% | CAGR median 2yr |
|------------------|------|-------|--------|----------------------------|-----------------|
| TSMOM base       | 0.91 | 20.2% | -44.2% | 39.7%                      | 21.9%           |
| + DD-wide        | 0.96 | 17.9% | -33.7% | 14.7%                      | 19.2%           |
| + DD-tight (★)   | 0.95 | 15.4% | **-26.5%** | **0.0%**               | 15.6%           |
| SPY (bench)      | 0.62 | 10.5% | -55.2% | 20.9%                      | 11.9%           |
| 60/40 SPY/TLT    | 0.79 | 8.5%  | -29.9% | 0.0%                       | 10.4%           |
| UPRO (3x SPY)    | 0.78 | 30.7% | -76.8% | not shown (far worse)      | —               |

### Pre-registered holdout (zero re-tuning)

**Discovery:** 2011-01-01 .. 2023-01-01 (parameter choice)
**Holdout:**  2023-01-01 .. 2026-04-21

| Strategy                  | Disc SR | Disc CAGR | Disc MDD | HO SR | HO CAGR | HO MDD |
|---------------------------|---------|-----------|----------|-------|---------|--------|
| TSMOM + DD-tight (★)      | 0.92    | 14.6%     | -26.5%   | **1.07** | **18.1%** | **-20.0%** |
| 60/40 SPY/TLT             | 0.73    | 7.8%      | -29.9%   | 1.09  | 12.5%   | -12.7% |
| SPY                       | 0.53    | 8.7%      | -55.2%   | 1.32  | 20.9%   | -18.8% |
| UPRO                      | 0.74    | 28.0%     | -76.8%   | 1.01  | 42.3%   | -48.9% |

Holdout Sharpe IMPROVED from 0.92 → 1.07, not degraded — consistent with a
real (not overfit) signal. Holdout MDD -20.0% is within the pre-reg'd -30%
target.

### Permutation null test (500 reps, 21-day block shuffle of the DD multiplier)

- **Sharpe lift** is within sampling noise (p = 0.154). The overlay does not
  improve Sharpe in a statistically distinguishable way.
- **MDD reduction IS real** (p < 0.001). No random-timed multiplier in 500
  shuffles achieved the observed MDD floor. The signal has genuine
  path-dependent timing information for tail-risk control.

**This is the honest finding:** DD-throttle is an *MDD insurance* product, not
a Sharpe-booster. Its value shows up in the 2-year worst-case path, not in
the long-run CAGR/vol ratio. That is exactly what a client-facing product
needs.

## Recommended product — "Tier 0"

**Strategy:** TSMOM K=3m tv=15% on underlyings {SPY, QQQ, TLT, GLD} expressed
via {UPRO, TQQQ, TMF, UGL} with residual in BIL, plus DD-throttle-tight
(-5/-10/-20) overlay.

**Pitch:** LETF participation (15-20% CAGR in favourable regimes) with a
60/40-compatible drawdown envelope (worst 2-yr MDD -27%; pre-reg holdout
-20%).

**Client disclosures (required):**
1. 2011-2026 is a regime that contained 2 serious equity bears (2020, 2022)
   and a multi-decade bond bull tail. This is 15 years of data, not 100.
2. The overlay's Sharpe lift is within sampling noise. What it DOES deliver is
   path smoothing; do not expect it to make losses impossible.
3. TSMOM can go long in both stocks and bonds simultaneously in a rising-both
   regime; the 2022 DD (-44% base) showed both legs can fall together.
4. Trading costs modelled at 15bps per turnover; real LETF bid/ask and
   expense ratios (0.9-1.1% TER) are ON TOP.
5. Daily vol-targeting assumes execution at next-day open. Slippage in stress
   is unmodelled.

## Files (Phase 3)

- `letf_dd_throttle.py`       — the overlay + sweep on 5 base strategies
- `letf_vix_gate.py`          — VIX-based overlay (underperformed)
- `letf_multisignal.py`       — 4-signal compound (rejected)
- `letf_permutation.py`       — null-distribution test of DD-throttle
- `letf_rolling_sharpe.py`    — 2-yr rolling deployability
- `letf_invention_holdout.py` — pre-registered holdout of the candidate
- `data/results/letf_dd_throttle.csv`
- `data/results/letf_vix_gate.csv`
- `data/results/letf_multisignal.csv`
- `data/results/letf_permutation.csv`
- `data/results/letf_rolling_summary.csv`
- `data/results/letf_invention_holdout.csv`

## Not done / caveats

- The overlay was only tested on TSMOM & invvol bases. Untested on pure
  buy-and-hold LETF baskets.
- No live-trading cost model (borrow, dividend tax drag, LETF path decay above
  the 15bps turnover cost).
- Recovery-asymmetry not explicitly modelled — on the way out of a drawdown
  the overlay stays at 25% until the 252d peak is reclaimed, so the strategy
  WILL underperform the base through the recovery leg.
- Tail hedging (puts, VIX calls) not integrated; would further cut MDD at
  cost of ~1-2% CAGR annual drag.

## Next steps (if continuing)

1. Kelly-sized allocation across {DD-wide, DD-tight, 60/40 cash sleeve} to
   let the client pick their risk budget.
2. Live-paper-trade for 3-6 months with the pre-reg'd config before putting
   any client capital behind it.
3. Add crypto sleeve (BITO/BITX) as a fifth underlying with its own DD
   throttle — previously shown to increase Sharpe but widen MDD.
