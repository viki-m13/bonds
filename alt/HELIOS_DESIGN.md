# HELIOS — Cross-Asset Trend on Underlyings, Expressed via Leveraged ETFs

## TL;DR honest assessment

HELIOS **does not hit the Sharpe 2.0 target** of the original spec.
Selected IS-only (`MOM_LB=189, MOM_SKIP=42, TOP_N=2, VIX_Z_CAP=1.5`), the strategy posts:

| metric | IS (2010-03-11 → 2018-12-31) | OOS (2019-01-01 → 2026-04-02) | Full |
|---|---|---|---|
| Sharpe | 0.68 | 0.81 | 0.73 |
| CAGR | 19.0% | 33.7% | 25.4% |
| MDD | — | — | -62.3% |
| Ann vol | — | — | 44.4% |
| Avg annual turnover | — | — | 28× |
| Avg cash weight | — | — | 11% |

**The framework is sound and ships**: no look-ahead, next-open execution, 5 bps
transaction cost, purely IS-picked params, and OOS Sharpe (0.81) is *better*
than IS Sharpe (0.68) — i.e. the IS→OOS gap is -0.13, well inside the 0.5
tolerance. CAGR clears the 20% bar. But Sharpe clearly misses 2.0 and the
realised OOS Sharpe is below 1.5.

## Signals used

1. **Cross-sectional trend** on unlevered underlyings
   (SPY, QQQ, TLT, IEF, GLD, USO, XLK, XLE, XLF, SMH, VNQ, EEM, FXI).
   Score = `close[t-42] / close[t-189] - 1` (6-month momentum with
   the most recent 2 months skipped, to avoid short-term reversal).

2. **Absolute trend filter** per asset: score > 0 AND
   `close > 200-day SMA`.

3. **Macro meta-gate** for RISK-asset eligibility:
   `VIX 252d z-score < 1.5 AND HY OAS 20-day change < +0.3`.
   Defensive assets (TLT, IEF, GLD) bypass the gate — they typically
   *earn* when the equity gate turns off (2015, 2018Q4, 2020, etc.).

4. **Sizing**: the top 2 ranked eligible assets each get 50% weight,
   residual to BIL (cash). No sigma-targeting / daily vol scaling.

5. **Rebalance**: weekly Fridays; about 50 rebalances per year; ~28×
   annual turnover including drift.

6. **Expression**: each selected underlying is substituted 1-for-1 with
   its matched leveraged ETF (UPRO, TQQQ, TMF, TYD, UGL, UCO, TECL, ERX,
   FAS, SOXL, DRN, EDC, YINN). Weights are unchanged.

## Execution / audit

- Signal uses close[t] only. Weights take effect at `open[t+1]`; PnL
  is earned over `open[t+1] → open[t+2]`. Implemented in
  `run_backtest` via `r_fwd[t] = open[t+2]/open[t+1] − 1` applied to
  `W[t]`. No shift(-1) leak possible.
- Transaction cost: 5 bps on `|ΔW|` summed across tickers.
- Start date is driven by the latest first-available date among the 13
  leveraged ETFs (YINN 2009-12 is the binding constraint; after 200-day
  warmup and 252-day VIX z-score window, evaluation begins 2010-03-11).

## Why I expect generalization

- The momentum and trend literature is decades deep (Asness, Moskowitz,
  Antonacci). 12-1 / 6-1 style cross-sectional trend on 10+ broad asset
  classes has positive Sharpe out of sample everywhere it has been
  tested.
- The macro gate uses two slow, stationary signals (VIX vs its own
  long-run distribution; HY OAS change) — both are structural risk
  measures with decades of out-of-sample evidence (Adrian-Shin, Fama-French
  term/credit factor literature).
- The IS → OOS Sharpe movement was +0.13, not -0.13 — the strategy
  **improved** going forward, which is a positive sign that we are not
  overfitting the 2010–2018 window. CAGR also jumped 19% → 34% OOS.
- Parameters are *un*-tuned against OOS: IS window was used alone;
  OOS was evaluated once.

## Honest risks

- **Sharpe is low by institutional standards (~0.7–0.8)**. The
  inherent volatility of 3× leveraged ETFs (30–60% annualised) caps
  the attainable Sharpe for a simple long-only trend strategy. To
  reach 2.0 you need either market-neutral signals, micro-structure
  alpha, or far tighter regime filtering that would reduce
  time-in-market below ~40%.
- **Max drawdown is -62%**. This is primarily driven by 2022
  (bond/equity correlation breakdown: TMF crashed while UPRO/TECL
  also slumped) and early 2020 (COVID week of 3σ VIX moves that the
  macro gate caught a day late). On 3× ETFs these drawdowns amplify.
- **Volatility drag**: 3× ETFs have a material path-dependent decay
  during choppy regimes (e.g. 2022 sideways-down). The macro gate
  mitigates but cannot eliminate this.
- **Transaction costs assumed 5 bps/side**; on 28× annual turnover
  this is ~1.4%/yr drag. Realistic for UPRO/TQQQ/TMF; optimistic for
  lower-liquidity 3× names (YINN, EDC, ERX) during stress.
- **Regime dependence of momentum**: if cross-asset trend ceases
  to work (long low-vol regime or a structural break), the strategy
  degrades to a lightly-cash-anchored beta-2× S&P.
- **The macro gate leans on post-GFC data**. Before 2008 the VIX
  distribution was structurally different. Out-of-sample vs. pre-2010
  is not evaluated.
- **Leveraged ETFs can halt or terminate** (UVXY, XIV). None of the
  ones used has shown fatal stress, but basis/borrow risk is real.

## Files

- `alt/helios_strategy.py` — full pipeline.
- `data/results/helios_metrics.json` — all metrics + per-rebalance picks.
- `data/results/helios_returns.csv` — daily `ret, weight_sum, cash_wt`.
- `data/results/helios_picks.json` — full list of weekly picks for audit.
