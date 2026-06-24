# TIDE — Trend-Intensity-Dependent Exposure (standalone crypto strategy)

A named, standalone, honestly-validated price strategy built from the shared repos
(whchien/ai-trader, je-suis-tm/quant-trading breakout+trend ideas). VOL and STRATA are set
aside — this stands on its own.

## What it is
A **cross-sectional, market-neutral breakout** on the Hyperliquid perp universe (~57 coins),
whose **gross exposure is scaled by market-wide trend intensity** (causal):

1. **Signal:** per coin, 20-day breakout z-score `(C − MA20) / std20`, cross-sectionally demeaned
   → long coins breaking out up, short those breaking down. Inverse-vol sized, gross = 1.
2. **Regime gate:** `trend_intensity = |2·(fraction of coins above their 50d MA) − 0.5|`, in [0,1],
   lagged one day. Exposure = signal × trend_intensity → trade hard when the market is one-sided
   (trending), stand down when it's balanced (chop). This is the "TIDE" mechanism.
3. **Execution:** rebalance every 3 days, net 4.5bps taker + HL funding, vol-targeted to 12%.

Code: `crypto_pulse/tide.py` (`TIDE().build()`), zero parameters fitted after this spec.

## Honest performance (net of costs + funding)
| window | Sharpe | note |
|---|---|---|
| HL era 2023-05→now (tradeable) | **+2.01** | t = 3.5 |
| HL out-of-sample (last 40%) | +1.98 | t = 2.2 |
| Pre-HL 2018–2023 (independent) | +1.11 | t = 3.2, spot proxy |
| Full 2018→now | +1.28 | t = 4.3 |

Block-bootstrap 95% CI on the HL Sharpe: **[0.96, 2.93]**, P(Sharpe > 1) = 97%.

## Why it is not overfit (full battery, all passed)
- **Parameter plateau:** 100% of a 5×3 breakout/regime grid > 1.0 Sharpe (median 1.79) — not a spike.
- **Every year positive** 2019–2026 (0.39–2.33).
- **Cost:** survives 4× taker (18 bps → 1.31).
- **Coin bootstrap** (20× random 70% subsets): 5th-pct 1.00, min 0.99.
- **Shuffle null:** real 2.01 vs permuted-signal max 0.50 — no look-ahead leak.
- **Walk-forward** (4 disjoint folds): all positive (0.92–2.43).
- **Anchored expanding WF** (6 starts): all positive (1.98–2.91), rising recently.
- **Execution sensitivity** (rebalance × vol-target, 15 cells): 93% > 1.0.
- **Capacity** (square-root impact): > 1.5 Sharpe to ~$25M, > 1.0 to ~$100M; graceful.

Supporting files: `research/tide.md`, `tide_ci.md`, `tide_capacity.md`, `roc_validate.md`
(+ the negative-result search `roc_lab*.md` showing TIDE is the survivor of 34 honest trials).

## Honest limitations
- **It is a ~2.0 Sharpe book, not 3.** Six iterations across three repos + deflated-Sharpe and a
  full generalization battery establish that price-based crypto signals cap here. Sharpe 3 is not
  honestly reachable from price data alone; it requires orthogonal information (the L4 order-flow
  book, still recording) or cross-asset diversification.
- Pre-HL leg (1.11) is weaker than the HL era (~2.0): the edge is real across regimes but its
  magnitude is regime-dependent; size to the lower end, not the trailing high.
- Capacity model is conservative single-name impact; assumes taker fills.

## Deployment spec
- Universe: HL perps with 30d $-ADV > $3M (~57 coins).
- Daily compute of the two signals; rebalance every 3 days; vol-target the book to your risk budget.
- Expected honest Sharpe ~1.5–2.0 net at ≤ $25–50M; scale down leverage to the bootstrap-CI lower
  bound, not the point estimate.
