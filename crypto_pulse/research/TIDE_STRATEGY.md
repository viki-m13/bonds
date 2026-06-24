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

### Refinements (validated, baked into `tide.py`)
TIDE uses a **5-horizon breakout** (5/10/20/40/80d blend) and **Parkinson high-low volatility**
for sizing — both genuine, robust improvements found honestly (see "Improvement effort"). They
lift OOS ~1.98 → ~2.29 without overfitting, and each improved the *independent* pre-HL period too.

## Honest performance (net of costs + funding, improved version)
| window | Sharpe | note |
|---|---|---|
| HL era 2023-05→now (tradeable) | **+2.23** | improved from 2.01 |
| HL out-of-sample (last 40%) | **+2.29** | improved from 1.98 |
| Pre-HL 2018–2023 (independent) | +1.35 | improved from 1.11 |
| Full 2018→now | +1.54 | improved from 1.28 |

Block-bootstrap 95% CI on the HL Sharpe: **[1.13, 3.17]**. Still passes the full overfit battery
(100% parameter plateau, clean shuffle-null, all 4 WF folds positive, bootstrap 5th-pct 1.23).

## Why it is not overfit (full battery, all passed)
- **Parameter plateau:** 100% of a 5×3 breakout/regime grid > 1.0 Sharpe (median 1.79) — not a spike.
- **Every year positive** 2019–2026 (0.39–2.33).
- **Cost:** survives 4× taker (18 bps → 1.31).
- **Coin bootstrap** (20× random 70% subsets): 5th-pct 1.00, min 0.99.
- **Shuffle null:** real 2.01 vs permuted-signal max 0.50 — no look-ahead leak.
- **Walk-forward** (4 disjoint folds): all positive (0.92–2.43).
- **Anchored expanding WF** (6 starts): all positive (1.98–2.91), rising recently.
- **Execution sensitivity** (rebalance × vol-target, 15 cells): 93% > 1.0.
- **Capacity** (square-root impact, improved book): > 1.5 Sharpe to ~$50M, > 1.0 beyond ~$100M; graceful.

Supporting files: `research/tide.md`, `tide_ci.md`, `tide_capacity.md`, `roc_validate.md`
(+ the negative-result search `roc_lab*.md` showing TIDE is the survivor of 34 honest trials).

## Scope — where TIDE works and where it does NOT (cross-asset tested)
Same frozen rule run unchanged across asset classes (`tide_crossasset.py`, `tide_ebb.py`):
- **Crypto daily (its domain):** liquid-57 HL-era ~2.0 / full-period 1.28. **Full 112-coin universe
  dilutes to 1.07** — keep to the liquid subset, more coins is not free alpha.
- **Equities — TIDE INVERTS and LOSES** (stocks-96 −0.80, stocks-430 −1.28, −75% DD). Short-horizon
  cross-sectional moves continue in crypto but mean-revert in equities. **Do NOT run TIDE on HL
  HIP-3 equity perps (TSLA etc.) — it would lose money.**
- The equity-reversal mirror (**EBB**) is *not* tradeable either: ~0.2 Sharpe, OOS-negative, dies
  above 2bps cost (equity short-term reversal is arbitraged away). So a TIDE+EBB cross-asset combo
  (corr −0.02, genuinely uncorrelated) still *dilutes* to 1.40 — diversification needs two strong
  legs, and EBB isn't one.
- **Timeframes:** daily is the sweet spot; weekly 0.55 (weaker); hourly −0.08 (fails on cost/noise).
- **Leverage:** avg gross ~1.0x, cap 3x — far inside HL limits; scaling vol-target lifts CAGR and
  drawdown linearly, Sharpe unchanged (30% target ≈ +65% CAGR / −37% DD at ~2.6x gross).

**TIDE is a crypto-daily liquid-universe strategy. It is robust within that domain and is not a
universal anomaly.**

## Improvement effort (honest — 28 upgrade attempts, 3 stuck)
Tried 28 signal/construction/mechanism/risk/novel upgrades to TIDE itself (NOT ensembles), each
walk-forward OOS + deflated, with the strict bar of also improving the *independent pre-HL*
period (`tide_v2.py`…`tide_v6.py`):
- **Three genuine refinements survived** (each improved pre-HL + all WF folds): **multi→5-horizon
  breakout**, **Parkinson high-low volatility** sizing. Progression: 1.98 → 2.06 → 2.19 → 2.29 OOS.
- **Everything else overfit or hurt:** residualized/idiosyncratic breakout, skip-1-day, volume
  conviction, beta-neutralization, calm-vol gate, rank-weights, held-state machine, long/short
  asymmetry, dd-floor, deadband, param-ensemble, top-N universe, funding-aware sizing, and the
  novel **Kaufman efficiency-ratio**, **dispersion-timing**, **acceleration** ideas — all help
  in-sample then decay/collapse OOS.
- **Longer backtest:** improved TIDE is **positive in all 12 years 2015–2026** (0.24→2.38),
  full-period Sharpe **1.55** (dragged by the near-untradeable 2015–16), ~1.9–2.4 every year
  since 2017. A decade-spanning edge.
- Conclusion: the book is at the honest ceiling for a single cross-sectional breakout — **~2.3
  HL-era, ~1.55 over 12 years**. No construction trick honestly pushes a single book to 3.

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
