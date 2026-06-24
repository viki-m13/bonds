# Honest search for a Sharpe-3 strategy — master summary

Goal: a genuinely high-Sharpe, honestly-validated, deployable book. Validated causal, OOS,
net of fees/funding/slippage, with walk-forward + deflated Sharpe + cross-asset tests as
guards against overfitting. Repos used: whchien/ai-trader, je-suis-tm/quant-trading,
StockSharp/AlgoTrading.

## The deliverable: TIDE (~2.0, crypto-daily)
**TIDE** — Trend-Intensity-Dependent Exposure: cross-sectional market-neutral 20d breakout,
gross scaled by causal market trend-intensity. Built from the repos' breakout+trend ideas.
- HL-era Sharpe **2.01** (OOS 1.98), independent pre-HL **1.11** (t=3.2), full-period 1.28.
- Block-bootstrap 95% CI **[0.96, 2.93]**, P(Sharpe>1)=97%.
- Passes every overfit test: parameter plateau, every-year-positive, 4× cost, coin bootstrap,
  shuffle-null, 4-fold + anchored walk-forward, execution sensitivity, capacity to ~$25–100M.
- Scope (honest): **crypto-daily, liquid universe only.** Inverts on equities; weak weekly;
  fails hourly. Not a universal anomaly. Spec: `TIDE_STRATEGY.md`, code: `tide.py`.

## Why not Sharpe 3 — the wall, six ways
| route | result | file |
|---|---|---|
| 9 single price signals | cap ~1.4 OOS | `roc_lab.py` |
| Ensembles / signal blends | 0.7, underperform parts | `roc_lab.py` |
| Vol-managed CTA trend | 1.15 | `roc_lab3.py` |
| Regime-conditional → **TIDE** | **2.0** (winner) | `roc_lab5.py`, `tide.py` |
| Cross-validated robust select | 0.84 (still IS→OOS decay) | `roc_lab4.py` |
| Within-crypto diversification | 1.66 < TIDE (books co-move) | `tide_portfolio.py` |
| **Deflated Sharpe (34 trials)** | combos fail the 95% bar | `roc_lab*.py` |

Price signals are all variants of the same trend/momentum premium → mutually correlated →
can't diversify each other past ~2. Adding trials only lowers the deflated bar.

## Orthogonal diversifiers — exist but too weak to lift the combo
The honest lever toward 3 is a *strong, uncorrelated* second book. We found genuine
orthogonality but no strong second leg:
| candidate | corr to TIDE | own Sharpe | combo vs TIDE | file |
|---|---|---|---|---|
| EBB (equity reversal) | −0.02 | ~0.2 (OOS neg, dies >2bps) | 1.40 < 2.01 | `tide_ebb.py` |
| FLOW (daily taker flow) | +0.00 | ~0 (0/6 stable IS&OOS) | 1.89 < 2.01 | `flow_daily.py` |
Lesson (marginal-Sharpe rule): a near-zero-Sharpe book dilutes even at zero correlation.

## What is NOT deployable
- TIDE on HL HIP-3 **equity** perps (TSLA etc.) — it *loses* (equities mean-revert).
- EBB equity reversal — arbitraged away net of realistic costs.
- Daily aggregate order flow — mostly noise.
- Intraday breakout / pairs / XGBoost ensembles / intraday sleeves (earlier sessions).

## The one open lever: per-account L4 order flow
The 29h L4 tape showed whale-flow IC positive but not yet significant; the daily AGGREGATE
flow here is noise — together they say order-flow alpha (if any) lives in **per-account
granularity** (informed-wallet isolation), which needs the multi-week L4 recording still
running on GitHub Actions. It is a *fresh hypothesis on fresh data* — the only honest route
that could exceed ~2, and it inherits none of the 34-trial deflation debt.

## Honest bottom line
**TIDE (~2.0 crypto-daily) is the validated deliverable. Sharpe 3 is not honestly reachable
from any price, equity, or aggregate-flow data available now.** The only remaining credible
path is per-account L4 flow, gated on recording time, not on more ideas.
