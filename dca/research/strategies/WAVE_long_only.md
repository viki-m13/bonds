# WAVE — Long-Only Concentrated Stock-Picker (NO margin, NO shorting)

> **Status: CURRENT deployment track.** Long-only, no leverage, no shorting —
> matches the live mandate. Survivorship-clean PIT backtest.
> Scripts: `dca/research/exp79_ml_longonly.py` (picker), `exp77_runners.py`
> (runner mechanics), `exp70_composite.py` (factor composite).

## The strategy
Pick a concentrated book of ~12 names each month from the **ML stock-picker**
(walk-forward HistGBM trained only on past data, ranking the cross-section on 36
fundamental+technical+insider features), gated to names that are **already moving
("runners")** and in an uptrend, then **ride winners and cut losers**.

**Rules:**
1. **Universe:** US stocks, price ≥ $3, above 10-month MA (uptrend gate).
2. **Select:** top ~12 by ML probability **AND** 3-month momentum > 0
   (runner-gate — only buy names the market is already rewarding).
3. **Size:** equal-weight at entry; *ride* (let winners grow) or equal-rebalance.
4. **Cut losers:** trailing stop −30% from peak, exit on close below 10-mo MA.
5. **Rebalance monthly**; refill empty slots with the best non-held runners.

## Validated results (PIT survivorship-clean, 2015–2025*)
| Config | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|
| QQQ | 18.5% | 1.03 | −33% |
| **ML N12 + runner-gate (best Sharpe)** | 19.6% | **1.36** | **−19%** |
| ML N12 equal-rebalance (best CAGR) | 21.3% | 1.34 | −22% |
| ML N8 (more concentrated) | 19.2% | 1.16 | −25% |

\*2015–2025 because the ML picker needs prior history to train. On 2012–2025 the
non-ML long-only multi-sleeve blend (moonshot + composite) did 22% / Sharpe 1.28
(`exp71_blend.py`) — a good pre-2015 stand-in.

Sub-period (best blend variant): 2015-18 Sharpe 1.40 (QQQ 0.82), 2019-21 1.75
(QQQ 1.85), 2022-25 0.75 (QQQ 0.66) — beats QQQ in 2 of 3 eras, risk-adjusted.

## What worked / what didn't (so we don't relearn)
- **Runner-gate (buy what's already moving) is the key edge** — lifted Sharpe
  1.26 → 1.36 and cut drawdown to −19%. "Concentrate into runners" works.
- **Cut losers helps momentum names** (trailing stop + trend exit) but **hurts
  quality/value names** (whipsaw) — apply exits to the momentum book only.
- **Aggressive staged culls** (cut if not +X% by 3/6m) whipsaw long-only — gentle
  trailing/trend exits are better.
- **Blending the ML picker with the slow composite HURT** (composite-ride dragged);
  the concentrated ML runner alone is best.
- ML beats the hand-built factor composite for *selection*; concentration to
  ~12 names is the sweet spot (N=8 too noisy, N=20 dilutes).

## Roadmap
1. Earnings-drift & post-event runner entries; news/8-K catalyst gate.
2. 13F institutional accumulation as a selection feature.
3. Volatility-scaled position sizing; regime exposure throttle.
4. Extend ML history pre-2015 (needs earlier fundamentals coverage).
