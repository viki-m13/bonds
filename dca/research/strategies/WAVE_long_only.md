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
| **ML N12 + runner-gate + SIGNAL-ACCELERATION (CHAMPION)** | **21.5%** | **1.41** | **−17.8%** |
| ML N12 + runner-gate (prior) | 19.6% | 1.36 | −19% |
| ML N12 equal-rebalance | 21.3% | 1.34 | −22% |

**CHAMPION = ML rank + runner-gate (mom3>0) + signal-acceleration gate
(ML score rising: prob_t > prob_{t-2}).** Sub-period: 2015-18 Sharpe 1.64 (QQQ
0.82), 2019-21 1.85 (=QQQ), 2022-25 **1.11 (QQQ 0.66)** — beats/matches QQQ every
era. Script: `exp83_creative.py`.

### Signal-acceleration — the creative edge (exp83)
Buying names whose **ML conviction is *rising*** (not just high) lifted Sharpe
1.24→1.41 and CAGR to 21.5%. Improving stocks outrun already-strong ones (a
2nd-derivative / "fundamental+technical momentum" effect). What did NOT help:
conviction-weighting (neutral), vol-calibrated stops (slightly worse), and
**blending insider/13F into the score (hurt — 0.91; the ML is already the optimal
combiner, don't dilute its rank — use extra signals as GATES, not score blends).**

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

## Tested overlays that DID NOT help (exp80) — don't relearn
On the base ML N12 runner-gate (Sharpe 1.36), three standard enhancements were
tested and ALL hurt:
- **PEAD boost** (tilt to fresh revenue-surprise names): Sharpe 1.36 -> 0.83. The
  ML picker already ingests revenue-accel features; an extra boost distorts the rank.
- **Vol-scaled sizing** (1/vol weighting): 1.36 -> 1.33 (neutral/slightly worse;
  the runner-gate already controls vol).
- **Market-regime throttle** (cash when QQQ<10mo-MA): 1.36 -> 1.16 (50%) / 1.06
  (0%). Whipsaw + missed V-recoveries in a strong-uptrend decade.
- All combined: 0.60. => Base WAVE is already well-tuned; leave these off.

## 13F institutional accumulation (exp81-82) — modest add
Downloaded 16 quarters of SEC 13F holdings (78k CUSIPs, mapped 2,800 → ticker via
OpenFIGI; data in `data/sec/_13f_cusip.pkl` + `_13f_cusipmap.pkl`). Tested as a
cross-sectional signal (2023-2025, 1,933 names):
- inst. **value growth** IC(fwd3m) +0.063, L/S Sharpe 1.98 — BUT inflated: value =
  shares×price, so it's partly momentum (price-contaminated).
- inst. **breadth change** (Δ #managers, the clean measure) IC +0.037, L/S Sharpe
  0.65 — modest, on par with insider/rev-accel.
Verdict: real but modest; only 33 months of dense data. Fold the breadth-change
into the ML feature set when more history is available; don't over-weight.

## Roadmap (still open)
1. Extend 13F history (datasets go back to 2013) + add breadth-change to ML features.
2. News/8-K hard-catalyst gate (needs 8-K event fetch).
3. Extend ML history pre-2015 (needs earlier fundamentals coverage).
4. Sector/industry context features.
