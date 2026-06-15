# META — meta-labeling layer on 12-1 momentum

Code: [`../strategy_meta.py`](../strategy_meta.py); scores cached to
`meta_scores.parquet`.

## The idea (why it's different from the failed return-rankers)

The CNN / PatchTST / LightGBM-ranker / Chronos all tried to predict *which*
stock outperforms and hit IC ≈ 0. Meta-labeling (López de Prado, *AFML* ch.3)
asks the narrower, easier question instead: **given that 12-1 momentum already
nominated a name, will this bet beat QQQ over the next quarter?** The base
signal keeps doing the selecting; a LightGBM classifier only decides which of
the top-15 momentum leaders to keep — and gets **regime context** the
cross-sectional models never saw (SPY vs 200dma, breadth, VIX, term spread,
cross-sectional dispersion).

* Candidate pool: top-15 eligible names by 12-1 momentum each date.
* Binary meta-label (train only): the name's forward-63d return > QQQ's.
* 16 features: 10 cross-sectional (momentum 12-1/6-1/3-1, vol 63/126, beta,
  dist-from-52wk-high, drawdown, vol-of-vol, dollar-volume trend; all
  rank-transformed within date) + 6 regime (SPY gap, breadth, VIX, VIX change,
  term spread, dispersion).
* Walk-forward: LightGBM refit every 126 td on labels closed ≤ fit date;
  predict strictly after. Scores 2009 → present.

## Result 1 — reordering momentum: FAILS (worse than the base signal)

Active-period DCA grid (180 windows, quarterly starts 2010+, biweekly, 5 bps):

| signal | win vs QQQ | win vs SPY | median vs QQQ | worst vs QQQ |
|---|---|---|---|---|
| META (meta-labeled) k=2 | 53% | 78% | +1.4% | -37.9% |
| META (meta-labeled) k=3 | 48% | 77% | -0.5% | -35.1% |
| **raw 12-1 momentum k=2** | **69%** | **91%** | **+11.5%** | **-22.4%** |
| **raw 12-1 momentum k=3** | **69%** | **89%** | **+10.6%** | **-19.9%** |

Meta-labeling to *reorder* the momentum leaders **destroys** alpha: −16pp
win-rate, −12pp median, and a *worse* worst case. Gain-importance explains why —
the top features are `beta252`, `term`, `disp`, `dd`, `vix` (momentum itself is
~0.05): the classifier learns a **defensive low-beta/regime tilt**, which lags
the mega-cap tech bull that dominates 2010-2026. This is the *same failure mode*
as the earlier LightGBM return-ranker (`results_ml.md`): any model layer that
re-weights momentum toward "safer" names subtracts from the one thing that
works.

## Result 2 — timing gate: marginal, NOT significant

The pool-average meta-probability *does* line up weakly with whether momentum is
about to work, as a **market-timing** (not selection) signal:

* `corr(mean pool meta-prob, momentum top-3 forward-63d excess vs QQQ)` =
  **+0.055** (p = 0.11, n = 865 overlapping windows → effective N much smaller).
* High-gate half: momentum top-3 beats QQQ by **+2.4%/qtr**; low-gate half:
  **−0.4%/qtr**.

Directionally sensible (stand down when few momentum names look likely to beat
QQQ) but **does not clear significance**, and overlapping windows overstate n.
It would need the full `VALIDATION_METHODOLOGY.md` treatment (OOS split +
cutoff-date trajectory) before being believed.

## Verdict

* **Being "selective" about *which stock* — even as a meta-label on a working
  signal — does not add alpha. It subtracts it.** Across every method tried
  (5 model families now), the cross-sectional layer tilts defensive and lags
  QQQ. The alpha in this dataset *is* raw momentum; models dilute it.
* The only thread that even hints at value is **regime timing** (Result 2),
  and it is not statistically significant here. If anything is worth pursuing
  for durable, validated alpha it is **time-series risk management** (when to
  hold momentum vs rotate to the bond/cash leg), not a better stock selector —
  consistent with this repo's bottom line that there is no OOS cross-sectional
  selection alpha over QQQ in public price/volume data.
* Not promoted to a live factsheet.
