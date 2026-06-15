# CNN stock-selection signal

A 1-D convolutional network that reads each stock's recent price/volume
"shape" and predicts whether it will out-perform the cross-section over the
next quarter. Its per-date, per-ticker out-performance logits plug into the
existing biweekly-DCA harness exactly like any other `scores` matrix
(`protocol.evaluate_signal`), so it is graded on the identical
window grid and benchmarks (QQQ-DCA, SPY-DCA, random-pick control) as SUMMIT
and the momentum baselines.

Code: [`strategy_cnn.py`](strategy_cnn.py). Run `python strategy_cnn.py` to
train walk-forward, write `research/cnn_scores.parquet`, and print/save the
scorecards.

## Model

* **Input** — for each (ticker, date d) a `(3 x 126)` tensor: 126 points
  spaced every 2nd trading day, i.e. ~1 year of history through the close of
  d. Three channels, all trailing-normalised so nothing peeks past d:
  1. vol-normalised log return (own momentum, scale-free),
  2. vol-normalised excess return vs the equal-weight market (relative
     strength),
  3. trailing z-scored log volume (participation).
* **Net** — `Conv1d(3→16,5) · BN · ReLU · Conv1d(16→32,5) · BN · ReLU ·
  Conv1d(32→32,3) · ReLU · AdaptiveMaxPool · Dropout · FC(32→16) · FC(16→1)`.
  ~7k parameters. Global max-pool makes it shift-tolerant: it fires on a
  pattern wherever it occurs in the year, instead of hard-coding the 9-1 / 12-2
  formation offsets the momentum baselines use.
* **Label** — 1 if the name beats the eligible cross-sectional **median**
  forward 63-day (≈ one quarter) return, else 0. Trained with
  `BCEWithLogitsLoss`; the output logit is the score (rank descending, take
  top-k).

## Causality (RESEARCH_PROTOCOL.md, non-negotiable)

* Every feature value at date d uses closes/volumes **through the close of d**.
  All normalisation is trailing (rolling vol / rolling z-score) — no centred
  windows, no full-sample statistics.
* The 63-day forward label is used **only in training**.
* **Walk-forward fit.** At each annual refit date T the net trains *from
  scratch* on samples whose label window has fully closed on or before T, then
  scores strictly after T (until the next refit). No sample sees its own
  future; no later year leaks into an earlier prediction. First fit 2009-01,
  giving a clean out-of-sample score stream 2009 → present. Scores are emitted
  every 5 trading days and forward-filled (≤ 2 weeks) to cover signal dates —
  still strictly causal, since only *past* predictions are reused.

## Results

Active-period DCA grid (180 windows, quarterly starts 2010+, horizons 3/5/10y
+ to-end, biweekly, 5 bps/trade). Run `python cnn_report.py` to regenerate:

| signal | win vs QQQ | win vs SPY | median vs QQQ | worst vs QQQ |
|---|---|---|---|---|
| CNN k=2 | 17% | 57% | -14.7% | -46.2% |
| CNN k=3 | 13% | 52% | -16.0% | -44.8% |
| CNN k=10 | 7% | 29% | -18.9% | -50.1% |
| 9-1 momentum k=3 | **69%** | **89%** | **+10.6%** | -19.9% |
| random k=3 | 9% | 44% | -15.7% | -40.9% |

Out-of-sample cross-sectional information coefficient (rank corr of the CNN
logit vs forward return, averaged over scoring days):

| forward horizon | mean IC | annualised IR |
|---|---|---|
| 21d | +0.003 | +0.07 |
| 63d | +0.008 | +0.11 |
| 126d | +0.011 | +0.10 |

## Honest verdict & limitations

* **The CNN learns a real but small signal — it beats the random control and
  SPY-DCA more than half the time (k=2: 57% vs SPY) — but it does NOT beat
  QQQ-DCA, and it is well behind plain 9-1 momentum.** Reported as-is per the
  repo's validation-first protocol: it does **not** clear the
  `win_qqq ≥ 85%` bar and is **not** promoted to a live factsheet.
* **The pooling fix that mattered.** An earlier version used global *max*-pool
  only and scored ≈ 0 IC — max-pool is shift-tolerant but cannot *integrate*
  returns, so it literally could not represent cumulative momentum. Adding an
  *average*-pool branch (concat avg+max → MLP) restored a positive IC that
  **grows with horizon** (+0.003 at 21d → +0.011 at 126d), i.e. the net is now
  picking up genuine medium-term momentum/trend rather than 1-month noise. The
  lookback must also span ~1 year; a 64-day window can't see 6-12m momentum.
* **Why beating QQQ is brutal here.** QQQ over 2010-2026 was driven by a
  handful of mega-cap tech names; the random-pick control beats QQQ-DCA in only
  ~9% of windows, so *any* broad equal-dollar selection from the S&P 500 starts
  ~90% behind. The CNN's diffuse cross-sectional edge improves on random
  (17% vs 9% win-vs-QQQ; 57% vs 44% win-vs-SPY) but does not survive
  concentration into a top-k DCA book against a cap-weighted mega-cap index.
  Note the edge is in the *spread*, not the extreme tail: top-3 (k=2/3) beat
  top-30 (k=10), but all trail QQQ.
* OHLCV + volume only — no fundamentals, no macro regime gate. SUMMIT clears
  the bar precisely because it adds the two things this net lacks: a
  risk-on/off switch and a mega-cap (dollar-volume) size tilt. The natural next
  step is to use the CNN logit as one *factor* inside that machinery (replace /
  blend with the momentum term) rather than as a standalone top-k selector.
* Coverage starts 2009 (walk-forward needs ~4y of training history first), so
  grid windows beginning 2006-2008 hold cash and are excluded from the
  active-period read above.
