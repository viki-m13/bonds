# TIMING — regime risk-management overlay (momentum ⇄ Treasuries)

Code: [`../strategy_timing.py`](../strategy_timing.py) (signal + rotation
backtest), [`../validate_timing.py`](../validate_timing.py) (validation suite).
Risk-ON probability cached to `timing_riskon.parquet`.

## The idea

The selection models all subtract from momentum, so this *keeps* the 12-1
momentum book and only adds a **time-series** decision: each biweekly signal
date, hold momentum (risk-ON) or rotate the whole stock sleeve into 7-10y
Treasuries / IEF (risk-OFF). A LightGBM classifier on trailing regime features
(SPY trend, breadth, VIX, term spreads, SPY vol/drawdown) predicts risk-ON;
its label (train only) is "did the momentum book out-return IEF over the next
63d", refit walk-forward. Goal: better **risk-adjusted** return (Sortino /
drawdown), not beating QQQ on raw return.

## Result — fails comprehensively, and fails the OOS / recency tests

Active period 2010+, k=3, 5 bps:

| strategy | mult | annRet | Sharpe | Sortino | maxDD |
|---|---|---|---|---|---|
| timing (rotate) | 2.10 | +12.4% | 0.43 | 0.53 | **−57%** |
| always-momentum | 11.53 | +23.5% | 0.87 | 1.15 | −47% |
| QQQ-DCA | 6.40 | +19.7% | 0.96 | 1.23 | −35% |
| always-bonds (IEF) | 1.11 | +2.8% | 0.42 | 0.65 | −24% |

The overlay made **both** return *and* drawdown **worse** than simply holding
the momentum book — the worst of both worlds. It is not a risk/return
trade-off; it is strictly dominated.

**Why (the validation suite pinpoints it):**

* **OOS collapse.** 2010-2017 timing Sortino 0.95 (~ok); **2018-2026 Sortino
  0.45, maxDD −56%**, mult 1.62 vs always-momentum 3.46. The regime model is an
  in-sample artifact that does not generalize.
* **Cutoff-date trajectory (ratio vs QQQ), start 2010:** 0.92 → 0.96 → 1.02
  (2019) → **0.48 → 0.36 → 0.35 → 0.33** (end). It kept pace through 2019, then
  fell off a cliff — it sat in bonds through the 2020-2025 bull.
* **The defensive leg failed too.** Gain importance is dominated by the term
  spread (`term2y` 0.22, `term3m` 0.21). That yield-curve regime *inverted* its
  meaning after 2018, and in 2022 Treasuries fell *with* stocks — so the "hedge"
  lost money exactly when it was supposed to protect, hence the −57% maxDD.
* **Rolling beat-rate:** timing beats QQQ in **26%** of 3y windows / **15%** of
  5y; beats always-momentum in **19%** / **11%**. Market timing almost never
  wins over multi-year holds — the textbook result, reproduced cleanly here.

## Verdict

* **Regime timing does not add alpha — risk-adjusted or otherwise.** It gave up
  the bull market and the bond hedge failed in the one regime (2022) it was
  meant for. It fails precisely the OOS-split and cutoff-trajectory tests in
  `VALIDATION_METHODOLOGY.md` that killed SUMMIT.
* Combined with the selection results (CNN, PatchTST, LightGBM ranker, Chronos,
  meta-labeling), the program has now tested **both** axes a model could add
  value on — *which* to hold and *when* to hold — and neither survives
  out-of-sample over a momentum-heavy benchmark with public price/volume+macro
  data. The honest bottom line stands: **the alpha is buy-and-hold momentum /
  QQQ itself; every model layer we add subtracts from it.**
* Not promoted to a live factsheet.
