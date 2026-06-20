# DCA Stock-Selection Research Protocol

Goal: a biweekly DCA strategy buying 1–5 stocks per period that beats DCA
into QQQ **and** SPY in (approximately) **every** window of the evaluation
grid — not just in aggregate.

> **Validation first:** before believing any scorecard from this harness, apply
> [`research/VALIDATION_METHODOLOGY.md`](research/VALIDATION_METHODOLOGY.md) —
> the survivorship-clean / out-of-sample / QQQ-benchmark / recency-cutoff tests
> that this in-repo harness does *not* fully enforce. Several strategies that
> scored ≥90% win-vs-QQQ here did not survive those tests.

## Infrastructure (all in `/home/user/bonds/dca`)

```python
import data, protocol
P = data.build_panel()           # dict: open/high/low/close/volume/member
                                 # (5647 days x 725 tickers, 2004→present,
                                 #  PIT S&P 500 membership mask in 'member')
scores = ...                     # DataFrame dates x tickers
card = protocol.evaluate_signal(scores, "my_signal", k=3)
```

* Prices are split+dividend adjusted. `member` is True only when the ticker
  was in the S&P 500 on that date (point-in-time, fja05680 dataset).
* FRED macro series in `data/fred/*.csv` (VIX = VIXCLS, HY OAS, yields...);
  benchmark ETFs (SPY, QQQ, sector ETFs, TLT, GLD...) via
  `data.load_benchmark("SPY")` or `data/etfs_extended/*.csv`.

## Causality contract (NON-NEGOTIABLE)

* `scores.loc[d]` may use information **through the close of day d only**.
  Execution is at the **next day's open** (the engine handles this).
* No centered rolling windows, no `shift(-1)`, no use of full-sample
  statistics (means/stds/quantiles fitted on all years), no resampling that
  peeks past d. Cross-sectional ranks/z-scores within row d are fine.
* If a model needs fitting (ML), fit walk-forward: refit at most monthly,
  using only data ≤ fit date; predict strictly after.

## Evaluation

`protocol.evaluate_signal(scores, name, k=3, every=10, offset=0,
cost_bps=5, sell=None)`:

* Grid: quarterly starts 2006→2023 × horizons {3y, 5y, 10y, to-end} = 244
  windows + 8 named regime windows (GFC, COVID, 2022 bear, ...).
* For each window, runs the biweekly DCA (capital arrives every 10 trading
  days, invested at next open across top-k, 5 bps per trade) and compares
  final money-multiple vs same-cadence QQQ and SPY DCA.
* Scorecard JSON lands in `dca/research/scorecards/<name>.json`.
* Headline numbers: `win_qqq`, `win_spy` (share of grid windows won),
  `med_vs_qqq`, `worst_vs_qqq`, plus per-regime results.
* Optional `sell` DataFrame (bool): True at row d ⇒ liquidate that holding
  at next open; proceeds recycle into the next biweekly buy.

## Baselines to beat (top-3, biweekly, 5 bps)

| signal | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq |
|---|---|---|---|---|
| naive 6m momentum | 59% | 86% | +6% | -26% |
| 12-2 momentum k=2 | 61% | 87% | +5% | -23% |
| 9-1 momentum k=3 (best pure mom) | 66% | 86% | +8% | -24% |
| random picks | beats SPY 55% / QQQ 8% of windows | | | |

NOTE (2026-06-12): the panel was rebuilt after discovering corrupted Yahoo
records on ~14 delisted tickers (garbage 1000x price spikes). Any scorecard
produced before the rebuild is inflated — re-run builders on the clean panel
before trusting numbers.

A candidate is interesting if `win_qqq ≥ 85%` with `med_vs_qqq` clearly
positive; it must also beat the random control (survivorship check).

## Known data caveats

* Universe covers ~57% of 2005 members rising to ~99% today (delisted names
  Yahoo lacks). Hence: always compare against the random-pick control, which
  carries the same bias.
* Tickers truncated at first >30-day gap (ticker-recycling guard).
* No fundamentals; OHLCV + FRED macro only.
