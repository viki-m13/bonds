# DCA Stock-Selection Research Summary (running document)

Mission: biweekly DCA into 1-5 stocks, hold long-term, beat QQQ-DCA and
SPY-DCA across every tested window. Strict anti-leakage, PIT universe,
delisting-aware, cost-aware.

## Infrastructure & data integrity work

* Point-in-time S&P 500 membership 1996-2026 (fja05680 dataset); 986
  constituents since 2004, 730 with usable Yahoo price history (panel
  5647 days x 720 tickers after cleaning). Coverage of members: 57% (2005)
  -> 99% (2026); missing names are mostly pre-2015 delistings.
* NASDAQ-100 PIT membership 2015-2026 (jmccarrell dataset) as a secondary
  transfer universe.
* **Data corruption found & fixed**: ~14 delisted tickers carried garbage
  Yahoo records (e.g. TNB: $0.68 -> $11,000 phantom spikes) and recycled
  tickers (CFC data years after Countrywide died). Fixes: first-segment
  truncation at >30-day gaps, neighborhood-median bad-tick repair,
  truncation at impossible (>75%/day) moves. **Every pre-fix backtest was
  inflated** (naive momentum win-rate vs QQQ: 73% dirty -> 59% clean).
* Engines: reference event-driven DCA engine + vectorized lot-based fast
  engine, agreeing to 4 decimals. Execution strictly at next open after
  signal close. 5 bps/trade base cost.
* Evaluation: 244-window grid (quarterly starts 2006-2023 x 3y/5y/10y/full
  horizons) + 8 regime windows, vs same-cadence QQQ/SPY DCA.
* Survivorship control: random-pick DCA from the same eligible universe
  ≈ SPY (median ratio 1.012, beats QQQ in only 8% of windows) ⇒ the
  universe itself contributes no edge vs SPY; signals must earn everything.

## Benchmark reality check

QQQ-DCA beats SPY-DCA in 97% of grid windows (median +16%). QQQ is the
binding benchmark. Even QQQ loses to SPY in 8/244 windows — a reminder
that "win literally every window" is a property not even major indexes
have against each other.

## Signal families explored (clean-panel numbers, biweekly, 5 bps, top-k)

| family | best clean config | win_qqq | win_spy | med_vs_qqq | worst | verdict |
|---|---|---|---|---|---|---|
| momentum/trend | 9-1 mom (189d skip 21) k=3 | 66% | 86% | +8% | -24% | core selector |
| residual momentum | resid vs SPY beta, 126d, k=2 | 65% | 86% | +8% | -39% | complementary |
| volume/accumulation | mom + distribution-veto k=1 | 65% | 82% | +10% | -33% | veto overlay only |
| vol-compression breakout | comp-gated mom k=1 | 64% | 82% | +6% | -34% | no edge; compression hurts |
| risk-adjusted mom (Sharpe/low-vol) | sortino252 k=3 | ~59% | 82% | +5% | -37% | low-vol tilts fatal vs QQQ |
| 52w-high / trend-quality | — | 3-23% | — | negative | — | dead vs QQQ |
| naive blends (rank-avg) | m91+resid k=2 | 66% | 83% | +8% | -35% | no break-through |

Failure anatomy is identical across all families: crash/transition regimes
(GFC 2007-09, recovery 2009-12 momentum crash, vol-2018, bear-2022).
Bull/sideways/AI-bull windows are consistently won. ⇒ The path to
every-window dominance is a regime-conditional architecture (bear behavior
+ recovery capture), not better bull-market stock selection.

## In progress

* EDA: parabolic-move precursors (deciles, ICs by regime).
* Bear-regime behavior: defensive sleeve, rebound capture, sell triggers.
* Walk-forward LightGBM cross-sectional ranker.
* Chronos-bolt re-ranking experiment (vs matched momentum control).
* Literature review brief.

## Decisions & non-starters

* LLM-based selection: not defensible with price/volume-only data (no
  point-in-time news/fundamentals in repo); skipped deliberately.
* Fundamentals (Piotroski, profitability): no PIT fundamental data
  available; out of scope, noted as future work.
