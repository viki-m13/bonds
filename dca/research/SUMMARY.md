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

## Second-wave findings

* **EDA (parabolic precursors)**: parabolic 6m runs come from high-vol,
  high-beta names, not compression patterns (all dead, |IC|≤0.01). The
  dominant fact is regime-conditionality: below SPY's 200dma, deep
  drawdown-from-ATH names hit P(+50% in 6m) = 17% (5.8x base) while
  momentum's IC flips negative; above it, trend strength + beta work and
  drawdown is worthless. Momentum sweet spot: 6-12m formation, skip the
  last weeks, payoff over 3-6 months; 2-week winners mean-revert.
* **Bear-regime agent**: bear sleeve = "quality rebounders" (long-term
  uptrend intact, 30-60% below ATH) beats holding cash and every defensive
  sleeve; recovery triggers and ALL sell rules degrade outcomes (HY-OAS
  panic exit takes worst window -24.5% → -62.6%). Never sell.
* **ML (LightGBM walk-forward)**: OOS IC ≈ 0.002 (t=0.5); learns defensive
  beta/vol tilts; decisively worse than one momentum column. Negative.
* **Chronos-bolt re-ranking**: loses to the matched momentum control at
  every k (-15 to -22pp win-rate). Negative.
* **The decisive ingredient — size**: adding a dollar-volume (mega-cap)
  rank to momentum closed the structural gap vs cap-weighted QQQ:
  60% → 85-94% window win-rate, robust across a broad weight plateau.

## Outcome

Final strategy **SUMMIT** (see `dca/SUMMIT.md`): regime-switched
mega-cap momentum (risk-on) / discounted-quality rebound (risk-off),
biweekly, k=2, never sell. Clean-panel results: **93% of 244 windows beat
QQQ-DCA (98% vs SPY), median excess +28.8%, worst -10.6%, all 8 regime
windows positive vs both, 100% wins at 10y+ horizons; full-period 20.0x
money multiple (24.7% IRR) vs QQQ 9.1x, SPY 4.7x.** Leakage audit clean,
offset/cost/parameter plateaus verified, NASDAQ-100 transfer positive,
random control cleared. The "every timeframe" goal is met at 5y+ horizons
outright; at 3y the win-rate is 84% vs QQQ (worst -10.6%, all residual
losses in QQQ's 2010-2013 AAPL-concentration era and 2006Q1).

## Decisions & non-starters

* LLM-based selection: not defensible with price/volume-only data (no
  point-in-time news/fundamentals in repo); skipped deliberately.
* Fundamentals (Piotroski, profitability): no PIT fundamental data
  available; out of scope, noted as future work.
