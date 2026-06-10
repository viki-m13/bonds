# winrate30 — Walk-Forward Validation Report

Generated: 2026-06-10  |  Data: 2000-01-03 .. 2026-06-10  |  Universe: 339 stocks  |  Horizon: 21 trading days (~30 calendar days)

## Headline out-of-sample result

- **Signals (non-overlapping, fully out-of-sample): 268**
- **Positive after 21 trading days: 190  ->  hit rate 70.9%**
- **95% Wilson lower bound: 65.2%**
- Average forward return per signal: 5.4%
- Median: 4.3%  |  5th percentile: -6.5%  |  worst: -21.8%
- Signal frequency: 3.4/month on average; signals occurred in 9 of 79 months

## Basket-level result (recommended way to use the tool)

Buying every signal of a calendar month as one equal-weight basket and holding each position ~30 days diversifies away single-stock misses:

- **9 of 9 signal-months had a positive basket return = 100.0%** (Wilson 95% lower bound 70.1%)
- Average basket month return: 7.6%; worst basket month: 1.9%

## Per-year walk-forward results

Rules are re-selected each year using only prior data, then tested on the following year.

| Test year | Signals | Hit rate | Avg 30d ret | Worst signal | Baseline (any stock) |
|---|---|---|---|---|---|
| 2016 | 0 | n/a | n/a | n/a | 62.5% |
| 2017 | 0 | n/a | n/a | n/a | 66.1% |
| 2018 | 4 | 100.0% | 12.9% | 5.1% | 50.5% |
| 2019 | 0 | n/a | n/a | n/a | 65.5% |
| 2020 | 199 | 69.3% | 5.4% | -17.2% | 59.8% |
| 2021 | 61 | 72.1% | 4.4% | -21.8% | 60.8% |
| 2022 | 0 | n/a | n/a | n/a | 48.9% |
| 2023 | 0 | n/a | n/a | n/a | 54.8% |
| 2024 | 4 | 100.0% | 14.8% | 1.0% | 57.3% |
| 2025 | 0 | n/a | n/a | n/a | 55.8% |
| 2026 | 0 | n/a | n/a | n/a | 51.2% |

## Worst months (cross-sectional risk)

Signals cluster in time and stocks move together, so the binomial confidence interval understates tail risk. The worst signal-months:

| Month | Signals | Hit rate | Avg ret |
|---|---|---|---|
| 2020-09 | 72 | 61.1% | 1.9% |
| 2021-01 | 45 | 62.2% | 3.6% |
| 2020-11 | 3 | 66.7% | 3.1% |
| 2020-10 | 89 | 68.5% | 7.3% |
| 2020-07 | 29 | 86.2% | 7.3% |
| 2020-06 | 6 | 100.0% | 11.0% |
| 2018-02 | 4 | 100.0% | 12.9% |
| 2021-12 | 16 | 100.0% | 6.8% |
| 2024-08 | 4 | 100.0% | 14.8% |

## Production rules (selected on full history)

- **spy_above_200 + vix_gt30 + rsi_lt35 + vol_low** — n=187 non-overlapping signals, hit rate 82.4%, Wilson LB 76.3%
  - i.e. buy when: S&P 500 above its 200-day average (market uptrend); VIX above 30 (panic); RSI(14) below 35 (oversold); volatility in the bottom third of its own 1-year range
- **spy_above_200 + vix_gt30 + golden + dd_-10_-25 + vol_low + mom_pos** — n=160 non-overlapping signals, hit rate 80.0%, Wilson LB 73.1%
  - i.e. buy when: S&P 500 above its 200-day average (market uptrend); VIX above 30 (panic); stock above 200-day avg and 50-day avg above 200-day avg; 10-25% below its 52-week high; volatility in the bottom third of its own 1-year range; positive 12-month momentum (excluding last month)

## Methodology & caveats

- A 'hit' = adjusted close is higher 21 trading days after the signal.
- Signals are deduplicated per stock (a stock cannot re-signal within 21 trading days), so outcomes do not double count overlapping windows.
- Rule selection per fold never sees the test year (training data even ends 21 trading days before it, so no forward window leaks).
- **Survivorship bias**: the universe is today's large caps; failed companies are absent. Large-cap restriction limits but does not eliminate this; true forward-looking hit rates are likely somewhat lower than backtested ones.
- **Regime risk**: most of the validation window is a structural bull market. In a 2008-style year the realized hit rate would be far below the average — see the worst-months table.
- Returns ignore transaction costs, slippage and taxes (small at a 30-day horizon for liquid large caps, but not zero).
- This is research tooling, not investment advice.