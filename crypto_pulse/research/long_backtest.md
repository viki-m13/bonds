# Longer backtest — price-sleeve stack over 2014-2026 (multi-regime)

TREND + BAB + SQUEEZE + ACCEL (no funding needed), equal-risk, vol-targeted to 12%, 4.5bps taker, 111-coin crypto universe with $3M liquidity filter. The funding sleeves (CARRY/FUNDFADE) can't extend before the HL era, so this is the price-book robustness test.

## Full sample 2015-05-20 -> 2026-04-24 (3993 days, 10.9y)

- **Full-sample Sharpe +1.09**, ann +16.8%, CAGR +16.9%, maxDD -24.1%.
- HL era (2023-05+): Sharpe +1.01; pre-HL (2014 -> 2023-05): Sharpe +1.12.

## By calendar year

| year | Sharpe | ann ret | maxDD | days |
|---|---|---|---|---|
| 2015 | -0.26 | -4.8% | -17.5% | 226 |
| 2016 | -0.32 | -5.5% | -19.8% | 366 |
| 2017 | +1.86 | +43.7% | -12.7% | 365 |
| 2018 | +1.17 | +15.2% | -9.6% | 365 |
| 2019 | +1.76 | +23.8% | -7.4% | 365 |
| 2020 | +1.99 | +28.2% | -7.6% | 366 |
| 2021 | +2.93 | +38.9% | -9.2% | 365 |
| 2022 | +0.04 | +0.5% | -11.6% | 365 |
| 2023 | +0.65 | +8.7% | -7.0% | 365 |
| 2024 | +1.01 | +12.8% | -9.8% | 366 |
| 2025 | +1.32 | +18.9% | -6.4% | 365 |
| 2026 | +0.23 | +4.0% | -10.1% | 114 |

## By regime (BTC vs its 200d MA)

- BTC-bull days: Sharpe +1.19 (63% of days)
- BTC-bear days: Sharpe +0.90 (37% of days)

## Verdict

- The price book is positive in **10/12** years and works in BOTH bull and bear regimes (it's market-neutral L/S). Full-sample Sharpe +1.09 over 11 years is the honest multi-regime number — the edge is structural, not a recent fluke. The funding sleeves add ~0.3-0.5 more in the HL era (grand stack ~1.5) but can't be checked pre-2023.
- A longer backtest does NOT raise the Sharpe — it CONFIRMS it (~1.0-1.4 for the price book across a decade). More history buys confidence, not a higher number; the ceiling is structural.
