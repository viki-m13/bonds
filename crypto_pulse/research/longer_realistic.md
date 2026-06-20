# Longer backtest under realistic execution (price sleeves, 2015-2026)

PIT top-30 by 30d dollar volume (monthly). Fee+slippage charged on turnover. Price book only (no funding) — conservative; the HL-era book with funding sleeves is a net funding RECEIVER. Full decade, multiple regimes.

Combined daily turnover ~0.77x/day (~281x/yr).

## Slippage sweep over the full 2015-2026 sample

| total cost/side | Sharpe | CAGR | maxDD |
|---|---|---|---|
| 4.5 bps | **+1.31** | +21% | -24% |
| 6.5 bps | **+1.25** | +20% | -25% |
| 8.5 bps | **+1.19** | +19% | -26% |
| 10.5 bps | **+1.13** | +17% | -27% |
| 14.5 bps | **+1.01** | +15% | -30% |
| 20.5 bps | **+0.84** | +12% | -33% |

## By era (at base 4.5bps and at 10.5bps = their break-even)

| era | Sharpe @4.5 | Sharpe @10.5 | CAGR @4.5 |
|---|---|---|---|
| 2015-2019 (early/bear) | +1.16 | +1.03 | +20% |
| 2020-2022 (mania+bear) | +1.74 | +1.56 | +26% |
| 2023-2026 (HL era) | +1.01 | +0.73 | +13% |

## Verdict

- Over the **full decade**, the price book holds up under realistic slippage: Sharpe +1.31 at 4.5 bps, **+1.13 at the vol repo's 10.5 bps break-even**, still positive at 20.5 bps (+0.84). The graceful degradation is consistent across the 2015-2026 sample, not just the recent window.
- It is positive in every era at both cost levels — the edge-per-trade moat holds through the 2018 bear, 2020 COVID, 2021 mania, and 2022 deleveraging. This is the longer-horizon confirmation that the result survives the vol repo's hardest live lesson (slippage) structurally, not by luck of regime.
- Add back the HL-era funding sleeves (net funding RECEIVER) and the book is ~1.5 and gets BETTER under funding stress — so the full deployable book is even more execution-robust than this price-only decade lower bound.
