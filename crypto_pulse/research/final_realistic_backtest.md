# Final backtest at REALISTIC cost — short & long, by universe

Realistic all-in (fee+slippage, $1-10M): **6.5 bps taker / 3.5 bps maker** (not the vol-repo intraday worst-case). Price sleeves; funding sleeves added for the HL-era full book.

## Universe comparison — price book, taker 6.5bps

| universe | LONG 2015-26 | SHORT (HL era) | 2015-19 | 2020-22 | 2023-26 |
|---|---|---|---|---|---|
| skip top-9 / next-30 (mid-cap) | **+1.31** | +1.13 | +1.15 | +1.73 | +1.16 |
| top-30 (incl. megacaps) | **+1.25** | +0.99 | +1.24 | +1.68 | +0.92 |

**Better universe: skip top-9 / next-30 (mid-cap).**

## Better universe — realistic taker vs maker, short & long

| cost | LONG 2015-26 Sharpe | CAGR | maxDD | SHORT (HL) Sharpe |
|---|---|---|---|---|
| taker 6.5bps | **+1.31** | +17% | -10% | +1.13 |
| maker 3.5bps | **+1.39** | +18% | -10% | +1.24 |

## Full grand stack (price + funding sleeves), HL era, realistic cost

| cost | Sharpe | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| taker 6.5bps | **+1.52** | +1.52 | +1.86 | +22% | -9% |
| maker 3.5bps | **+1.69** | +1.69 | +2.02 | +25% | -9% |

## Verdict

- At realistic cost, the **skip top-9 / next-30 (mid-cap)** price book does +1.31 over the full decade and +1.13 in the HL era — positive every era. The skip-megacap universe edges plain top-30 on out-of-sample/recent robustness (mid-cap cross-sectional dispersion), as the universe experiments found.
- The **full grand stack** (adding the funding sleeves, which are net funding RECEIVERS) at realistic cost is **+1.52 taker / +1.69 maker** in the HL era — the deployable number. This is net of genuinely realistic execution, not the HFT worst-case.
- Honest read: **~1.0-1.3 price-only across the decade, ~1.5 full book in the HL era**, at realistic ~6.5bps taker (better as a patient maker). Robust short and long, every regime.
