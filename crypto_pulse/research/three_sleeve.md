# Three-sleeve book: TREND + CARRY + ORDER-FLOW (HL, net)

Real funding + 4.5bps taker, IS=first60/OOS=last40. Order flow = OHLC proxy (close-location-value x volume), continuation, 5d hold.

## Standalone sleeves

| sleeve | Sharpe | IS | OOS |
|---|---|---|---|
| TREND | +0.35 | +0.55 | +0.07 |
| CARRY | +0.93 | +0.50 | +1.41 |
| ORDERFLOW | +0.44 | +0.70 | -0.02 |

Sleeve correlations: TREND-CARRY=+0.08, TREND-ORDERFLOW=+0.04, CARRY-ORDERFLOW=+0.04

## Blends — does order flow add value?

| book | Sharpe | IS | OOS | ann | maxDD | Calmar |
|---|---|---|---|---|---|---|
| trend+carry (2-sleeve base) | **+0.86** | +0.91 | +0.79 | +11.3% | -12.4% | 0.91 |
| trend+carry+OF equal (3-sleeve) | **+0.97** | +1.01 | +0.89 | +12.9% | -11.7% | 1.10 |
| trend+carry+OF risk-weighted | **+1.12** | +1.15 | +1.07 | +15.5% | -12.2% | 1.28 |

## Verdict

- The order-flow sleeve **ADDS value**: 3-sleeve Sharpe +1.12 vs 2-sleeve +0.86 (maxDD -12.2% vs -12.4%). OF correlation to trend = +0.04. This uses the OHLC PROXY; real signed taker volume (record_orderflow.py, forward) should be cleaner. Honest expectation: a modest lift, not a Sharpe jump — the deployable book is ~1.1-1.4.
