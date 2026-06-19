# BAB / low-beta sleeve — does it add to the 3-sleeve book?

HL era, real funding + 4.5bps taker, IS=first60/OOS=last40, weekly hold. BAB = long-low-beta/short-high-beta (BTC market, 90d beta); low-vol = long-low/short-high 90d idio vol.

## Standalone (net, vol-targeted)

| sleeve | Sharpe | IS | OOS |
|---|---|---|---|
| BAB (low-beta) | +0.69 | +0.58 | +0.86 |
| low-vol | +0.17 | +0.02 | +0.40 |
| crypto 3-sleeve (ref) | +1.00 | +0.81 | +1.26 |

Correlation to 3-sleeve book: BAB +0.12, low-vol -0.04

## Combination: 3-sleeve + BAB (Sharpe-optimal IS weights)

IS weights: crypto3 51%, BAB 49%

| book | Sharpe | IS | OOS | ann | maxDD |
|---|---|---|---|---|---|
| crypto 3-sleeve (base) | **+1.00** | +0.81 | +1.26 | +14.6% | -13.9% |
| 3-sleeve + BAB | **+1.25** | +1.07 | +1.54 | +16.2% | -11.2% |

## Verdict

- BAB **ADDS value**: combined Sharpe +1.25 vs 3-sleeve +1.00. BAB standalone Sharpe +0.69, correlation to the book +0.12. The research flagged low-beta as the one cross-sectional crypto factor that survives taker costs (slow signal, low turnover); this is the honest in/out-of-sample test of that claim on our universe.
