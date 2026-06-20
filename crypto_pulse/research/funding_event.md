# Event-driven funding-extreme fade (time-series contrarian) on HL

Net of 4.5bps taker + realized funding, vol-targeted, IS=first60/OOS=last40. Fade crowded positioning in the tails; hold for liquidation-cascade reversion.

## Parameter scan (funding z-window / trigger z / hold days)

| zwin | trig | hold | Sharpe | IS | OOS | ann | maxDD | active% |
|---|---|---|---|---|---|---|---|---|
| 30 | 1.0 | 2 | +0.10 | -0.31 | +0.87 | +1.4% | -27.9% | 100% |
| 30 | 1.0 | 3 | +0.10 | -0.32 | +0.90 | +1.4% | -27.7% | 100% |
| 30 | 1.0 | 5 | +0.22 | -0.28 | +1.17 | +3.2% | -26.5% | 100% |
| 30 | 1.5 | 2 | +0.23 | -0.16 | +0.98 | +3.3% | -26.8% | 99% |
| 30 | 1.5 | 3 | -0.04 | -0.42 | +0.69 | -0.5% | -28.1% | 100% |
| 30 | 1.5 | 5 | +0.11 | -0.35 | +0.96 | +1.6% | -26.2% | 100% |
| 30 | 2.0 | 2 | +0.56 | -0.15 | +1.89 | +8.1% | -28.9% | 95% |
| 30 | 2.0 | 3 | +0.50 | -0.02 | +1.47 | +7.3% | -22.2% | 98% |
| 30 | 2.0 | 5 | +0.49 | +0.08 | +1.26 | +7.3% | -21.7% | 100% |
| 60 | 1.0 | 2 | +0.25 | -0.20 | +1.10 | +3.6% | -27.8% | 100% |
| 60 | 1.0 | 3 | +0.13 | -0.35 | +1.04 | +1.8% | -26.2% | 100% |
| 60 | 1.0 | 5 | +0.13 | -0.28 | +0.91 | +1.9% | -27.8% | 100% |
| 60 | 1.5 | 2 | +0.57 | -0.02 | +1.69 | +8.1% | -24.0% | 99% |
| 60 | 1.5 | 3 | +0.22 | -0.20 | +1.01 | +3.2% | -25.8% | 100% |
| 60 | 1.5 | 5 | +0.09 | -0.33 | +0.86 | +1.2% | -29.5% | 100% |
| 60 | 2.0 | 2 | -0.15 | -0.77 | +1.04 | -2.1% | -34.5% | 93% |
| 60 | 2.0 | 3 | -0.15 | -0.60 | +0.70 | -2.1% | -30.2% | 96% |
| 60 | 2.0 | 5 | -0.19 | -0.74 | +0.83 | -2.7% | -33.7% | 99% |

**Best (by min(IS,OOS)):** zwin=30, trig=2.0, hold=5 -> Sharpe +0.49 (IS +0.08 / OOS +1.26).

## Verdict

- Funding-extreme fade is not robust net of cost (Sharpe +0.49, IS +0.08, OOS +1.26); correlation to the directional stack -0.29. As an event sleeve its value is timing diversification (it fires in stress, when trend whipsaws), even at a modest standalone Sharpe.
