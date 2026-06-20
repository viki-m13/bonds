# Statistical arbitrage (Avellaneda-Lee) on US EQUITIES

430 US names (2009+), 2.0bps/side, dollar-neutral residual reversion, vol-targeted, IS=first60/OOS=last40. Same machinery that was taker-blocked in crypto — equities mean-revert.

## Parameter scan

| win | K | hold | Sharpe | IS | OOS | ann | maxDD | turn/day |
|---|---|---|---|---|---|---|---|---|
| 60 | 10 | 2 | +0.54 | +0.93 | +0.09 | +4.3% | -27.2% | 0.37 |
| 60 | 10 | 5 | +0.79 | +1.24 | +0.26 | +6.3% | -16.0% | 0.22 |
| 60 | 15 | 2 | +0.47 | +0.87 | +0.03 | +3.7% | -23.2% | 0.38 |
| 60 | 15 | 5 | +0.83 | +1.44 | +0.11 | +6.3% | -20.2% | 0.22 |

**Best (by min(IS,OOS)):** win=60, K=10, hold=5 -> Sharpe +0.79 (IS +1.24 / OOS +0.26).

## Verdict

- Equity residual reversion is weak/mixed (Sharpe +0.79, IS +1.24, OOS +0.26). This is the archetype working in its native market. It is structurally uncorrelated to a crypto book (different assets, market-neutral), so it is a genuine diversifying sleeve for a multi-asset stack — and HIP-3 equity perps could make it HL-tradeable.
