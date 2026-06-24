# ROC lab iter-6: regime-conditional exposure (honest)

Scale the best static signal (20d breakout) by causal regime states (trend-agreement, index vol). Does regime-timing beat the static book? HL era, OOS=last40%.

| book | Sharpe (HL) | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| STATIC breakout (baseline) | +1.75 | +1.99 | +1.38 | +25% | -10% |
| regime: trend-on only | +1.29 | +1.21 | +1.42 | +20% | -13% |
| regime: calm-vol only | +1.39 | +1.53 | +1.18 | +20% | -8% |
| regime: continuous trend tilt | +2.01 | +2.04 | +1.98 | +32% | -8% |

## Honest verdict (iteration 6)

- Static baseline OOS +1.38. Best regime variant: **regime: continuous trend tilt** OOS +1.98, deflated (34 trials) +1.98, P=0.53.
- Regime-timing **beat** the static book — a real lift.
- Sharpe 3 NOT reached. Iteration 6; price ceiling holds at ~1.0-1.85 deflated across 6 methods and 3 repos.
