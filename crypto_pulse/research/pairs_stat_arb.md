# Cointegration pairs stat-arb on HL crypto — does it add a sleeve?

Walk-forward cointegration pairs: trailing 120d selection, re-select every 21d, hold up to 20 pairs (half-life 2-25d) from the top-24 liquid coins. Mean-revert spread z (|z|<=1.5), dollar-neutral legs, net 4.5bps/leg + funding. HL era, IS/OOS. Selection & hedge ratios use only past data.

Avg pairs held per rebalance: 20.0 (over 51 rebalances).

| book | Sharpe | IS | OOS | maxDD | corr to STRATA |
|---|---|---|---|---|---|
| PAIRS stat-arb | **-0.58** | -0.87 | -0.03 | -41% | -0.24 |
| STRATA (7-sleeve) | **+1.58** | +1.37 | +1.85 | -11% | — |
| STRATA + PAIRS | **+1.59** | +1.36 | +1.88 | -11% | — |

## Verdict

- PAIRS standalone OOS -0.03, corr to STRATA -0.24, optimizer weight in the blend 0%. Adding it takes STRATA OOS +1.85 -> **+1.88** (+0.03). The optimizer assigns PAIRS only 0% weight, so any OOS wobble is covariance-reshuffle noise, not a real contribution. Crypto cointegration is unstable: hedge ratios drift and spreads break OOS, so the standalone book can't carry a positive IS Sharpe and gets ~zero allocation even though its return stream is genuinely anti-correlated (-0.24) to STRATA.

