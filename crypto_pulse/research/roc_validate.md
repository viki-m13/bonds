# Validate the frozen iter-6 rule on independent data (honest)

One FROZEN rule: regime-tilted 20d breakout, x-sectional market-neutral, net 4.5bps+funding (funding only in HL era), vol-targeted. No new parameters. Tested unchanged across independent periods. The pre-HL period is fully out-of-sample.

| period | Sharpe | t-stat | CAGR | maxDD | N days |
|---|---|---|---|---|---|
| PRE-HL 2018..2023 (independent, spot proxy) | **+1.11** | +3.2 | +22% | -20% | 3114 |
| HL-IS (2023-05..cut) | **+2.04** | +2.7 | +34% | -7% | 647 |
| HL-OOS (cut..now) | **+1.98** | +2.2 | +29% | -8% | 432 |
| HL-full (2023-05..now) | **+2.01** | +3.5 | +32% | -8% | 1079 |
| FULL 2018..now | **+1.28** | +4.3 | +25% | -20% | 4193 |

## Honest verdict

- Pre-HL (independent) Sharpe **+1.11**, HL-OOS Sharpe **+1.98**.
- The frozen rule holds in BOTH the independent pre-HL period AND the HL-OOS window — that consistency across regimes the rule was never fitted on is real evidence, far less vulnerable to the 34-trial deflation than a single-period number. 
- **On Sharpe 3:** still not reached on any single period; the credible cross-period Sharpe is ~1.1-2.0. This is the honest standalone price book — strong, regime-robust, but a ~2 Sharpe, not 3. To go higher needs orthogonal data (L4), not more price-signal trials.
