# IGNITION on the standard DCA grid (vs QQQ / SPY / random)

Harness: dca/protocol.evaluate_signal — biweekly never-sell DCA, 244-window grid + 8 regimes, 5 bps/trade. `win_qqq`/`win_spy` = share of windows beating QQQ-/SPY-DCA; `med_vs_qqq` = median excess final multiple; `full_mult` = whole-period money multiple. The DCA objective is compounding, NOT parabolic capture — read alongside backtest.md.

| signal | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq | full_mult |
|---|---|---|---|---|---|
| PARABOLIC_ignition | 39% | 80% | -3.6% | -33.1% | 12.6x |
| PARABOLIC_ignition_beta | 52% | 83% | +1.0% | -32.1% | 14.5x |
| PARABOLIC_practitioner | 8% | 73% | -10.7% | -32.7% | 6.9x |
| PARABOLIC_pure_energy | 52% | 77% | +1.1% | -35.4% | 12.9x |
| PARABOLIC_mom91 | 43% | 79% | -2.4% | -25.2% | 11.2x |
| random-pick (k=10, control) | 7% | — | — | — | — |

Reading: IGNITION should clear the random-pick floor on `win_qqq` and beat SPY in most windows, while trailing QQQ (the high-variance tail objective costs compounding consistency). The practitioner-breakout and pure-energy variants are included for contrast.
