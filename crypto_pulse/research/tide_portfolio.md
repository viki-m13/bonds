# TIDE-anchored portfolio — diversification toward higher Sharpe (honest)

TIDE + orthogonal price/volume books, risk-parity (IS weights). HL era, OOS=last40%.

| book | Sharpe (HL) | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| TIDE | +2.01 | +2.04 | +1.98 | +32% | -8% |
| DIR-TREND | +1.40 | +1.70 | +0.94 | +21% | -12% |
| OBV-mom | +0.65 | +1.40 | -0.46 | +8% | -14% |
| VOL-shock | +0.39 | +0.78 | -0.15 | +5% | -20% |

## Correlation (HL era)

| | TIDE | DIR-TREND | OBV-mom | VOL-shock |
|---|---|---|---|---|
| TIDE | +1.00 | +0.40 | +0.49 | +0.46 |
| DIR-TREND | +0.40 | +1.00 | +0.32 | +0.32 |
| OBV-mom | +0.49 | +0.32 | +1.00 | +0.66 |
| VOL-shock | +0.46 | +0.32 | +0.66 | +1.00 |

## Combined books

| combine | Sharpe (HL) | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| Risk-parity portfolio (TIDE+DIR-TREND+OBV-mom+VOL-shock) | +1.37 | +2.19 | +0.25 | +20% | -13% |
| TIDE + DIR-TREND (50/50, least-corr) | +2.17 | +2.47 | +1.66 | +38% | -8% |

## Honest verdict

- TIDE alone OOS +1.98. Best diversifier vs TIDE: **DIR-TREND** (corr +0.40).
- Risk-parity portfolio OOS **+0.25** (-1.73 vs TIDE alone). The added books are too correlated / weaker to lift the combo.
- Sharpe 3 NOT reached. Honest combined ceiling ~2.0. The price/volume books co-move (all trend-driven), so diversification adds little on top of TIDE — confirming the ~2 wall. Genuine orthogonality needs non-price data (L4 order flow), still recording.
