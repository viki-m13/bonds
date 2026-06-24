# ROC lab iter-5: cross-validated robust ensemble (honest)

A signal is ROBUST only if Sharpe>0.3 in ALL 3 IS sub-folds (not just IS overall). Survivors combined equal-risk. HL era, OOS=last40%.

| signal | fold1 | fold2 | fold3 | IS | OOS | robust? |
|---|---|---|---|---|---|---|
| multiROC | +1.02 | +0.04 | +1.32 | +0.82 | +0.39 | no |
| riskadj | +0.73 | -0.50 | +1.07 | +0.46 | +1.02 | no |
| Donchian | +1.06 | +1.60 | -0.69 | +0.63 | +0.21 | no |
| MACD | +1.43 | +0.52 | +2.12 | +1.36 | +0.36 | YES |
| AwesomeOsc | -0.00 | -0.09 | +1.06 | +0.31 | -0.05 | no |
| breakout20 | +2.78 | +1.55 | +1.56 | +1.99 | +1.38 | YES |
| TStrend(dir) | +2.25 | +0.65 | +2.20 | +1.70 | +0.94 | YES |

## Robust ensemble: MACD, breakout20, TStrend(dir)

- Sharpe (HL) +1.68, IS +2.32, **OOS +0.84**, CAGR +26%, maxDD -11%.
- Deflated OOS (30 trials): **+0.84**, P(SR>0)=0.13 (does NOT clear 95%).
- Sharpe 3 NOT reached.

## Honest verdict (iteration 5)

- Cross-validated selection is the legitimate fix for IS-overfitting, and it still lands under Sharpe 3. The instability is intrinsic to price-based crypto signals, not a selection mistake.
- After 5 honest iterations across 3 strategy repos, the deflated price ceiling is ~1.0-1.85 OOS. Sharpe 3 is not honestly reachable from price data alone.
