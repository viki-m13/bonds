# XGBoost ensemble sleeve — does it improve STRATA OOS?

13 XGBoost models (objectives x depth x seed), walk-forward (retrain 42d, embargo 6d), rank-ensembled, predicting cross-sectional fwd-5d return. Net 4.5bps + funding. HL era.

| book | Sharpe | IS | OOS | maxDD | corr to STRATA |
|---|---|---|---|---|---|
| XGB ensemble sleeve | **-0.10** | +0.10 | -0.44 | -27% | -0.09 |
| STRATA (7-sleeve) | **+1.58** | +1.37 | +1.85 | -11% | — |
| STRATA + XGB | **+1.58** | +1.37 | +1.84 | -11% | — |

## Verdict

- XGB ensemble sleeve OOS -0.44, corr to STRATA -0.09. Adding it takes STRATA OOS +1.85 -> **+1.84** (-0.01). It does NOT robustly improve STRATA OOS — the ML ensemble extracts no edge beyond the factor sleeves after honest walk-forward + cost (crypto cross-sectional returns are near-unpredictable; the features ARE the sleeves).

