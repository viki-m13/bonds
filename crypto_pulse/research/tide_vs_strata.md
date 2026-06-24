# TIDE vs STRATA vs VOL — reconciliation for the VOL operator (honest)

All vol-targeted to 15% (the VOL+STRATA handoff convention), net. HL era (common dates).

## Standalone books (HL era)

| book | Sharpe | CAGR | maxDD |
|---|---|---|---|
| VOL | +1.67 | +33% | -14% |
| STRATA | +1.98 | +30% | -13% |
| TIDE | +2.35 | +52% | -12% |

## Correlation (HL era, daily)

| | VOL | STRATA | TIDE |
|---|---|---|---|
| VOL | +1.00 | +0.17 | +0.19 |
| STRATA | +0.17 | +1.00 | +0.44 |
| TIDE | +0.19 | +0.44 | +1.00 |

## Risk-parity combos (HL era)

| combo | Sharpe | CAGR | maxDD |
|---|---|---|---|
| VOL+STRATA | +2.18 | +37% | -8% |
| VOL+TIDE | +2.27 | +43% | -11% |
| STRATA+TIDE | +2.24 | +37% | -10% |
| VOL+STRATA+TIDE | +2.56 | +46% | -9% |

## What this means for the VOL operator

- **TIDE vs STRATA correlation = +0.44.** They are only partly correlated — TIDE can be a distinct sleeve.
- **TIDE vs VOL correlation = +0.19** — low, so TIDE diversifies VOL much like STRATA does.
- Best simple book here: **VOL+STRATA+TIDE** (Sharpe +2.56).
- Honest guidance: TIDE and STRATA are the same family (cross-sectional crypto). Run **one** of them as the market-neutral leg next to VOL — TIDE is the simpler, fully-documented, higher-capacity choice; STRATA is the 7-sleeve version. Do NOT double-count by running both at full size. TIDE+carry(phase-2) is the higher-Sharpe but uncertified variant.
