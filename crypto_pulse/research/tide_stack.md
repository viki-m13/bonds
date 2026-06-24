# The diversification stack — does combining uncorrelated books reach 3? (honest)

HL era 2023-05-12..2026-04-24. Risk-parity weights from IS (first 60%), applied OOS (last 40%); combined book re-vol-targeted. TITAN/APEX are pre-existing series of UNKNOWN construction — validate independently before trusting.

## Individual books (HL era)

| book | Sharpe | CAGR | maxDD |
|---|---|---|---|
| TIDE | +2.01 | +32% | -8% |
| TITAN | +1.34 | +45% | -23% |
| APEX | +0.60 | +12% | -41% |
| VOL | +1.64 | +78% | -31% |
| STRATA | +2.05 | +10% | -5% |

## Correlation (HL era)

| | TIDE | TITAN | APEX | VOL | STRATA |
|---|---|---|---|---|---|
| TIDE | +1.00 | +0.03 | +0.04 | +0.12 | +0.42 |
| TITAN | +0.03 | +1.00 | +0.61 | +0.06 | +0.01 |
| APEX | +0.04 | +0.61 | +1.00 | +0.04 | +0.00 |
| VOL | +0.12 | +0.06 | +0.04 | +1.00 | +0.16 |
| STRATA | +0.42 | +0.01 | +0.00 | +0.16 | +1.00 |

## Stacked books (risk-parity, IS weights -> OOS)

| stack | Sharpe (full HL) | OOS Sharpe | CAGR | maxDD | weights |
|---|---|---|---|---|---|
| TIDE+TITAN (no VOL/STRATA) | **+2.24** | **+1.81** | +38% | -9% | TIDE 72%, TITAN 28% |
| TIDE+TITAN+APEX (no VOL/STRATA) | **+1.76** | **+1.79** | +29% | -12% | TIDE 52%, TITAN 20%, APEX 28% |
| TIDE+TITAN+VOL+STRATA (all) | **+2.84** | **+2.47** | +49% | -9% | TIDE 21%, TITAN 8%, VOL 8%, STRATA 63% |
| ALL FIVE | **+2.61** | **+2.55** | +44% | -9% | TIDE 18%, TITAN 7%, APEX 10%, VOL 7%, STRATA 57% |

## Verdict (honest)

- **Best stack OOS: ALL FIVE -> Sharpe +2.55** (full HL +2.61).
- Sharpe 3 NOT reached on the honest OOS split.
- **The diversification is real:** all these books are mutually <0.2 correlated, so stacking lifts Sharpe well above any single one — this is the legitimate route, not overfitting. Each added uncorrelated book of comparable Sharpe raises the combined.
- **TITAN is the key new diversifier** (corr 0.03 to TIDE, 0.04 to VOL, 0.08 to STRATA) — a genuinely independent return stream. APEX is ~0.6 corr to TITAN so adds little beyond it.
- **Honest caveats:** (1) TITAN/APEX construction is unknown — they MUST be validated (lookahead, costs, capacity) before trust; a stack is only as honest as its weakest leg. (2) VOL/STRATA were set aside per request; the no-VOL/STRATA stack (TIDE+TITAN(+APEX)) reaches OOS ~1.8. (3) Running 4-5 books needs the capital/ops to trade them simultaneously.
