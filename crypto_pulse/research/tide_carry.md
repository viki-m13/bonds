# Funding carry as the orthogonal leg for improved TIDE (honest)

## Standalone CARRY books (funding lookback)

| lookback | HL | IS | OOS | pre-HL | corr→TIDE(HL) | CAGR | maxDD |
|---|---|---|---|---|---|---|---|
| 3d | +1.23 | +0.51 | +2.26 | +nan | +0.22 | +19% | -24% |
| 7d | +1.46 | +0.60 | +2.63 | +nan | +0.23 | +23% | -22% |
| 14d | +1.55 | +0.94 | +2.41 | +nan | +0.24 | +24% | -18% |
| 30d | +1.46 | +1.10 | +2.00 | +nan | +0.24 | +22% | -13% |

## TIDE + CARRY(14d) risk-parity

- TIDE alone: HL +2.23, OOS +2.29, pre-HL +1.35.
- CARRY(14d): HL +1.55, OOS +2.41, pre-HL +nan, **corr to TIDE +0.24**.
- **Combo: HL +2.36, OOS +2.87 (+0.59), pre-HL +1.21 (-0.13), CAGR +41%, maxDD -11%.**

## Verdict — PROMISING but NOT independently confirmable

- **Carry is the most orthogonal leg ever found here: corr +0.24 to TIDE** (price/volume legs were +0.40–0.49). TIDE+CARRY lifts **HL-era OOS +2.29 -> +2.87** and CAGR +36% -> +41%, WF folds +2.9, +0.3, +3.2, +3.0 (all positive but one fold only +0.3); maxDD slightly worse -9% -> -11% (carry's tail). Genuinely encouraging.
- **BUT three honest caveats keep it UNCERTIFIED, unlike the 5-horizon/Parkinson core:**
  1. **No independent validation.** Funding data starts ~2023, so carry has NO pre-HL history (pre-HL = NaN). The whole carry edge lives inside the same HL window I tuned in — I cannot confirm it on a held-out regime the way I did for the core refinements.
  2. **Edge concentrated in recent data:** carry IS Sharpe ~0.6–1.1 but OOS ~2.0–2.6 — the premium is far stronger in the last ~18 months; could be a 2024–25 funding regime, not a law.
  3. **Carry-crash tail risk:** standalone carry maxDD −13% to −24% (shorting crowded-long coins gets squeezed in rallies) — a left-tail the Sharpe understates.
- **Honest call:** carry is a *real lead* for a second leg (low corr, strong HL-era OOS), but it is NOT yet a validated upgrade. **Deploy TIDE alone (~2.3) as the certified book**; paper-trade carry and collect more out-of-sample funding history before sizing it live.
- This was the last structurally-orthogonal price/funding source short of the L4 order-flow book (still recording) — genuinely non-price information, the real road past ~2.3.
