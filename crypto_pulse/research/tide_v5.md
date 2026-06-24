# Improving TIDE round-4 — universe/robustness/funding (honest)

On improved base (multiH+Parkinson). Robust bar: beat base OOS AND pre-HL AND all WF folds. base OOS +2.19, pre-HL +1.25.

| variant | HL | IS | OOS | dOOS | pre-HL | deflated P |
|---|---|---|---|---|---|---|
| base (improved TIDE) | +2.20 | +2.20 | +2.19 | +0.00 | +1.25 | 0.68 |
| +5horizons | +2.23 | +2.20 | +2.29 | +0.09 | +1.35 | 0.70 |
| +winsor | +2.19 | +2.20 | +2.17 | -0.03 | +1.25 | 0.67 |
| +topN20 | +1.40 | +1.02 | +1.99 | -0.20 | +1.03 | 0.59 |
| +fundaware | +2.18 | +2.15 | +2.24 | +0.04 | +1.25 | 0.70 |

## Verdict

- Robust survivors: +5horizons.
- **TIDE improved further: OOS +2.19 -> +2.29** (pre-HL +1.35, WF +2.3, +1.3, +2.2, +3.3).
- Honest single-book level **~2.3**. Confirmed across 23 upgrade attempts: a single independent breakout book does not honestly reach 3.
