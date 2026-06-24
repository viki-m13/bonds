# Improving TIDE round-6 — conviction/concentration/risk-balance (honest)

Strict bar: beat base OOS + pre-HL + all WF folds. base OOS +2.29, pre-HL +1.27.

| variant | HL | IS | OOS | dOOS | pre-HL | deflated P |
|---|---|---|---|---|---|---|
| base (improved TIDE) | +2.25 | +2.24 | +2.29 | +0.00 | +1.27 | 0.66 |
| +agree (horizon agreement) | +2.18 | +2.11 | +2.30 | +0.02 | +1.24 | 0.69 |
| +conc20 (top-20/side) | +2.23 | +2.08 | +2.49 | +0.20 | +1.21 | 0.75 |
| +erc (rank-risk balance) | +1.28 | +1.40 | +1.09 | -1.19 | +0.99 | 0.18 |
| +agree+conc20 | +1.95 | +1.77 | +2.26 | -0.03 | +1.24 | 0.67 |

## Verdict

- Robust survivors: NONE.
- **No round-6 lever robustly improves the book** (~2.29). After 32 honest attempts across 6 rounds, TIDE's three real refinements (5-horizon breakout, Parkinson vol) are the whole improvement; the book is definitively at its single-strategy ceiling.
- **Honest single-book ceiling: ~2.3 HL-era (1.55 over 12 years, positive every year).** No construction idea — standard or novel — honestly pushes one independent breakout book to 3.
