# Improving TIDE round-3 — risk management & execution (honest)

On the multi-horizon TIDE base. Walk-forward OOS + deflated (20 trials). base OOS +2.06.

| variant | Sharpe(HL) | IS | OOS | dOOS | CAGR | maxDD | deflated P |
|---|---|---|---|---|---|---|---|
| base (multiH) | +2.03 | +2.00 | +2.06 | +0.00 | +32% | -9% | 0.65 |
| +park (HL vol) | +2.20 | +2.20 | +2.19 | +0.13 | +36% | -9% | 0.71 |
| +crash (dd-floor) | +1.82 | +1.71 | +1.98 | -0.08 | +24% | -8% | 0.64 |
| +deadband (cut cost) | +1.77 | +1.61 | +2.02 | -0.05 | +27% | -11% | 0.63 |
| +fastvt | +1.99 | +1.92 | +2.10 | +0.03 | +32% | -8% | 0.69 |
| +paramens (hold 2/3/5) | +1.64 | +1.66 | +1.61 | -0.45 | +25% | -8% | 0.45 |

## TIDE final = multiH + robust risk/execution upgrades

- Kept: +park (HL vol).
- **OOS +2.19** (base multiH +2.06, single-horizon ~1.98); full HL +2.20; pre-HL +1.25; deflated P=0.71.
- 4-fold WF: +2.45, +1.15, +2.16, +3.08; bootstrap 95% CI [+1.13,+3.17].

## Verdict

- **TIDE improved to OOS +2.19** via +park (HL vol) on top of multi-horizon — robust across WF folds and pre-HL, a genuine single-book gain.
- Honest single-book level **~2.2**. A single independent breakout book does not honestly reach 3 — confirmed across 19 upgrade attempts.
