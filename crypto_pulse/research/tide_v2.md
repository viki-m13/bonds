# Improving TIDE itself — honest signal/construction upgrades

Each is a single change to the TIDE rule, walk-forward OOS + deflated. Keep only robust OOS gains over base. HL era, OOS=last40%. base OOS +1.98.

| variant | Sharpe(HL) | IS | OOS | dOOS vs base | deflated P |
|---|---|---|---|---|---|
| base TIDE | +2.01 | +2.04 | +1.98 | +0.00 | 0.77 |
| +resid (idiosyncratic) | +1.16 | +0.69 | +1.96 | -0.02 | 0.78 |
| +skip1 (no 1d reversal) | +0.98 | +0.75 | +1.38 | -0.60 | 0.53 |
| +multiH (10/20/40) | +2.03 | +2.00 | +2.06 | +0.08 | 0.79 |
| +conv (volume confirm) | +1.98 | +2.31 | +1.48 | -0.50 | 0.64 |
| +betaN (beta-neutral) | +1.88 | +2.07 | +1.57 | -0.41 | 0.65 |
| +regime2 (calm-vol gate) | +1.81 | +1.85 | +1.75 | -0.23 | 0.70 |

## TIDE v2 = base + robust upgrades

- Kept: NONE (no single change robustly helps).
- **TIDE v2 OOS Sharpe +1.98** (base +1.98, delta +0.00); full HL +2.01; deflated P=0.77.
- Sharpe 3 NOT reached.

## TIDE v2 robustness

- 4-fold walk-forward Sharpe: +2.31, +0.92, +2.41, +2.43 (all positive).
- Pre-HL independent Sharpe: +1.11.

## Verdict

- **No construction change robustly improves TIDE.** The base rule is already near the honest ceiling for this signal family; the upgrades help in-sample but not OOS.
- TIDE v2 stays a single, independent, market-neutral crypto-daily book. Honest level ~2.0.
