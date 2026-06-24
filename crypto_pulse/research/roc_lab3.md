# ROC lab iter-4: vol-managed time-series trend (CTA-style)

Per-coin TS-momentum (20/60/120d), per-asset risk parity, portfolio vol-managed + crash protection; plus x-sec momentum×TS-strength tilt and their risk-parity combo. Net 4.5bps+funding. HL era, OOS=last40%.

| book | Sharpe (HL) | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| Directional trend (raw) | +1.40 | +1.70 | +0.94 | +21% | -12% |
| Directional trend (vol-managed) | +1.52 | +1.84 | +1.15 | +24% | -12% |
| X-sec momentum×TS-strength | +0.88 | +1.48 | +0.02 | +11% | -16% |
| Combo (managed+tilt, risk-parity) | +1.75 | +2.36 | +0.83 | +25% | -13% |

## Honest verdict

- Best book: **Directional trend (vol-managed)**, OOS Sharpe **+1.15**, deflated (26 trials) **+1.15**, P(SR>0)=0.30 (does NOT clear 95%).
- Sharpe 3 NOT reached. Vol-managed trend is the strongest price class but still lands in the ~1.5-2.0 band, the same honest ceiling. Iteration 4; price wall holds.
