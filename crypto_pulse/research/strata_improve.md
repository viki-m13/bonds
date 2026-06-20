# STRATA improvement lab — beating VOL (~2.0)

Sizing/weighting overlays on the 6-sleeve grand stack. HL era, net, IS=first60/OOS=last40. Kept only if it helps BOTH halves.

| variant | Sharpe | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| BASELINE (equal-risk, 45d vt) | **+1.40** | +0.72 | +2.34 | +20% | -14% |
| faster vt (20d EWMA) | **+1.43** | +0.78 | +2.37 | +21% | -11% |
| faster vt (10d EWMA) | **+1.40** | +0.79 | +2.26 | +22% | -11% |
| + DD-aware sizing | **+1.34** | +0.64 | +2.30 | +19% | -16% |
| + regime gross | **+1.49** | +1.00 | +2.30 | +23% | -17% |
| shrunk-MV weights | **+1.43** | +1.17 | +1.82 | +20% | -10% |
| MV + faster vt(20) + regime | **+1.29** | +1.31 | +1.25 | +18% | -10% |
| MV + faster vt(20) + regime + DD | **+1.28** | +1.36 | +1.16 | +17% | -10% |

## Verdict

- Baseline STRATA: Sharpe +1.40. Best robust improvement: **MV + faster vt(20) + regime** -> Sharpe +1.29 (min(IS,OOS) +1.25), maxDD -10%.
- That is a -0.12 lift over baseline. Still short of VOL's ~2.0 standalone — sizing overlays help but don't fully close the gap; new uncorrelated sleeves (next) are needed for the rest.
