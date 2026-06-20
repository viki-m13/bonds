# Dynamic STRATA/VOL allocator vs static 50/50 (HL era)

STRATA = 7-sleeve v2. Tilt toward the leading book on trailing 126d metric (monthly, lagged, clip 20-80%). IS/OOS.

| allocator | Sharpe | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| STATIC 50/50 | **+2.14** | +2.54 | +1.53 | +26% | -9% |
| DYN-ret | **+2.18** | +2.66 | +1.47 | +28% | -7% |
| DYN-sharpe | **+2.19** | +2.66 | +1.48 | +28% | -7% |
| DYN-invvol | **+2.13** | +2.51 | +1.52 | +26% | -9% |

## Verdict

- Best allocator: **STATIC 50/50** Sharpe +2.14 (min IS,OOS +1.53) vs static 50/50 +2.14. Static 50/50 is as good — leadership doesn't persist reliably enough to time without whipsaw; keep it simple.
