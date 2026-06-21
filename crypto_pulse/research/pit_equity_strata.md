# STRATA through PIT stock data (survivorship-free S&P)

Cross-sectional STRATA sleeves on the PIT S&P panel (720 names, real volume + PIT membership, top-150 liquid members), net 2bps, shrunk-MV.

## Equity-STRATA sleeves (full history)

| sleeve | Sharpe | weight |
|---|---|---|
| TREND | +0.15 | 7% |
| MOM | +0.37 | 29% |
| STREV | +0.20 | 12% |
| LOWVOL | +0.62 | 52% |
| BAB | -0.23 | 0% |
| ACCEL | -0.29 | 0% |
| VOLSHOCK | -0.62 | 0% |

**PIT equity-STRATA: Sharpe +0.73 (IS +0.99/OOS +0.35), maxDD -26%** (2004-08-09->2026-06-12)

## Multi-asset combine (weekly)

Per-book Sharpe: EQ-STRATA +0.23, VOL-CRYPTO +2.13, STRATA-CRYPTO +1.67

Mean pairwise correlation: **+0.14**

| combiner | Sharpe | IS | OOS | maxDD |
|---|---|---|---|---|
| equal-risk | **+2.04** | +2.36 | +1.57 | -8% |
| Sharpe-opt (IS) | **+2.36** | +2.68 | +1.88 | -8% |

## Verdict

- PIT equity-STRATA Sharpe +0.73; multi-asset combined OOS **+1.88**. Survivorship-free equity book is still modest after cost; stack ~2-2.5, not 3.

