# Equity STRATA + multi-asset toward Sharpe 3 OOS

Cross-sectional equity factors (MOM/STREV/LOWVOL/QMOM) on 430 US stocks, net 2bps, shrunk-MV. Then combined with crypto STRATA + VOL.

## Equity STRATA sleeves (daily, full history)

| sleeve | Sharpe |  weight |
|---|---|---|
| MOM | +0.54 | 52% |
| STREV | +0.39 | 33% |
| LOWVOL | +0.17 | 15% |
| QMOM | -0.13 | 0% |

**Equity STRATA combined: Sharpe +0.64, maxDD -26%** (2009-08-07->2026-05-07)

## Multi-asset combine (weekly): STRATA-EQ + VOL-CRYPTO + STRATA-CRYPTO

Per-book Sharpe (overlap): STRATA-EQ +0.64, VOL-CRYPTO +2.13, STRATA-CRYPTO +1.67

Mean pairwise correlation: **+0.12**

| combiner | Sharpe | IS | OOS | maxDD |
|---|---|---|---|---|
| equal-risk | **+2.29** | +2.44 | +2.05 | -5% |
| Sharpe-opt (IS) | **+2.42** | +2.71 | +1.98 | -7% |

## Verdict

- Combined OOS = **+1.98**. OOS +1.98 — equity STRATA Sharpe +0.64 is too weak to close the gap to 3 — equity factors here net ~0.6 after cost.

