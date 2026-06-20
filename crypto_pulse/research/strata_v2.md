# STRATA v2 — consolidated improvements vs VOL

7 sleeves (incl. VOLSHOCK), shrunk-MV, +/- regime gross + faster vol-target. HL era, net, IS/OOS.

| STRATA v2 variant | Sharpe | IS | OOS | maxDD |
|---|---|---|---|---|
| v2 shrunk-MV | **+1.37** | +1.24 | +1.55 | -16% |
| v2 + regime | **+1.60** | +1.80 | +1.24 | -19% |
| v2 + faster vt(20) | **+1.32** | +1.20 | +1.49 | -15% |
| v2 + regime + faster vt(20) | **+1.57** | +1.70 | +1.36 | -16% |

**STRATA v2 chosen: v2 + regime + faster vt(20)** -> Sharpe +1.57 (min(IS,OOS) +1.36).

## STRATA v2 vs VOL vs 50/50 blend (HL era)

| book | Sharpe | CAGR | maxDD |
|---|---|---|---|
| STRATA v2 | **+1.57** | +24% | -16% |
| VOL | **+1.67** | +26% | -11% |
| 50/50 BLEND | **+2.05** | +25% | -9% |

corr(STRATA2, VOL) = +0.25

## Verdict

- STRATA v2 = **+1.57** (from ~1.46 baseline) — the improvements (VOLSHOCK + shrunk-MV + sizing) are a real, robust gain. Still below VOL standalone (+1.67) — VOL's intraday vol-timing is hard to match in a daily taker book.
- 50/50 blend = **+2.05**, maxDD -9% (corr +0.25) — still the best single configuration.
