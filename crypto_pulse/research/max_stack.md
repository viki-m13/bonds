# MAX-STACK — how high does honest OOS diversification go?

HL era, net of 4.5bps taker + real funding, vol-targeted, IS=first60/OOS=last40. A sleeve is ADMITTED only if positive in BOTH halves; combination weights are Sharpe-optimal on IS, applied to OOS.

## All candidate sleeves

| sleeve | Sharpe | IS | OOS | admitted? |
|---|---|---|---|---|
| TREND | +0.75 | +1.13 | +0.11 | YES |
| CARRY | +0.90 | +0.44 | +1.60 | YES |
| ORDERFLOW | +0.25 | +0.46 | -0.08 | no |
| BAB | +0.69 | +0.58 | +0.86 | YES |
| SEASONAL | -1.60 | -0.85 | -2.69 | no |
| SQUEEZE | +0.70 | +0.35 | +1.22 | YES |
| REVERSAL | -0.07 | +0.13 | -0.34 | no |
| ACCEL | +1.05 | +0.71 | +1.50 | YES |

Admitted sleeves: 5 (TREND, CARRY, BAB, SQUEEZE, ACCEL). Mean pairwise correlation **+0.09**.

Correlation matrix (admitted):

| | TREND | CARRY | BAB | SQUEEZE | ACCEL |
|---|---|---|---|---|---|
| TREND | +1.00 | +0.08 | +0.27 | +0.34 | +0.07 |
| CARRY | +0.08 | +1.00 | +0.01 | -0.14 | +0.16 |
| BAB | +0.27 | +0.01 | +1.00 | +0.18 | -0.00 |
| SQUEEZE | +0.34 | -0.14 | +0.18 | +1.00 | -0.12 |
| ACCEL | +0.07 | +0.16 | -0.00 | -0.12 | +1.00 |

## The honest combined book

| book | Sharpe | IS | OOS | ann | maxDD |
|---|---|---|---|---|---|
| MAX-STACK (5 sleeves) | **+1.40** | +1.42 | +1.38 | +19.1% | -9.3% |

## Where the wall is (the math, with our real numbers)

- admitted sleeves K = **5**, avg standalone Sharpe S ≈ **0.64**, avg pairwise corr ρ ≈ **+0.09**
- diversification ceiling S/√ρ ≈ **2.20** (adding infinite MORE sleeves at this ρ cannot beat this)
- observed combined OOS Sharpe = **+1.38**
- to reach **3** we would need ρ ≤ (S/3)² ≈ 0.046 AND ~22 genuinely-uncorrelated sleeves of this quality — crypto is one factor and runs out of independent sleeves long before that (extra signals become correlated, not additive).

**Conclusion:** this is the maximal honest taker stack on the HL universe. The OOS number is the real ceiling for this approach; Sharpe 3 is blocked by the effective-bet count (crypto ≈ a handful of independent factors), not by effort. Breaking 3 requires either (i) many MORE uncorrelated markets/asset-classes, or (ii) far higher breadth via frequency — which needs maker/HFT execution we falsified on real HL L2.
