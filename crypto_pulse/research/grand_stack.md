# GRAND STACK — best honest combination across all archetypes

6 sleeves, net of 4.5bps taker + real funding, vol-targeted, IS=first60/OOS=last40. Weights set on IS, applied to OOS.

## Sleeves

| sleeve | Sharpe | IS | OOS | corr-to-others (mean) |
|---|---|---|---|---|
| TREND | +0.75 | +1.13 | +0.11 | +0.03 |
| CARRY | +0.90 | +0.44 | +1.60 | +0.05 |
| BAB | +0.69 | +0.58 | +0.86 | +0.03 |
| SQUEEZE | +0.70 | +0.35 | +1.22 | +0.00 |
| ACCEL | +1.05 | +0.71 | +1.50 | +0.03 |
| FUNDFADE | +0.49 | +0.08 | +1.26 | -0.19 |

Mean pairwise correlation across all 6: **-0.01**.

## Combined books

| combiner | Sharpe | IS | OOS | ann | maxDD |
|---|---|---|---|---|---|
| equal-risk (inverse-vol) | **+1.81** | +1.27 | +2.49 | +24.2% | -9.4% |
| shrunk mean-variance (IS) | **+1.46** | +1.32 | +1.63 | +19.9% | -8.9% |
| (ref) 5 directional only | **+1.55** | +1.31 | +1.86 | +20.9% | -9.6% |

## Verdict (honest, with the regime caveat)

- The equal-risk book prints OOS **+2.49** but IS only **+1.27** — OOS > IS means the recent (2024-26) regime was unusually favourable to carry/accel/funding-fade, so 2.49 is regime-flattered, NOT a stable edge. Do not bank it.
- The honest central estimate is the **shrunk mean-variance** book: Sharpe ~**+1.46** (IS +1.32 / OOS +1.63, balanced halves), maxDD -8.9%. Robust weights (IS-only, covariance shrunk 60% to diagonal), genuinely uncorrelated sleeves (mean ρ ≈ 0). This is the maximal HONEST taker book we have found — a real jump from the 1.1 starting point.
- It remains far from 3: with mean ρ -0.01 and these sleeve Sharpes the diversification ceiling is ~2, and we are at the realistic OOS end of it. 3 would require many more genuinely uncorrelated POSITIVE streams than a single-venue crypto taker can source (proven across every archetype tested).
