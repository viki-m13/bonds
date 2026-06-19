# Statistical arbitrage (Avellaneda-Lee residual reversion) on HL

Net of 4.5bps taker + real funding, vol-targeted, IS=first60/OOS=last40. Market/factor-neutral residual reversion.

## Parameter scan (window / #PCA factors / hold)

| win | K | hold | Sharpe | IS | OOS | ann | maxDD | turn/day |
|---|---|---|---|---|---|---|---|---|
| 40 | 2 | 2 | -1.66 | -1.46 | -1.96 | -21.9% | -54.2% | 0.46 |
| 40 | 2 | 3 | -1.02 | -0.85 | -1.27 | -13.6% | -42.0% | 0.36 |
| 40 | 2 | 5 | -0.59 | -1.02 | +0.08 | -8.1% | -36.2% | 0.26 |
| 40 | 3 | 2 | -1.36 | -1.28 | -1.47 | -17.9% | -49.0% | 0.47 |
| 40 | 3 | 3 | -0.80 | -0.91 | -0.62 | -10.7% | -36.3% | 0.36 |
| 40 | 3 | 5 | -0.78 | -1.12 | -0.23 | -11.1% | -40.8% | 0.26 |
| 40 | 5 | 2 | -1.47 | -1.52 | -1.38 | -19.3% | -50.9% | 0.48 |
| 40 | 5 | 3 | -0.98 | -1.25 | -0.56 | -13.1% | -41.6% | 0.36 |
| 40 | 5 | 5 | -0.68 | -0.94 | -0.29 | -9.1% | -34.4% | 0.26 |
| 60 | 2 | 2 | -0.17 | -0.60 | +0.50 | -2.2% | -33.2% | 0.39 |
| 60 | 2 | 3 | -0.37 | -0.55 | -0.08 | -4.8% | -33.3% | 0.31 |
| 60 | 2 | 5 | +0.28 | -0.42 | +1.35 | +3.7% | -23.5% | 0.22 |
| 60 | 3 | 2 | -0.59 | -1.04 | +0.10 | -7.7% | -38.2% | 0.39 |
| 60 | 3 | 3 | -0.32 | -0.43 | -0.17 | -4.2% | -32.0% | 0.31 |
| 60 | 3 | 5 | +0.32 | -0.10 | +0.97 | +4.3% | -17.3% | 0.23 |
| 60 | 5 | 2 | -1.02 | -1.31 | -0.59 | -13.2% | -44.6% | 0.40 |
| 60 | 5 | 3 | -0.80 | -0.78 | -0.82 | -10.3% | -39.1% | 0.31 |
| 60 | 5 | 5 | -0.31 | -0.77 | +0.41 | -4.1% | -30.4% | 0.23 |

**Best (by min(IS,OOS)):** win=60, K=3, hold=5 -> Sharpe +0.32 (IS -0.10 / OOS +0.97).

## Verdict

- Residual reversion is NOT robust net of taker cost (fails one half). Correlation of stat-arb to the directional sleeve stack: **-0.04**. Crypto residual reversion at DAILY+ horizon (not intraday, which was bid-ask bounce) is the honest test. If positive and uncorrelated it is a new sleeve for the stack; if not, the archetype is taker-blocked here and we pivot to the next.
