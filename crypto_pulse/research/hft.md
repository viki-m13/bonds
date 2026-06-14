# VELOCITY — 1-minute market-neutral crypto stat-arb (honest)

Data: Coinbase 1-min, 2026-04-15 14:01:00 -> 2026-06-14 14:00:00  (85,921 min, 15 coins). Execution: enter next-min OPEN, hold H min, exit at open (formation bar skipped). Dollar-neutral, beta-hedged. Sharpe annualized at 525,600 min/yr. IS = first 70% of the window, OOS = last 30%.

## Gross edge vs cost — the whole story is the breakeven

The signal is strong and OOS-robust GROSS, but the per-trade edge is sub-bp, so whether it nets out is purely an execution-cost question. `edge/trade` and `breakeven` are bps per side.

| hold | gross Sharpe | OOS gross | turn/reb | edge/trade (bps) | breakeven (bps/side) | net@0.2 | net@1.5 (HL maker) | net@4.5 (HL taker) |
|---|---|---|---|---|---|---|---|---|
| 1m | +58.0 | +78.6 | 1.44 | 0.236 | 0.164 | -12.6 | -460.4 | -1289.0 |
| 3m | +28.0 | +34.9 | 1.42 | 0.332 | 0.234 | +4.1 | -150.2 | -477.7 |
| 5m | +18.2 | +21.2 | 1.42 | 0.336 | 0.237 | +2.9 | -96.4 | -312.7 |
| 10m | +7.1 | +10.8 | 1.41 | 0.266 | 0.188 | -0.5 | -49.5 | -159.5 |
| 20m | +8.4 | +7.3 | 1.40 | 0.614 | 0.438 | +4.6 | -20.3 | -76.9 |
| 45m | +2.0 | +4.1 | 1.41 | 0.321 | 0.228 | +0.3 | -11.2 | -37.5 |

## Component decomposition (gross, H=5m)

| component | gross Sharpe | OOS |
|---|---|---|
| lead-lag continuation only | +0.2 | -4.3 |
| residual reversal only | +18.7 | +22.0 |
| combined (VELOCITY) | +18.2 | +21.2 |

## Verdict

- **GATE RESULT (see maker-fill sim below):** with a realistic taker EXIT the strategy is deeply negative even at maker-rebate fees — the ~0.3bps/trade edge cannot survive crossing the spread once. The positive net@0.2bps figures are the OPTIMISTIC idealization of making on BOTH legs at ~0 fee with guaranteed passive fills. So VELOCITY is viable only as full professional market-making, not as anything that ever takes liquidity.
- The lead-lag + residual-reversal signal is **real and OOS-robust** with enormous GROSS Sharpe (breadth: ~100k+ bets/yr).
- But the **edge is sub-bp per trade and diffuse** across the cross-section (concentrating into the extreme signals *loses* — the edge is liquidity provision, not direction). Breakeven cost is ~0.2–0.9 bps/side.
- **Net Sharpe ≥3 is reachable only at ≤~0.3 bps effective cost** — i.e. as a passive **maker capturing the spread + rebate**, NOT as a taker. At HL taker (4.5) or even maker (1.5) it is negative. Whether the maker fills actually materialise cannot be proven from 1-min bars — it needs L2/queue simulation. So the honest claim is: **the alpha exists and is huge gross, but it lives entirely inside the fee/spread and is a market-making strategy, not a taker bot.**

## Maker-fill reality check (touch-fill via 1-min high/low, real HL spreads)

Passive entry one half-spread inside the open, filled only if the bar actually trades to it (embeds adverse selection + misses); taker exit at +H (pays taker fee + half-spread). Real HL half-spreads: BTC/SOL/AVAX/DOGE ~0.06-0.08bps, ETH/BCH ~0.25-0.30, the rest 0.9-3.7bps. HL base maker fee is 1.5bps (only top rebate tiers reach ~0).

Tight-spread liquid subset: AVAX, BCH, BTC, DOGE, ETH, SOL

| universe | hold | maker fee | fill rate | net Sharpe | OOS |
|---|---|---|---|---|---|
| all 15 | 5m | base 1.5bps | 77% | -253.43 | -225.70 |
| all 15 | 5m | top 0.0bps | 77% | -202.83 | -182.44 |
| all 15 | 5m | rebate -0.3bps | 77% | -192.70 | -173.78 |
| all 15 | 10m | base 1.5bps | 77% | -125.96 | -113.52 |
| all 15 | 10m | top 0.0bps | 77% | -101.69 | -93.38 |
| all 15 | 10m | rebate -0.3bps | 77% | -96.84 | -89.36 |
| all 15 | 20m | base 1.5bps | 78% | -64.25 | -56.61 |
| all 15 | 20m | top 0.0bps | 78% | -51.41 | -46.24 |
| all 15 | 20m | rebate -0.3bps | 78% | -48.84 | -44.16 |
| | | | | | |
| tight 6 | 5m | base 1.5bps | 79% | -245.79 | -204.47 |
| tight 6 | 5m | top 0.0bps | 79% | -196.57 | -164.04 |
| tight 6 | 5m | rebate -0.3bps | 79% | -186.71 | -155.95 |
| tight 6 | 10m | base 1.5bps | 79% | -119.90 | -106.13 |
| tight 6 | 10m | top 0.0bps | 79% | -96.30 | -86.37 |
| tight 6 | 10m | rebate -0.3bps | 79% | -91.58 | -82.42 |
| tight 6 | 20m | base 1.5bps | 80% | -61.05 | -50.34 |
| tight 6 | 20m | top 0.0bps | 80% | -48.87 | -41.03 |
| tight 6 | 20m | rebate -0.3bps | 80% | -46.43 | -39.17 |
| | | | | | |

*Taker exit is the conservative choice; a maker exit (post + hope to get filled) would lower cost but add more miss/adverse-selection risk. Even so this shows whether the edge survives paying to cross only once.*
