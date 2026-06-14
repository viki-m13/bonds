# VELOCITY — 1-minute market-neutral crypto stat-arb (honest)

Data: Coinbase 1-min, 2026-04-15 14:01:00 -> 2026-06-14 14:00:00  (85,953 min, 13 coins). Execution: enter next-min OPEN, hold H min, exit at open (formation bar skipped). Dollar-neutral, beta-hedged. Sharpe annualized at 525,600 min/yr. IS = first 70% of the window, OOS = last 30%.

## Gross edge vs cost — the whole story is the breakeven

The signal is strong and OOS-robust GROSS, but the per-trade edge is sub-bp, so whether it nets out is purely an execution-cost question. `edge/trade` and `breakeven` are bps per side.

| hold | gross Sharpe | OOS gross | turn/reb | edge/trade (bps) | breakeven (bps/side) | net@0.2 | net@1.5 (HL maker) | net@4.5 (HL taker) |
|---|---|---|---|---|---|---|---|---|
| 1m | +62.7 | +77.8 | 1.44 | 0.230 | 0.160 | -15.6 | -509.4 | -1388.5 |
| 3m | +31.4 | +32.4 | 1.42 | 0.328 | 0.231 | +4.2 | -171.0 | -535.5 |
| 5m | +20.7 | +23.1 | 1.42 | 0.350 | 0.247 | +3.9 | -104.1 | -336.3 |
| 10m | +16.3 | +20.0 | 1.42 | 0.535 | 0.377 | +7.7 | -48.4 | -172.9 |
| 20m | +9.3 | +11.4 | 1.40 | 0.570 | 0.407 | +4.7 | -25.0 | -92.1 |
| 45m | +4.4 | +6.1 | 1.39 | 0.563 | 0.405 | +2.2 | -11.9 | -44.2 |

## Component decomposition (gross, H=5m)

| component | gross Sharpe | OOS |
|---|---|---|
| lead-lag continuation only | +2.3 | +3.7 |
| residual reversal only | +20.7 | +23.1 |
| combined (VELOCITY) | +20.7 | +23.1 |

## Verdict

- The lead-lag + residual-reversal signal is **real and OOS-robust** with enormous GROSS Sharpe (breadth: ~100k+ bets/yr).
- But the **edge is sub-bp per trade and diffuse** across the cross-section (concentrating into the extreme signals *loses* — the edge is liquidity provision, not direction). Breakeven cost is ~0.2–0.9 bps/side.
- **Net Sharpe ≥3 is reachable only at ≤~0.3 bps effective cost** — i.e. as a passive **maker capturing the spread + rebate**, NOT as a taker. At HL taker (4.5) or even maker (1.5) it is negative. Whether the maker fills actually materialise cannot be proven from 1-min bars — it needs L2/queue simulation. So the honest claim is: **the alpha exists and is huge gross, but it lives entirely inside the fee/spread and is a market-making strategy, not a taker bot.**
