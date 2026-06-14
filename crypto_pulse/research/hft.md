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

- The lead-lag + residual-reversal signal is **real and OOS-robust** with enormous GROSS Sharpe (breadth: ~100k+ bets/yr).
- But the **edge is sub-bp per trade and diffuse** across the cross-section (concentrating into the extreme signals *loses* — the edge is liquidity provision, not direction). Breakeven cost is ~0.2–0.9 bps/side.
- **Net Sharpe ≥3 is reachable only at ≤~0.3 bps effective cost** — i.e. as a passive **maker capturing the spread + rebate**, NOT as a taker. At HL taker (4.5) or even maker (1.5) it is negative. Whether the maker fills actually materialise cannot be proven from 1-min bars — it needs L2/queue simulation. So the honest claim is: **the alpha exists and is huge gross, but it lives entirely inside the fee/spread and is a market-making strategy, not a taker bot.**
