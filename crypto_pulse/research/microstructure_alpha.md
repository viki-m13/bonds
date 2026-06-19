# Microstructure alpha on rec_20260615_001055.jsonl

span 24.7 min, 8 coins. Information coefficient (signal vs forward mid-return) by horizon; pooled across coins (z-scored per coin). Net edge vs cost at the end.

## Information coefficient (IC) by feature x horizon

| feature | 1s | 2s | 5s | 10s |
|---|---|---|---|---|
| obi | +0.239 (t+35) | +0.211 (t+31) | +0.182 (t+27) | +0.150 (t+22) |
| micro | +0.204 (t+30) | +0.170 (t+25) | +0.145 (t+21) | +0.121 (t+18) |
| ofi | -0.003 (t-0) | -0.016 (t-2) | -0.036 (t-5) | -0.050 (t-7) |
| tflow | +0.017 (t+2) | +0.006 (t+1) | +0.002 (t+0) | -0.008 (t-1) |

Median quoted spread across coins: **0.9 bps** (half-spread 0.5). Taker round-trip cost ~9bps + spread.

## Cost-adjusted taker edge (best signal)

| feature | horizon | top-decile move (bps) | minus taker cost | n |
|---|---|---|---|---|
| obi | 10s | +1.13 | **-4.30** | 2197 |
| obi | 5s | +0.93 | **-4.50** | 2197 |
| obi | 2s | +0.69 | **-4.74** | 2197 |
| obi | 1s | +0.57 | **-4.86** | 2197 |
| micro | 10s | +0.50 | **-4.92** | 2194 |
| micro | 5s | +0.43 | **-5.00** | 2194 |
| micro | 2s | +0.39 | **-5.04** | 2194 |
| micro | 1s | +0.38 | **-5.05** | 2194 |

## Verdict

- The IC measures genuine short-horizon predictability; the cost-adjusted column is what a TAKER actually keeps. Every signal's predicted move is SMALLER than the half-spread + taker fee — the predictability is real but **not taker-monetizable**; it is maker-only (you must EARN the spread, not pay it), which puts us back in the queue/adverse-selection game maker_sim.py already showed we lose at retail latency. This is the honest microstructure result.

- Sample is 25 min — IC/edge are robust reads; a real Sharpe needs days of L2 (record_l2.py running forward).
