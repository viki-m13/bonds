# Realistic execution cost for OUR daily book (not the HFT worst-case)

Top-30 PIT universe. Median name ADV (held, HL era) ≈ **$2,291M/day** (25th pct $482M). Daily turnover ~0.77x, gross ~1.3x, 30 names.

## Estimated slippage by account size (square-root impact + half-spread)

| account | clip/coin/day | participation (median name) | est. slippage | taker all-in | maker all-in |
|---|---|---|---|---|---|
| $0.1M | $3k | 0.000% | 1.2 bps | 5.7 bps | 2.7 bps |
| $1.0M | $33k | 0.001% | 1.7 bps | 6.2 bps | 3.2 bps |
| $10.0M | $334k | 0.015% | 3.1 bps | 7.6 bps | 4.6 bps |
| $100.0M | $3,337k | 0.146% | 7.7 bps | 12.2 bps | 9.2 bps |

## Book Sharpe (HL era, price sleeves) at realistic vs HFT costs

| cost assumption | all-in bps | Sharpe |
|---|---|---|
| maker, $1-10M acct (patient limits) | 3.2 | **+1.14** |
| taker, $1M acct | 6.2 | **+1.00** |
| taker, $10M acct | 7.6 | **+0.93** |
| taker, $100M acct | 12.2 | **+0.72** |
| vol-repo HFT worst-case (borrowed) | 14.5 | **+0.61** |

## Verdict

- **Realistic cost for our book is ~6.2 bps taker (or ~3.2 bps maker) at $1-10M** — fee plus ~1.7 bps slippage, because each daily clip is a tiny fraction of the top-30's ADV and can be worked patiently. That is the LOW end of the earlier sweep.
- At that realistic cost the price book is **+1.00** (and the full grand stack with funding sleeves is ~1.5); the 10.5-20.5 bps figures (Sharpe down to +0.61) were the vol repo's INTRADAY worst-case, which over-penalizes a daily book — useful as a stress floor, NOT the expected cost.
- Crucially, a DAILY rebalance can execute as a MAKER patiently (post limits, re-post if unfilled, the cost of a missed daily fill is tiny) — WITHOUT the fast-MM adverse-selection that killed the intraday maker in our L2 study. So ~2.5-4 bps maker all-in is achievable, where the book is strongest. We modeled the worst; the realistic case is better.
- Only at $100M+ does slippage start to bite (participation rises); below that, capacity is not the constraint.
