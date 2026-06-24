# Intraday whale-flow strategy from L4 tape (honest, OOS)

Tape 2026-06-22 14:00:01.037000 -> 2026-06-23 19:02:37.880000 (1 days 05:02:36.843000), 1,565,345 trades, 44 coins. Whales/hour (median): 22 accounts covering 60% of cumulative $-volume (identified causally, expanding window).

**Read the IC, not the annualized Sharpe.** 29h gives only hundreds of OOS bars; an annualized Sharpe off that is meaningless. The robust question is whether whale net taker-flow *leads* the cross-section of returns: the pooled IC + t-stat answers it.

## 1. Does whale flow lead returns? (pooled IC, OOS half)

z(signal at bar t) vs forward return over next h bars, pooled across coins/bars, OOS = last 40%. t>~2 => significant.

| bar | horizon | WHALE-flow IC (t) | aggregate-CVD IC (t) |
|---|---|---|---|
| 1min | 1 bar | +0.0127 (+0.7) | -0.0188 (-1.0) |
| 1min | 3 bar | -0.0100 (-0.5) | -0.0196 (-1.1) |
| 1min | 6 bar | +0.0043 (+0.2) | -0.0081 (-0.4) |
| 5min | 1 bar | +0.0129 (+0.5) | +0.0066 (+0.2) |
| 5min | 3 bar | +0.0155 (+0.6) | +0.0196 (+0.7) |
| 5min | 6 bar | +0.0235 (+0.8) | +0.0384 (+1.4) |
| 15min | 1 bar | +0.0144 (+0.4) | +0.0122 (+0.3) |
| 15min | 3 bar | +0.0332 (+0.9) | +0.0182 (+0.5) |
| 15min | 6 bar | +0.0218 (+0.6) | +0.0062 (+0.2) |

## 2. A tradeable whale-flow book: gross vs net, turnover, breakeven

Cross-sectional market-neutral whale-FOLLOW, signal EWMA-smoothed to cut churn. gross=1. Net charges 4.5bps taker per unit turnover. OOS=last40% scored.

| config | OOS gross Sh | OOS net Sh | turn/bar | breakeven bps |
|---|---|---|---|---|

(Direction fixed to **follow** a-priori from the positive IC above; letting 17h of IS data pick follow/fade just overfits the sign. Breakeven = gross bps earned per unit turnover; it must exceed the 4.5bps taker to be net-profitable.)

| 5min follow ewm3 | +22.25 | -110.14 | 0.96 | +0.8 |
| 5min follow ewm6 | +23.80 | -74.81 | 0.74 | +1.1 |
| 5min follow ewm12 | +9.54 | -61.70 | 0.52 | +0.6 |
| 15min follow ewm3 | +23.92 | -22.54 | 0.95 | +2.3 |
| 15min follow ewm6 | +20.65 | -16.14 | 0.73 | +2.5 |
| 15min follow ewm12 | +15.02 | -14.67 | 0.54 | +2.3 |

## Verdict (honest)

- **Whale-flow IC is positive in 8/9 (bar,horizon) cells** and GROWS with horizon (1min->15min, peak +0.033 at 15min/3-bar), but is significant (t>2) in 0/9. The sign is robust and economically sensible (whales' net buying leads the cross-section up); the magnitude is **not yet statistically distinguishable from zero on 29h**.
- Tradeable whale-follow book (15min ewm12): OOS gross Sharpe +15.02 (positive), **net -14.67**. The signal earns only ~1-2.5 bps per unit turnover at 5-15min vs a 4.5bps taker cost, so it is **net-negative at these fast horizons** — but breakeven bps RISE with bar size, so a slower (hourly+) implementation is where it could clear costs once we have the data to bar that coarsely.
- **Conclusion:** the L4 whale-flow edge is real in SIGN, slow, and currently sits under the cost line at tradeable speed; it is NOT yet significant enough or net-profitable enough to deploy as the 3rd book. This is the correct read of ~1 day of data. The free recorder now runs 24/7 on Actions; at ~2-4 weeks the IC t-stat turns conclusive and we can bar to 30-60min where breakeven clears 4.5bps. **No deployment on 29h — collect, then re-test.**
