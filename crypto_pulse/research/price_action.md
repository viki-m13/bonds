# Price-action backtests on crypto (Coinbase 1-min, 15 coins, 60d), net of HL taker

Crypto 24/7: sessions on the UTC day. Enter next-bar open after signal, 4.5bps/side taker, equal-weight portfolio. **IS/OOS = first/second half of the 60-day window** — the honesty gate.

| strategy | port Sharpe | IS | OOS | annRet | med-coin Sharpe | %coins>0 |
|---|---|---|---|---|---|---|
| ORB-15 | +1.74 | -0.38 | +2.88 | +58% | +1.45 | 73% |
| ORB-30 | +0.68 | -2.91 | +2.59 | +21% | +0.38 | 60% |
| ORB-60 | +3.48 | -5.20 | +8.55 | +112% | +2.87 | 87% |
| intraday-mom-60 | -0.31 | -4.45 | +1.66 | -11% | -0.30 | 33% |
| intraday-mom-30 | -1.15 | -1.40 | -1.10 | -40% | -0.88 | 33% |
| vwap-reclaim(long) | -26.55 | -23.87 | -30.08 | -776% | -18.84 | 0% |

**Verdict:** the only high full-sample Sharpe (ORB-60, 3.48) is a MIRAGE — IS −5.2 vs OOS +8.5: it lost badly in the first 30 days and won big in the last 30, i.e. a pure trend-regime bet on 60 days, not an edge. Intraday-momentum is negative and VWAP-reclaim is destroyed by whipsaw+cost. None of these is a validated edge; 60 days is far too short and one regime. No taker-viable price-action Sharpe 3 here.
