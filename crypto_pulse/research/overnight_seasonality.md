# Overnight / time-of-day seasonality on crypto (hourly, net)

27 coins, 2024-06-14->2026-06-14 (17510 hourly bars). Causal per-hour drift (trailing 30d). Net of 4.5bps taker on position changes.

| strategy | Sharpe | IS | OOS | ann ret | turn/day |
|---|---|---|---|---|---|
| hour-drift sign (long/short) | **-5.89** | -5.62 | -6.45 | -453% | 24.4 |
| long positive hours only | **-4.32** | -3.54 | -5.82 | -234% | 12.2 |
| fixed long 00-08 UTC (overnight) | **-0.46** | +0.13 | -1.57 | -19% | 2.0 |
| fixed long 22-02 UTC | **-1.80** | -1.37 | -2.68 | -58% | 2.0 |

Mean basket return by UTC hour (bps, full-sample descriptive): best 21h (+5.5), 22h (+4.2); worst 23h (-6.5). Spread 12.0bps/hr.

## Verdict

- If every Sharpe above is weak/negative net of taker, the seasonality is real in gross terms but the hourly turnover x 4.5bps eats it — same taker wall as all intraday crypto. The fixed-window versions (lowest turnover) are the only ones with a chance; if even those don't clear, time-of-day is not a taker-viable sleeve here.
