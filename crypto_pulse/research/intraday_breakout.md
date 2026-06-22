# Intraday vol-channel breakout book on the HL universe (hourly)

Our own multi-coin version of VOL's edge: 20 coins, hourly bars 2021-06-01->2026-06-14, VWAP/sigma window 24h, band 1.0 hourly-vol units, net 4.5bps taker + funding, vol-targeted, daily. HL era, IS=first60/OOS=last40.

| book | Sharpe | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| DIRECTIONAL momentum | **+0.28** | +0.73 | -0.63 | +4% | -27% |
| DIRECTIONAL reversion | **-1.09** | -1.44 | -0.42 | -25% | -60% |
| NEUTRAL momentum | **-0.13** | -0.08 | -0.75 | -2% | -17% |
| NEUTRAL reversion | **-0.76** | -0.96 | -0.34 | -9% | -30% |
| VOL (benchmark) | **+1.58** | +2.46 | +0.26 | +74% | -37% |

## Verdict

- Best of ours by OOS: **NEUTRAL reversion**, OOS Sharpe **-0.34**.
- VOL OOS +0.26. Ours does NOT beat VOL standalone OOS. Correlation of best book to VOL: -0.02.
- VOL + our best (equal risk) OOS: +0.24 (VOL alone +0.26) — no real lift to VOL.

