# PULSE across asset universes — does the crypto edge travel?

The IDENTICAL PULSE signal (multi-timeframe trend-sign + 20d Donchian breakout, inverse-vol, gross-normalised, directional long-uptrend/short-downtrend, 12% vol target) on four spot universes. `L/S` = the validated directional book (needs shorting -> margin/perps/inverse-ETFs). `long-only` = shorts clipped to flat (the true-spot version). Costs/side: crypto 10bps, ETFs 2-3bps, stocks 3bps.

| universe | #assets | med live | **L/S Sharpe** | L/S ann | L/S maxDD | L/S 2023+ | **long-only Sharpe** | LO ann | LO maxDD |
|---|---|---|---|---|---|---|---|---|---|
| spot crypto | 111 | 40 | **+1.20** | +17.8% | -16.4% | +0.55 | **+0.91** | +14.9% | -32.0% |
| spot ETFs (unlevered) | 14 | 14 | **-0.30** | -2.8% | -74.8% | +2.17 | **+0.41** | +4.5% | -46.0% |
| spot leveraged ETFs | 17 | 17 | **+0.14** | +1.6% | -35.5% | -0.39 | **+0.58** | +6.9% | -25.8% |
| PIT S&P 500 stocks | 720 | 376 | **-0.61** | -8.1% | -86.3% | -0.62 | **+0.39** | +5.0% | -43.4% |

## Reading

- **The trend/breakout edge is crypto-specific.** Directional L/S clears Sharpe ~1.2 only on spot crypto; it is flat-to-**negative** on unlevered ETFs (−0.30), leveraged ETFs (+0.14) and individual S&P 500 stocks (−0.61, and negative *before* vol-scaling). Crypto is the inefficient, strongly-trending market where this works; efficient equities mean-revert at these horizons and whipsaw a directional trend book (the deep L/S drawdowns are trend reversals like 2009/2020/2022, amplified by vol-targeting).
- **The only positive equity numbers are long-only**, and they are just market drift harvested via trend-timing (Sharpe ~0.4–0.6), not a genuine cross-asset trend alpha — a long/flat index would do similarly.
- **Caveats:** ETF universes are small (14 / 17 names) so the 2023+ column is noisy; leveraged-ETF series before ~2010 are vendor backfills (underlying × leverage); PIT-stock high/low equal close in the shipped panel, so Donchian uses close.
