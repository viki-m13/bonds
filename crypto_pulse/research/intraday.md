# Intraday trend on intraday bars — honest smooth-curve attempt

Coinbase hourly resampled to intraday bars, multi-timeframe trend + Donchian, inverse-vol, 15% portfolio vol target, net of 4.5bps taker. Causality: signal at close of bar, traded next bar. IS=first 60% / OOS=last 40%.

| bar | Sharpe | IS | OOS | ann | vol | maxDD | turn/bar |
|---|---|---|---|---|---|---|---|
| 2h | **-1.01** | -0.23 | -2.10 | -15.7% | 16% | -32.7% | 0.48 |
| 4h | **+0.17** | +0.59 | -0.41 | +2.7% | 16% | -21.8% | 0.47 |
| 6h | **+0.29** | +0.64 | -0.22 | +5.0% | 17% | -23.4% | 0.47 |
| 8h | **+0.65** | +0.63 | +0.68 | +11.8% | 18% | -17.4% | 0.46 |
| 12h | **+0.98** | +1.28 | +0.43 | +19.5% | 20% | -15.9% | 0.44 |
| 1d | **-0.06** | -0.46 | +0.56 | -1.2% | 18% | -19.6% | 0.40 |

- 12h + 2xATR trailing stop: Sharpe +1.02, maxDD -14.6%
- 12h + 3xATR trailing stop: Sharpe +1.01, maxDD -16.0%

## Best: 12h bars — Sharpe 0.98, ann +19.5%, vol 20%, maxDD -15.9%
