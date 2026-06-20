# Vol-channel breakout sleeve (from the `vol` repo) on our hourly data

Adaptive VWAP+/-band*sigma breakout, 10h eval, vol-targeted + DD-scaled, per-coin on 27 HL coins (hourly, 2024-2026). Net of HL costs. The vol repo's headline (Sharpe 4-6) is maker-only; here is the honest read on OUR data at real fees.

| execution | Sharpe (ann) | ann ret | maxDD | turn/day |
|---|---|---|---|---|
| gross (0bps) | **+0.98** | +10% | -9% | — |
| maker (1.5bps) | **+0.79** | +8% | -10% | — |
| taker (4.5bps) | **+0.42** | +4% | -13% | — |

## Diversification vs the daily grand stack (daily, HL overlap)

- vol-channel daily Sharpe: taker +0.49, maker +0.87; grand stack +1.74.
- correlation to grand stack: taker **+0.21**, maker **+0.21**.

| book | daily Sharpe | maxDD |
|---|---|---|
| grand alone | **+1.51** | -9% |
| grand + volch (taker) | **+1.26** | -9% |
| grand + volch (maker) | **+1.40** | -9% |

## Verdict

- On OUR hourly data at real fees: gross Sharpe +1.0, **net-taker +0.42**, net-maker +0.79. This reproduces the vol repo's own conclusion: the edge is largely eaten by turnover at the taker fee; maker is far better.
- As a DIVERSIFIER it is genuinely uncorrelated to the grand stack (taker corr +0.21). Adding the taker version takes the blend to +1.26 vs +1.51 grand-alone — not a lift at the taker fee (the sleeve is too weak net of cost; only its maker version would add).

- Honest consolidation: the vol-channel is a genuinely different (intraday TS breakout) alpha and is ~uncorrelated to our daily book, so it WOULD diversify — but only if executed as a maker, which our real-L2 study showed is adverse-selected at retail latency. Net-taker it doesn't clear. Same wall, independently reached from both repos.
