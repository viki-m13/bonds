# MOSAIC — regime-adaptive technical/price-action ensemble (HL)

HL-tradeable era 2023-05-12->2026-04-24, 57 coins, real HL funding + 4.5bps taker, 12% vol target. IS=first 60%, OOS=last 40%.

## Standalone signals

| signal | family | Sharpe | IS | OOS |
|---|---|---|---|---|
| trend | trend | +0.75 | +1.13 | +0.11 |
| tsmom | trend | +0.94 | +1.07 | +0.74 |
| breakout | trend | +0.51 | +1.10 | -0.41 |
| accel | trend | +0.65 | +1.25 | -0.22 |
| carry | mr | +0.98 | +0.05 | +2.36 |
| lowvol | mr | +0.83 | +1.41 | -0.02 |
| volshock | trend | +0.77 | +1.13 | +0.19 |

Mean pairwise sleeve correlation: **+0.18** (low = good for ensembling).

## Ensembles vs the best single sleeve

Best single sleeve: **carry** (Sharpe +0.98).

| ensemble | Sharpe | IS | OOS | ann | maxDD |
|---|---|---|---|---|---|
| static equal-weight (7 sig) | **+0.39** | +1.05 | -0.58 | +5.1% | -18.3% |
| regime-adaptive | **+0.42** | +0.99 | -0.44 | +5.6% | -18.1% |
| IC-decay weighted | **+0.45** | +1.16 | -0.55 | +6.2% | -22.9% |
| regime + IC (MOSAIC, 7 sig) | **+0.50** | +1.15 | -0.47 | +6.8% | -22.4% |
| PARSIMONIOUS trend+carry (2 sleeve) | **+1.11** | +0.74 | +1.68 | +12.1% | -14.9% |

## Verdict (the honest, counter-intuitive result)

- **More techniques did NOT help.** The elaborate 7-signal MOSAIC (Sharpe +0.50, OOS negative) is *beaten* by the PARSIMONIOUS 2-sleeve trend+carry blend (Sharpe +1.11, IS +0.74, OOS +1.68). The extra price-action signals (breakout, acceleration, volume-shock, low-vol) are mostly redundant trend exposure or noise — they dilute rather than diversify, and they dragged the OOS half (a trend-hostile, carry-favourable regime).
- Diversification across the two *genuinely* uncorrelated families (mean sleeve corr +0.18) is what matters; more correlated trend variants add variance, not Sharpe. The adaptive combiners (regime rotation, IC-decay) help the kitchen-sink version at the margin (0.39->0.50) but can't rescue it.
- **Net of real HL fees+funding the honest deployable book is the 2-sleeve trend+carry (~1.1-1.3), not a many-signal ensemble** — and still short of 3 (maker-only, per STRATEGY_RESEARCH.md). Parsimony beats the ensemble here.
