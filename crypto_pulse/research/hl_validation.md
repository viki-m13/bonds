# PULSE on Hyperliquid perps — validation

Universe: 57 coins listed on HL perps AND in our daily panel. Funding history present for 39/57 (HL launched ~2023-05). Costs: HL taker 4.5 bps/side; realized hourly HL funding charged on signed notional. 12% annual portfolio vol target.

> NOTE: funding download still in progress (39/57 coins); funding-inclusive rows are partial until it completes.

## Headline (HL-tradeable era, 2023-05 -> present)

| config | Sharpe | ann | vol | maxDD | days |
|---|---|---|---|---|---|
| PULSE-HL (fees+funding) | +0.79 | +10.4% | 13.2% | -13.0% | 1079 |
|   fees only (no funding) | +0.92 | +12.1% | 13.2% | -12.7% | 1079 |

## Full sample (pre-2023 = spot proxy, no real HL funding)

| config | Sharpe | ann | vol | maxDD | days |
|---|---|---|---|---|---|
| fees only, full sample | +1.33 | +19.7% | 14.8% | -15.7% | 4148 |

## P&L attribution, HL era (annualized return contribution)

- gross trend/breakout: **+14.2%**
- HL taker fees:        **-2.0%**  (turnover 0.12/day)
- funding (net):        **-1.8%**
- **net:                +10.4%**

## Leverage & liquidation (HL era)

- gross leverage (sum|w|): mean 0.26x, 95th pct 0.43x, max 0.66x
- net leverage (directional tilt): mean -0.03x, range [-0.52, +0.51]
- worst 1-day P&L: -3.0%; max drawdown -13.0%
- account collateral vs HL maintenance margin: at mean gross leverage 0.26x and a blended ~5% maintenance margin, a same-day adverse move of ~379% across the whole book would be required to liquidate — vs the observed worst day of -3.0%. Liquidation risk is negligible at this vol target.

## HL-era by year (fees+funding)

| year | Sharpe | ann | maxDD | days |
|---|---|---|---|---|
| 2023 | +2.04 | +28.3% | -7.8% | 234 |
| 2024 | +0.62 | +8.5% | -13.0% | 366 |
| 2025 | +0.51 | +6.4% | -10.4% | 365 |
| 2026 | -0.58 | -7.3% | -7.7% | 114 |

## Leverage scenarios (HL era) — the strategy is intrinsically low-leverage

| vol target | mean gross lev | peak gross lev | ann return | Sharpe | maxDD |
|---|---|---|---|---|---|
| 10% | 0.22x | 0.44x | +8.7% | +0.79 | -10.9% |
| 20% | 0.43x | 0.87x | +17.3% | +0.79 | -21.1% |
| 40% | 0.87x | 1.74x | +34.7% | +0.79 | -39.5% |
| 60% | 1.30x | 2.62x | +52.0% | +0.79 | -54.9% |

Leverage scales return and risk together (Sharpe invariant); even a 60% vol target runs ~1x gross on HL, far inside the 10-40x caps. Liquidation is not the binding constraint — drawdown tolerance is.

## Funding stress (HL era)

| funding multiplier | Sharpe | ann | net funding drag |
|---|---|---|---|
| 1x | +0.79 | +10.4% | -1.8% |
| 2x | +0.66 | +8.6% | -3.5% |
| 3x | +0.52 | +6.9% | -5.3% |
