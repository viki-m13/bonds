# PULSE on Hyperliquid perps — validation

Universe: 57 coins listed on HL perps AND in our daily panel. Funding history present for 57/57 (HL launched ~2023-05). Costs: HL taker 4.5 bps/side; realized hourly HL funding charged on signed notional. 12% annual portfolio vol target.

## Headline (HL-tradeable era, 2023-05 -> present)

| config | Sharpe | ann | vol | maxDD | days |
|---|---|---|---|---|---|
| PULSE-HL (fees+funding) | +0.75 | +9.9% | 13.2% | -13.1% | 1079 |
|   fees only (no funding) | +0.92 | +12.1% | 13.2% | -12.7% | 1079 |

## Full sample (pre-2023 = spot proxy, no real HL funding)

| config | Sharpe | ann | vol | maxDD | days |
|---|---|---|---|---|---|
| fees only, full sample | +1.33 | +19.7% | 14.8% | -15.7% | 4148 |

## P&L attribution, HL era (annualized return contribution)

- gross trend/breakout: **+14.2%**
- HL taker fees:        **-2.0%**  (turnover 0.12/day)
- funding (net):        **-2.3%**
- **net:                +9.9%**

## Leverage & liquidation (HL era)

- gross leverage (sum|w|): mean 0.26x, 95th pct 0.43x, max 0.65x
- net leverage (directional tilt): mean -0.03x, range [-0.52, +0.51]
- worst 1-day P&L: -3.0%; max drawdown -13.1%
- account collateral vs HL maintenance margin: at mean gross leverage 0.26x and a blended ~5% maintenance margin, a same-day adverse move of ~379% across the whole book would be required to liquidate — vs the observed worst day of -3.0%. Liquidation risk is negligible at this vol target.

## HL-era by year (fees+funding)

| year | Sharpe | ann | maxDD | days |
|---|---|---|---|---|
| 2023 | +2.00 | +27.7% | -7.9% | 234 |
| 2024 | +0.57 | +7.7% | -13.1% | 366 |
| 2025 | +0.49 | +6.1% | -10.5% | 365 |
| 2026 | -0.61 | -7.7% | -7.7% | 114 |

## Leverage scenarios (HL era) — the strategy is intrinsically low-leverage

| vol target | mean gross lev | peak gross lev | ann return | Sharpe | maxDD |
|---|---|---|---|---|---|
| 10% | 0.22x | 0.44x | +8.2% | +0.75 | -11.0% |
| 20% | 0.43x | 0.87x | +16.4% | +0.75 | -21.3% |
| 40% | 0.87x | 1.75x | +32.9% | +0.75 | -39.8% |
| 60% | 1.30x | 2.62x | +49.3% | +0.75 | -55.3% |

Leverage scales return and risk together (Sharpe invariant); even a 60% vol target runs ~1x gross on HL, far inside the 10-40x caps. Liquidation is not the binding constraint — drawdown tolerance is.

## Funding stress (HL era)

| funding multiplier | Sharpe | ann | net funding drag |
|---|---|---|---|
| 1x | +0.75 | +9.9% | -2.3% |
| 2x | +0.57 | +7.6% | -4.6% |
| 3x | +0.40 | +5.3% | -6.9% |
