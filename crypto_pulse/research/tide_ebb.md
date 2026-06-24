# EBB (equity reversal) + TIDE+EBB cross-asset portfolio (honest)

TIDE inverts on equities, so EBB = x-sectional REVERSAL on stocks (long oversold/short overbought), gated to choppy regimes. Validated, not just sign-flipped. Equity cost 2bps.

| book | Sharpe | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| EBB stocks-96 (regime) | **+0.24** | +0.53 | -0.12 | +2% | -34% |
| EBB stocks-96 (plain) | **+0.25** | +0.55 | -0.12 | +2% | -34% |
| EBB stocks-430 (regime) | **+0.52** | +0.67 | +0.21 | +8% | -51% |

## EBB robustness (stocks-96)

| year | Sharpe |
|---|---|
| 2010 | +0.17 |
| 2011 | +0.01 |
| 2012 | -1.87 |
| 2013 | +1.49 |
| 2014 | +1.02 |
| 2015 | +0.03 |
| 2016 | -0.18 |
| 2017 | -0.63 |
| 2018 | -0.18 |
| 2019 | +0.60 |
| 2020 | +0.45 |
| 2021 | -0.59 |
| 2022 | -0.46 |
| 2023 | -1.14 |
| 2024 | +0.79 |
| 2025 | +0.88 |
| 2026 | -3.72 |

| equity cost | Sharpe |
|---|---|
| 2bps | +0.24 |
| 5bps | -0.20 |
| 10bps | -0.92 |
| 20bps | -2.32 |

## TIDE (crypto) + EBB (equity) cross-asset portfolio — HL era

- Correlation TIDE vs EBB: **-0.02** (genuinely uncorrelated -> real diversification).
- TIDE +2.01, EBB +0.06, **risk-parity combo +1.40** (-0.61 vs the better leg).
- Sharpe 3 NOT reached; combined ~1.4. Deployable on HL: TIDE on crypto perps, EBB on HIP-3 equity perps.

## Verdict (honest — EBB does NOT validate)

- **The reversal SIGN is right but EBB is not a tradeable book.** Equity reversal is only +0.24 Sharpe on large-caps (IS +0.53 / **OOS -0.12**), regime-unstable year-to-year, and **dies above ~2bps cost** (5bps -> -0.20, 10bps -> -0.92). Short-term equity reversal is real academically but arbitraged away net of realistic costs — flipping TIDE's sign does NOT recover a clean book.
- **So the cross-asset combo does NOT help.** TIDE-EBB correlation is genuinely -0.02 (the diversification premise was correct!), but EBB's ~0 Sharpe means adding it DILUTES rather than diversifies: combo +1.40 < TIDE-alone +2.01. Cross-asset diversification only lifts Sharpe when BOTH legs are individually strong; EBB isn't.
- **Net: TIDE alone (+2.01) stays the answer.** The honest equity-perp takeaway: neither TIDE (momentum, loses) nor EBB (reversal, too weak net of costs) is deployable on HL HIP-3 equity perps. Sharpe 3 unreached; the price/equity routes are exhausted, L4 order flow remains the only orthogonal lever.
