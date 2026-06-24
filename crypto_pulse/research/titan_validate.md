# Validate TITAN — causal or lookahead? (honest)

TITAN ships weights + returns. Reconstruct from weights x actual coin returns (50 coins matched). Lag-1 = causal; lag-0 = peeks at same-day returns.

## Reconstruction

| series | Sharpe | CAGR | corr to published |
|---|---|---|---|
| published returns | +1.47 | +51% | 1.00 |
| **causal (lag-1, 20bps)** | **+1.49** | +53% | +1.00 |
| contemporaneous (lag-0) | +1.51 | +52% | +0.98 |
| causal @ 40bps (stress) | +1.41 | +49% | +1.00 |

## Lookahead diagnostic

- Lag-0 minus lag-1 Sharpe gap: **+0.01**. Small gap -> weights are causal; lag-1 already works.
- Causal reconstruction correlation to published: +1.00 (tracks the published series -> returns derive causally from the weights.

## Year-by-year (causal reconstruction)

| year | Sharpe |
|---|---|
| 2017 | +2.86 |
| 2018 | -0.09 |
| 2019 | +1.41 |
| 2020 | +1.48 |
| 2021 | +1.25 |
| 2022 | -1.35 |
| 2023 | +0.75 |
| 2024 | +2.60 |
| 2025 | -0.04 |
| 2026 | -1.79 |

- Causal HL-era Sharpe +1.37; published HL-era +1.34.
- Causal last-365d +0.08.

## Verdict

- **TITAN validates as causal:** the lag-1 reconstruction is +1.49 Sharpe, tracks the published series (corr +1.00), and the lag-0/lag-1 gap is small (+0.01) — no evidence of same-day lookahead. It survives doubled cost (+1.41 @ 40bps).
- It is a DIRECTIONAL multi-sleeve crypto trend/breakout CTA (21 sleeves) — structurally orthogonal to TIDE's market-neutral book, which is why corr is ~0.03 (real, not fitted).
- Caveat: 21 sleeves is a lot of freedom; even if causal, its live Sharpe will likely be below backtest. Size the stack to the causal/stressed number, not the published one.
