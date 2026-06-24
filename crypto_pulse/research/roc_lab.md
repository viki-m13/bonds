# ROC-improvement lab — how high does a price-momentum book HONESTLY go?

ai-trader momentum/breakout signals as x-sectional crypto books, net 4.5bps+funding, vol-targeted. HL era; IS=first60% / OOS=last40%. 9 variants tried (deflation applied).

| signal | Sharpe (HL) | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| ROC20 (baseline) | +0.93 | +0.64 | +1.29 | +13% | -14% |
| ROC multi-horizon | +0.65 | +0.82 | +0.39 | +8% | -16% |
| 12-1 momentum | -0.37 | -0.19 | -0.64 | -6% | -37% |
| risk-adj momentum | +0.68 | +0.46 | +1.02 | +9% | -19% |
| acceleration | -0.18 | +0.70 | -1.34 | -3% | -31% |
| Donchian breakout | +0.46 | +0.63 | +0.21 | +5% | -15% |
| MACD x-sec | +0.94 | +1.36 | +0.36 | +12% | -11% |
| Bollinger z | +1.75 | +1.99 | +1.38 | +25% | -10% |
| TS-momentum tilt | +1.17 | +0.87 | +1.65 | +16% | -13% |

## Combined books (causal)

| combine | Sharpe (HL) | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| Signal-level z-blend (IS-admitted, netted) | +0.94 | +1.11 | +0.71 | +13% | -12% |
| Adaptive WF ensemble (trail-Sharpe wt) | +0.81 | +1.41 | -0.04 | +11% | -16% |
| Equal-wt of IS-positive signals | +1.23 | +1.60 | +0.73 | +17% | -12% |

Signal-level blend admitted on IS: ROC20 (baseline), ROC multi-horizon, risk-adj momentum, acceleration, Donchian breakout, MACD x-sec, Bollinger z, TS-momentum tilt.
Signal-level blend deflated OOS Sharpe (9 trials): +0.71, P(SR>0)=0.24.

## Honest verdict

- Best combined OOS Sharpe: **+0.73**.
- **Deflated Sharpe** of the ensemble OOS (haircut for 9 trials): annualized **-0.04**, P(SR>0 after deflation) = 0.06. Does NOT clear the multiple-testing bar at 95%.
- **Sharpe 3 NOT reached.** A pure price-momentum book on crypto, honestly walk-forwarded and deflated, lands around 0.7 — improving ROC (multi-horizon, 12-1, risk-adjusting, ensembling) lifts it from ~0.9 but plateaus well short of 3. This is the same ceiling STRATA's full multi-signal book hits (~1.85 OOS): price data alone does not yield Sharpe 3.
- Iterating further on price signals re-tests the SAME OOS and would only manufacture a lucky 3 via selection — the deflated Sharpe is precisely the guard against that. **The honest answer is the deflated number, and it is not 3.**
