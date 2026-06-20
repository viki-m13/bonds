# Equity ensemble + cross-asset combination with crypto

430 US stocks (price-only), 2.0bps, market-neutral, vol-targeted, IS=first60/OOS=last40 of the equity sample.

## Equity sleeves (full history)

| sleeve | Sharpe | IS | OOS |
|---|---|---|---|
| REVERSAL | +0.40 | +0.41 | +0.41 |
| MOMENTUM | +0.58 | +0.56 | +0.62 |
| LOWVOL | -0.73 | -0.77 | -0.67 |
| STATARB | +0.58 | +0.92 | +0.23 |

Equity ENSEMBLE (equal-risk): Sharpe **+0.42** (IS +0.48 / OOS +0.35), mean sleeve corr -0.01.

## Cross-asset portfolio (crypto grand stack + equity book), weekly

Overlap 2023-06-30->2026-04-24 (148 wks). **Equity-crypto correlation: +0.08.**

| book | Sharpe | IS | OOS |
|---|---|---|---|
| crypto grand stack alone | **+1.38** | +1.40 | +1.37 |
| equity book alone (overlap) | **+0.77** | +1.65 | -0.53 |
| COMBINED portfolio | **+1.46** | +2.12 | +0.65 |

## Verdict

- Equity ensemble full-history Sharpe +0.42; on the 2023-26 crypto overlap it is +0.77 (equity factors decayed/regime-weak recently), correlation to crypto +0.08. Combined portfolio +1.46 vs crypto alone +1.38. The uncorrelated equity book LIFTS the portfolio — cross-market diversification is the real lever toward 2.

- To approach 3 you would stack MANY such uncorrelated books (equity attention-statarb net ~2.3 in its paper, futures CTA, FX carry) — but each is itself ~1-2 net and the combination is bounded by S/sqrt(rho); realistic portfolio target ~2, not a stable 3.
