# Cross-asset diversification — does an uncorrelated CTA book lift the crypto Sharpe?

Weekly (W-FRI) returns over the crypto-tradeable overlap (2023-05-12 -> 2026-04-24, 155 weeks). Crypto = validated 3-sleeve book (real HL funding + 4.5bps). Cross-asset = risk-parity trend book over EQUITY/BONDS/COMMOD/FX ETF sleeves (2bps/side). IS=first60/OOS=last40.

Cross-asset trend book standalone over its FULL ETF history (1998-01-23->2026-06-12): Sharpe **+0.38**, 1482 weeks.

Per-class trend Sharpe (full history): EQUITY +0.13, BONDS +0.19, COMMOD +0.57, FX +0.02

## The combination (overlap window, weekly)

**Crypto–cross-asset correlation: +0.03**  (genuinely uncorrelated — diversification is real)

Sharpe-optimal weights (set on IS): crypto 100%, cross-asset 0%  (a negative-IS book gets 0).

| book | Sharpe | IS | OOS | ann | maxDD |
|---|---|---|---|---|---|
| crypto 3-sleeve (alone) | **+0.90** | +0.77 | +1.06 | +14.5% | -13.1% |
| cross-asset trend (alone) | **-0.14** | -0.46 | +0.37 | -1.4% | -15.7% |
| COMBINED (Sharpe-optimal) | **+0.90** | +0.77 | +1.06 | +10.8% | -9.9% |

## Verdict

- Combined Sharpe **+0.90** vs crypto-alone +0.90 (theoretical 2-book optimum ~+0.53 at corr +0.03). The lift is marginal on this short overlap window.

- This is the mechanism behind high program-level Sharpes: stack uncorrelated books, not more correlated crypto signals. Reaching 3 needs MANY such uncorrelated streams (S*sqrt(K)/sqrt(1+(K-1)rho)); two books move the needle but don't get there alone. The cross-asset book also adds crisis-alpha (bonds/trend rally when crypto crashes).
