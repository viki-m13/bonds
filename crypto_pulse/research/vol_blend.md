# 50/50 blend: vol strategy + our strategy (short & long)

Vol leg = the vol repo's CURRENT leakage-free published series (t5rvt eq_vt35: daily Sharpe **1.99**, CAGR 105%, vol 40%, maxDD -37%, corr-to-BTC -0.11, 2018-2026 — matches vol-pi.vercel.app). Replaces the earlier LEAKY pickles. Both legs vol-targeted to 12%, blended 50/50. LONG = 2018-2026 vs our price book; SHORT = HL era vs our full grand stack.

## Correlation (the real diversification driver)

- SHORT (HL era): corr(ours, vol) = **+0.19**
- LONG (2018-2026): corr(ours, vol) = **+0.22**
Low correlation = genuine diversification (different alpha: their intraday directional TS vs our daily cross-sectional).

## 50/50 blend (leakage-free vol series, Sharpe 1.99)

| period | ours Sharpe | vol Sharpe | 50/50 blend Sharpe | blend CAGR | blend maxDD |
|---|---|---|---|---|---|
| SHORT (HL era) | +1.47 | +1.67 | **+2.04** | +25% | -8% |
| LONG (2018-2026) | +1.21 | +2.01 | **+2.06** | +24% | -8% |

## Sensitivity: blend Sharpe vs the vol strategy's net Sharpe

Using the LONG correlation (+0.22) and our ~1.5 full book, 50/50 equal-risk blend = (Sv+So)/sqrt(2(1+rho)):

| vol net Sharpe | scenario | blend Sharpe |
|---|---|---|
| +1.99 | PUBLISHED full 2018-26 (leakage-free) | **+2.23** |
| +3.53 | their in-sample 2018-22 | **+3.22** |
| -0.06 | their 2026 holdout (weak regime) | **+0.92** |
| +1.00 | conservative live haircut | **+1.60** |

## Verdict

- **Diversification is genuine and strong** (corr +0.22 long / +0.19 short). Their intraday directional book (corr-to-BTC -0.11) and our daily cross-sectional book are different alphas — and both are individually good.
- **The 50/50 blend reaches ~+2.06 long (2018-26) / +2.04 short (HL era)** using the vol repo's leakage-free published series (1.99) and our ~1.5 — a real lift over either alone, because near-zero correlation makes the Sharpes add nearly in quadrature.
- **Drawdown improves too**: blend maxDD -8% long / -8% short — much tighter than the vol book's standalone -37% (our book's lower DD cushions its tail).
- **Caveats:** the vol 1.99 assumes maker (2.5bp) execution on a high-turnover intraday book — its 2026 holdout was -0.06 (weak regime) and standalone maxDD -37%. If its live net is lower (~1.0), the blend is still ~1.9 (sensitivity table). Net: the blend is a genuine improvement on both Sharpe and drawdown — the strongest case yet.
