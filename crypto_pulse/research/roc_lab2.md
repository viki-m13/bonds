# ROC lab iter-3: momentum + reversion two-bucket book (honest)

Diversified momentum bucket (7 signals) + reversion bucket (3 signals), risk-parity combined (IS weights 0.50/0.50). Net 4.5bps+funding, vol-targeted. HL era, OOS=last40%.

| book | Sharpe (HL) | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| Momentum bucket | +0.86 | +0.93 | +0.75 | +11% | -12% |
| Reversion bucket | -1.86 | -2.12 | -1.46 | -22% | -56% |
| Risk-parity combo | -0.99 | -0.97 | -1.03 | -13% | -38% |

- Momentum/Reversion correlation (HL era): **-0.60** (correlated, limited diversification).
- Combo deflated OOS Sharpe (22 cumulative trials): **-1.03**, P(SR>0)=0.00 (does NOT clear 95% bar).

## Honest verdict

- Two-bucket combo OOS Sharpe **-1.03**, deflated **-1.03**. Sharpe 3 NOT reached.
- Even with two buckets the deflated number stays well under 3. 
- This is iteration 3; the price ceiling holds. Continuing per request, but each new trial lowers the deflated bar — the honest number is reported, not the lucky max.
