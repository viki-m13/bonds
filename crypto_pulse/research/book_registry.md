# Book registry — the Sharpe-3 stack, honestly (HL era, net, vol-targeted)

| book | status | Sharpe | CAGR | maxDD | corr→stack |
|---|---|---|---|---|---|
| VOL | deployed | +1.67 | +26% | -11% | +0.81 |
| STRATA | ready | +1.97 | +28% | -10% | +0.68 |
| VOL+STRATA 50/50 | **recommended stack** | +2.04 | +35% | -9% | +1.00 |
| ai-ROC momentum | candidate (corr 0.50 to STRATA) | +0.93 | +13% | -14% | +0.29 |

## Distance to Sharpe 3

- Current recommended stack (VOL+STRATA 50/50): **Sharpe 2.04** full HL era.
- To reach 3.0 we need ~2 more books at ~Sharpe 1.9, corr <0.2. The ai-ROC candidate is positive but corr 0.50 to STRATA (it IS momentum), so it adds little — kept as a documented fallback, not a stack member.
- The only genuinely-orthogonal source in progress is the **L4 whale-flow book** (see flow_intraday.md / .png): sign-positive and slow, currently net-negative at tradeable speed on 29h of tape — revisit at multi-week history.
- Honest status: **2.04 now; the path to 3 is 2 more orthogonal books, and the leading candidate is data-gated, not ready.**
