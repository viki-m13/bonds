# FLOW — daily order-flow book from crypto_of (honest, OOS)

Taker-buy imbalance, 35 coins 2021-01-01..2026-06-18, net 4.5bps+funding, vol-targeted. follow vs fade chosen on IS (first60%), OOS=last40%.

| signal | dir | Sharpe | IS | OOS | robust (both>0.2)? |
|---|---|---|---|---|---|
| flow-level | follow | +0.11 | -0.12 | +0.76 | no |
| flow-level | fade | -0.38 | -0.24 | -0.85 | no |
| flow-z | follow | -0.72 | -0.94 | +nan | no |
| flow-z | fade | +0.43 | +0.56 | +nan | no |
| flow-momentum | follow | -0.17 | -0.63 | +1.08 | no |
| flow-momentum | fade | -0.06 | +0.32 | -1.12 | no |

## Best flow book (IS-picked): flow-z fade

- Standalone Sharpe full +0.43 (IS +0.56 / OOS +nan), CAGR +4%, maxDD -24%.

## TIDE + FLOW combo (HL era)

- Correlation TIDE vs FLOW: **+0.00**.
- TIDE +2.01, FLOW -0.01, **risk-parity combo +1.89** (-0.12 vs better leg).
- Sharpe 3 NOT reached; combined ~1.9.

## Verdict (honest)

- **0 of 6 flow configs are positive in BOTH IS and OOS.** The daily aggregate taker-flow signal is unstable: OOS winners (flow-momentum follow) are IS-losers and vice versa — regime luck, not a stable edge. Aggregate imbalance is mostly noise at daily frequency.
- **Genuinely orthogonal but too weak:** TIDE-FLOW correlation +0.00 (truly uncorrelated), but FLOW's ~0 Sharpe means the combo +1.89 DILUTES TIDE alone (+2.01). Same lesson as EBB: diversification needs a strong second leg.
- **The honest cut:** aggregate daily flow doesn't help. If order flow has alpha it is in PER-ACCOUNT granularity (informed-wallet isolation), which is what the L4 tape captures — but that needs the multi-week recording still in progress, not this aggregate daily series. TIDE alone (~2.0) remains the answer.
