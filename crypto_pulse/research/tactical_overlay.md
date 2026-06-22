# VOL-core with tactical STRATA overlay (preserve CAGR, cushion slumps)

Both vol-targeted to 15%. Overlay rotates into STRATA only when VOL's own realised performance weakens (causal), then re-targets to keep risk constant. Compared to pure VOL and a static 50/50.

| book | CAGR | Sharpe | maxDD | CAGR (2025-26) | Sharpe (2025-26) | avg STRATA wt |
|---|---|---|---|---|---|---|
| Pure VOL | +38% | **+2.02** | -14% | +14% | +0.87 | 0% |
| Static 50/50 | +51% | **+2.40** | -14% | +48% | +2.25 | 50% |
| Tactical: DD>10% -> +STRATA | +39% | **+1.93** | -16% | +12% | +0.75 | 0% |
| Tactical: lowSharpe -> +STRATA | +43% | **+2.09** | -14% | +18% | +1.10 | 9% |
| Tactical: continuous tilt | +43% | **+2.07** | -14% | +19% | +1.15 | 10% |

## Static VOL/STRATA mixes (vol-targeted, no timing)

| VOL % | STRATA % | CAGR | Sharpe | maxDD | CAGR 25-26 |
|---|---|---|---|---|---|
| 100 | 0 | +40% | **+1.96** | -15% | +12% |
| 80 | 20 | +46% | **+2.19** | -15% | +24% |
| 70 | 30 | +49% | **+2.30** | -15% | +31% |
| 65 | 35 | +50% | **+2.35** | -15% | +36% |
| 50 | 50 | +51% | **+2.40** | -14% | +48% |
| 35 | 65 | +49% | **+2.30** | -14% | +57% |

## Verdict

- **Pure VOL:** CAGR +38%, Sharpe +2.02, maxDD -14%.
- **Adding STRATA RAISES VOL's CAGR, it does not dilute it.** At equal risk (vol-targeted), higher Sharpe => higher CAGR, so every VOL-tilted mix beats pure VOL on BOTH axes: even a VOL-dominant 80/20 lifts CAGR +38%->+46% and Sharpe +2.02->+2.19, same drawdown. The worry about giving up VOL's CAGR is not borne out.
- **Tactical timing does NOT beat a static blend.** The best slump-timer (lowSharpe -> +STRATA) only reaches Sharpe +2.09 because it sits ~90%+ in VOL — by the time VOL's drawdown/low-Sharpe fires, the move is late and whipsaws. A plain static 70/30 (Sharpe ~2.30) dominates it. Continuous diversification > trying to time the slump.
- **Recommendation to PROVE VOL while improving the book:** a VOL-dominant *static* blend (65-80% VOL / rest STRATA). VOL stays the core engine and proof; STRATA is always-on ballast that lifts Sharpe ~2.0->2.3 and CAGR ~40%->~50% at the same -15% DD, and specifically rescues VOL's 2025-26 slump. No market-timing required. The L4 whale-flow book, once it has history, would be the next always-on diversifier to add on top.
