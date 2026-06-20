# Diversified Multi-Sleeve Portfolio — Assembly & Honest Verdict

Script: dca/research/exp105_portfolio.py. The requested final deliverable: combine
the validated long-only sleeves with risk-parity weights into one allocation.

## Sleeves (2015-2025, PIT survivorship-clean)
| Sleeve | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|
| WAVE (ML momentum-quality picker) | 21.4% | 1.34 | -18% |
| QQQ | 18.5% | 1.03 | -33% |
| Cross-asset Trend (dual-momentum) | 7.3% | 0.71 | -22% |
| Compounder (durable-survivor composite) | 1.5% | 0.17 | -43% |

## Correlation matrix — the problem
All pairwise correlations 0.48-0.63 (WAVE-QQQ 0.63, Trend-QQQ 0.63, Comp-QQQ 0.61).
The long-only sleeves are ALL long US risk assets => they co-move; not the 0.3-0.4
needed for strong diversification.

## Blends — none beats QQQ or WAVE-alone
| Blend | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|
| risk-parity (rolling inv-vol) | 11.8% | 0.99 | -24% |
| equal-weight | 12.3% | 0.98 | -25% |
| no-QQQ (WAVE+Comp+Trend) | 10.2% | 0.92 | -22% |
| + vol-target 14% (lev<=2) | 14.5% | 0.95 | -27% |
DCA $1k/mo: blend $275k vs QQQ $415k.

## HONEST VERDICT
The diversified long-only blend does NOT beat QQQ or WAVE-alone. Reasons:
(1) sleeves too correlated (~0.55) — all long equity/risk; (2) risk-parity
over-weights the low-return defensive sleeves (Trend/Compounder), dragging CAGR;
(3) the Compounder sleeve was weak 2015-25; (4) QQQ 2015-25 (Sharpe 1.03) is a
brutal benchmark. => **WAVE alone (21.4%/1.34/-18%) is the deployable champion**,
beating QQQ on all three; blending in the weaker correlated sleeves HURTS.
The genuine path to portfolio Sharpe 1.5+ requires an UNCORRELATED sleeve = the
market-neutral factor/ML L/S (corr ~0 to QQQ, net Sharpe ~1.0-1.8) — but that
needs shorting+leverage (SUMMIT, parked, outside the long-only mandate).

## Final recommendation
- Long-only, no margin (your mandate): DEPLOY WAVE (or, for lower drawdown at the
  cost of return, WAVE 80% + Trend 20% — modest DD reduction, lower CAGR).
- If shorting/leverage ever allowed: add SUMMIT market-neutral overlay — that is
  the only thing that genuinely lifts portfolio Sharpe via true diversification.
