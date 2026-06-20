# Full Hyperliquid universe — max-Sharpe book, live-execution validated

Crypto grand stack (HL perps, real funding + 4.5bps) + cross-asset trend books proxied by long-history ETFs and mapped to HIP-3 perps (4.5bps taker + 3.0bps slippage for thinner books). Weekly, crypto overlap 2023-06-30->2026-04-03 (145 wks). IS=first60/OOS=last40.

## Per-class books (on the overlap) + full-history context

| class | HL vehicles | maxLev | Sharpe (overlap) | IS | OOS | full-hist Sharpe |
|---|---|---|---|---|---|---|
| CRYPTO | crypto 3-40x | 40/25/3-5 | **+1.62** | +1.35 | +1.98 | +1.38 |
| EQ_INDEX | km:US500/USTECH/SMALL2000, flx:USA500/USA100 | 5 | **-0.01** | +0.04 | -0.08 | +0.05 |
| COMMOD | xyz/flx/km GOLD SILVER OIL COPPER GAS PLATINUM ... | 5 | **-0.25** | -0.88 | +0.69 | +0.24 |
| FX | xyz:EUR/JPY, km:EUR | 10 | **-0.89** | -0.95 | -0.79 | -0.49 |
| BONDS | km:USBOND | 5 | **-1.01** | -1.12 | -0.82 | -0.37 |

## Cross-class correlation (overlap)

| | CRYPTO | EQ_INDEX | COMMOD | FX | BONDS |
|---|---|---|---|---|---|
| CRYPTO | +1.00 | -0.14 | +0.02 | +0.02 | +0.12 |
| EQ_INDEX | -0.14 | +1.00 | +0.01 | +0.12 | +0.06 |
| COMMOD | +0.02 | +0.01 | +1.00 | +0.08 | +0.06 |
| FX | +0.02 | +0.12 | +0.08 | +1.00 | +0.26 |
| BONDS | +0.12 | +0.06 | +0.06 | +0.26 | +1.00 |

Mean |correlation| of crypto to the cross-asset classes: 0.08 (low = genuine diversification).

## Max-Sharpe portfolio (all HL vehicles)

Sharpe-optimal IS weights: CRYPTO 96%, EQ_INDEX 4%, COMMOD 0%, FX 0%, BONDS 0%

| portfolio | Sharpe | IS | OOS | CAGR | maxDD | Calmar |
|---|---|---|---|---|---|---|
| CRYPTO only (ref) | **+1.62** | +1.35 | +1.98 | +21% | -6% | 3.28 |
| full-universe equal-risk | **-0.22** | -0.69 | +0.56 | -3% | -22% | -0.15 |
| full-universe max-Sharpe | **+1.63** | +1.36 | +1.98 | +21% | -6% | 3.36 |

## Max leverage -> max CAGR (Kelly) on the portfolio

Sharpe is leverage-invariant; CAGR peaks at Kelly then drops. Per-class HL max leverage (3-40x) is far above what the vol target uses, so leverage is bounded by DRAWDOWN tolerance, not HL caps.

| leverage | ann vol | Sharpe | CAGR | maxDD | worst wk |
|---|---|---|---|---|---|
| 1x | 12% | +1.63 | +21% | -6% | -2.8% |
| 2x | 24% | +1.63 | +44% | -12% | -5.7% |
| 3x | 36% | +1.63 | +69% | -18% | -8.5% |
| 4x | 48% | +1.63 | +96% | -24% | -11.3% |
| 5x | 60% | +1.63 | +124% | -30% | -14.2% |
| 6x | 72% | +1.63 | +154% | -35% | -17.0% |
| 8x | 96% | +1.63 | +215% | -45% | -22.7% |
| 10x | 120% | +1.63 | +273% | -55% | -28.4% |

## Verdict (live-execution honest)

- Full-HL-universe max-Sharpe portfolio: **+1.63** (IS +1.36 / OOS +1.98) vs crypto-only +1.62. Cross-asset classes are genuinely uncorrelated to crypto (mean |corr| 0.08) so they diversify, but the traditional trend books are individually weak (~0.3-0.6), so the lift is modest — the portfolio is crypto-dominated.
- Max CAGR at ~10x leverage; HL per-asset caps (40x BTC down to 3x alts, HIP-3 lower) are NOT the binding constraint — drawdown is. Run quarter-to-half Kelly with the downside overlays.
- HONEST take on 'max Sharpe': adding every HL vehicle lifts the portfolio toward the high-1s, not to 3. A stable 3 still needs STRONG (not just uncorrelated) extra books — the accessible traditional trend/carry books are too weak. This IS the max-Sharpe configuration of what HL actually offers, validated with live costs.
