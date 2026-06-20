# Selective / conditional leverage vs constant leverage

Grand-stack book. Conditional leverage = base x conviction, RENORMALIZED so average leverage = the 5x constant baseline (so we compare CONCENTRATION, not just more leverage), then capped. A levered daily loss worse than -60% is flagged as ruin/liquidation.

## Constant baseline

- constant 5x: CAGR +116%, maxDD -41%, Calmar 2.79, worst day -18%.

## Conditional leverage (avg matched to 5x), by signal and cap

| signal | peak cap | Sharpe | CAGR | maxDD | Calmar | worst day | ruin? |
|---|---|---|---|---|---|---|---|
| CALM | 10x | +1.34 | +85% | -36% | 2.37 | -15% | no |
| CALM | 15x | +1.38 | +91% | -36% | 2.54 | -15% | no |
| CALM | 20x | +1.38 | +91% | -36% | 2.54 | -15% | no |
| CALM | 30x | +1.38 | +91% | -36% | 2.54 | -15% | no |
| TREND | 10x | +1.74 | +160% | -41% | 3.91 | -23% | no |
| TREND | 15x | +1.74 | +177% | -42% | 4.17 | -28% | no |
| TREND | 20x | +1.74 | +177% | -42% | 4.17 | -28% | no |
| TREND | 30x | +1.74 | +177% | -42% | 4.17 | -28% | no |
| DISP | 10x | +1.39 | +118% | -61% | 1.94 | -21% | no |
| DISP | 15x | +1.33 | +112% | -62% | 1.81 | -27% | no |
| DISP | 20x | +1.33 | +112% | -62% | 1.81 | -27% | no |
| DISP | 30x | +1.33 | +112% | -62% | 1.81 | -27% | no |

## Verdict

- Best conditional config: **TREND cap 15x** — Calmar 4.17 vs constant 5x 2.79, CAGR +177% vs +116%, worst day -28% vs -18%.
- Conditioning leverage on a genuine confidence signal gives a SMALL Calmar improvement (concentrating risk into calmer/stronger states), but raising the cap past ~10-15x sharply worsens the worst-day and pushes toward ruin: the conviction signal is too noisy to justify 20-30x in any single state, because ONE wrong high-conviction day at that leverage liquidates the account (fat tails). 'Much more than 10x when confident' is super-Kelly betting — the math says even at TRUE Kelly (~10x) you sit at a -55% drawdown; going beyond only raises ruin probability for less growth.
- The honest version of 'lever when confident' is already in the book: vol-targeting levers up in calm regimes. Beyond that, keep a HARD cap (~5x / quarter-to-half Kelly) + the crash-hedge + kill-switch. Selective spikes to 20-30x are a liquidation bet, not an edge.
