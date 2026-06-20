# Highest-CAGR leverage (Kelly) on the grand-stack book

Book = validated 6-sleeve grand stack, net of real funding + 4.5bps taker, base vol-targeted to ~12% (Sharpe 1.46). Sample 2023-06-26->2026-04-24 (2.8y). Leverage L = multiple of the 12% book; net P&L scales ~linearly with L (gross, fees, funding all scale).

## CAGR vs leverage (the Kelly curve)

| leverage | ~ann vol | Sharpe | **CAGR** | maxDD | worst day | x in 2.8y |
|---|---|---|---|---|---|---|
| 1.0x | 14% | +1.46 | **+21%** | -9% | -4% | 1.7x |
| 2.0x | 27% | +1.46 | **+44%** | -18% | -7% | 2.8x |
| 3.0x | 41% | +1.46 | **+67%** | -26% | -11% | 4.3x |
| 5.0x | 68% | +1.46 | **+116%** | -41% | -18% | 8.8x |
| 5.5x | 75% | +1.46 | **+127%** | -45% | -20% | 10.2x  <- half-Kelly |
| 8.0x | 109% | +1.46 | **+176%** | -61% | -29% | 17.7x |
| 10.0x | 137% | +1.46 | **+197%** | -71% | -36% | 21.9x |
| 11.0x | 150% | +1.46 | **+200%** | -76% | -39% | 22.6x  <- MAX CAGR |
| 14.0x | 191% | +1.46 | **+176%** | -87% | -50% | 17.8x |
| 20.0x | 273% | +1.46 | **+32%** | -98% | -72% | 2.2x |

- **Max CAGR is at ~11.0x leverage** (≈150% vol): CAGR **+200%**, but maxDD **-76%** and worst day **-39%**. Beyond it, CAGR falls (vol drag).
- **Liquidation reality:** the book's worst day is -3.6% at 1x; at ~**28x** a single day like that ≈ −100% = account wiped. Full-Kelly 11x sits dangerously close — on HL, margin/liquidation would trigger well before −100%, and fat tails make realized Kelly LOWER than the in-sample optimum. **Full Kelly is not survivable live.**
- **Half-Kelly (~5.5x)** keeps most of the growth — CAGR **+127%** — at far lower risk (maxDD -45%, worst day -20%). This is the practical aggressive pick.
- Sharpe stays ~1.5 at every leverage (the ratio is leverage-invariant); leverage buys CAGR up to Kelly, then destroys it. The number that improved was never the Sharpe — only the risk you take to convert it into return.
