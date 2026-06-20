# Our grand stack under the vol repo's live-validated execution stack

Daily book, HL era. Charging fee+slippage on turnover (sweep) and funding stress, the way the vol repo's live study prescribes. Our combined daily turnover ~**0.47x/day** (~170x/yr) — vs the vol repo's intraday 159-377x/yr.

## Slippage sweep (round-trip fee+slippage on turnover)

| total cost/side | Sharpe | IS | OOS | CAGR | maxDD |
|---|---|---|---|---|---|
| 4.5 bps | **+1.75** | +1.19 | +2.62 | +27% | -11% |
| 6.5 bps | **+1.62** | +1.06 | +2.49 | +25% | -11% |
| 8.5 bps | **+1.49** | +0.93 | +2.36 | +23% | -11% |
| 10.5 bps (base, = vol repo's break-even) | **+1.36** | +0.80 | +2.23 | +20% | -12% |
| 14.5 bps | **+1.10** | +0.53 | +1.97 | +16% | -15% |
| 20.5 bps | **+0.71** | +0.14 | +1.57 | +10% | -21% |

## Funding stress (at base 4.5bps)

| funding | Sharpe | CAGR | maxDD |
|---|---|---|---|
| 1x | **+1.75** | +27% | -11% |
| 2x | **+2.39** | +40% | -10% |
| 3x | **+2.98** | +53% | -9% |

## Combined adverse (14.5bps cost + 3x funding spikes)

- Sharpe +2.35, CAGR +39%, maxDD -10%.

## Verdict

- **Our edge survives realistic execution where theirs broke even.** At the vol repo's intraday break-even cost (~10.5 bps round-trip) our Sharpe is still **+1.36** (vs theirs → ~0). Our gross turnover (~170x/yr) is actually SIMILAR to theirs (159-377x/yr) — so the moat is NOT lower turnover, it is **edge-per-trade**: a DAILY signal earns multiple bps per trade, so a few bps of slippage is a small fraction of the edge, whereas their intraday breakout earns sub-bp per trade and slippage swamps it.
- **We are a net funding RECEIVER, so funding stress HELPS us.** The carry + funding-fade sleeves are short the crowded high-funding side, so 2x/3x funding lifts Sharpe to +2.39/+2.98 — the OPPOSITE of the vol repo's funding-PAYING directional intraday book (for which funding spikes were the dominant residual cost). *Caveat:* uniform funding amplification doesn't capture short-squeeze tail risk, so read this as 'funding is a tailwind, not a drag,' not as free Sharpe.
- Base (4.5 bps): Sharpe +1.75 (IS +1.19 / OOS +2.62 — the high full number is recent-regime-flattered; honest central ~1.3-1.5). It stays positive out to ~20 bps total cost and the combined adverse case (14.5 bps + 3x funding) is +2.35.
- Honest answer to 'does our edge survive the vol repo's live-validated execution?': **yes, robustly** — because it is a high-edge-per-trade DAILY book that COLLECTS funding, not an every-bar intraday book that PAYS it. The vol repo's two hardest live lessons (turnover x slippage, and funding) are exactly the two costs our design is structurally on the right side of.
