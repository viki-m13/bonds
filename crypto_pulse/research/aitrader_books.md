# Do ai-trader's strategies add a book toward Sharpe 3? (honest test)

ai-trader = classic TA + long-only rotation. We implement its rotation logic as deployable market-neutral cross-sectional HL-perp books (long top / short bottom, inverse-vol, net 4.5bps + funding, vol-targeted to 12%), TEXTBOOK params (no tuning). HL era; OOS = last 40%.

**Existing stack VOL+STRATA 50/50:** Sharpe 2.11 (full HL), 1.64 (OOS).

A new book helps the stack only if its Sharpe > corr*stack_Sharpe (marginal rule).

| book | Sharpe | OOS | corr VOL | corr STRATA | corr stack | adds to stack? | stack+book OOS |
|---|---|---|---|---|---|---|---|
| ai:ROC-rot (momentum) | +0.93 | +1.29 | +0.05 | +0.50 | +0.28 | no | +1.40 |
| ai:RSI-rot (reversal) | -1.65 | -1.11 | -0.07 | -0.46 | -0.29 | no | +0.54 |
| ai:Bollinger-rot (breakout) | +1.03 | +0.46 | +0.08 | +0.42 | +0.28 | no | +1.11 |
| ai:RSRS-rot (support slope) | +0.07 | -0.04 | +0.08 | +0.16 | +0.14 | no | +1.08 |
| ai:TripleRSI-rot (reversal) | -1.79 | -1.32 | -0.08 | -0.46 | -0.30 | no | +0.39 |

## The standalone ai-trader strategy (its best ideas, ensembled)

Equal-risk blend of ai-trader's positive cross-sectional books (momentum + breakout + support-slope; reversal variants excluded by sign). Market-neutral, net of costs+funding, vol-targeted. This is the honest best the toolkit yields:

- **Sharpe +0.63** (full HL), **+0.47** (OOS). CAGR +8%, maxDD -13%.
- Correlation to VOL +0.06, to STRATA +0.48.
- This is a **real, positive, tradeable strategy — but it is ~0.6, not 3.** Classic technical analysis on liquid crypto, net of costs, does not produce a Sharpe-3 standalone book; it never has in honest OOS testing. Anyone claiming a TA indicator at Sharpe 3 is either gross of costs, in-sample, or overfit.


## Honest stack assembly

- **None of the ai-trader rotation books clears the marginal-Sharpe bar.** Their signals overlap STRATA's existing trend/reversal sleeves and/or are too weak net of costs. The honest stack stays VOL+STRATA at Sharpe 1.64 OOS — adding ai-trader books does not move it toward 3.

## What Sharpe 3 honestly requires

From the current stack (1.64 OOS, rho~0.17 between legs), reaching 3.0 needs ~2 more books each ~Sharpe 1.9 at corr <0.2. ai-trader's classic TA does not supply them (see table). The credible remaining sources of a genuinely-new book are: (a) the L4 per-account whale-flow book now being data-collected (different signal entirely — order flow, not price), and (b) a cross-asset leg (equity/FX), which diversifies but cannot be run on HL. Honesty: Sharpe 3 net OOS is at the frontier; it is reached by stacking, not by any single indicator in this or any TA library.
