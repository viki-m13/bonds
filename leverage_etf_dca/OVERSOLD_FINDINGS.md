# "Buy oversold ETFs to beat QQQ-DCA everywhere, including dot-com" — tested, honest verdict

Per the request to *"dynamically buy ETFs (gold, sectors, leveraged + normal) when
undersold so we outperform DCA-into-QQQ in every period, even dot-com."* This is
**mean-reversion (buy the dip)** — the opposite of the momentum rotation already killed
in `ROTATION_FINDINGS.md`. It is a genuinely distinct hypothesis, so it got a genuine test.
Scripts: `scripts/oversold.py` (rotation engine) and `scripts/regime.py` (the synthesis).

## What is real (and it IS real)
Buying **oversold-but-still-in-a-200d-uptrend** assets, diversified across the base menu,
with a cash exit when nothing qualifies, **beats QQQ-DCA during busts and roughly halves
the drawdown**:

| bust window | best oversold sleeve | ratio vs QQQ-DCA | strat maxDD | QQQ-DCA maxDD |
|---|---|--:|--:|--:|
| dot-com 1999–2003 | below-50d-MA, base menu | 1.11× | −36% | −81% |
| dot-com 1999–2006 | dip-in-uptrend, base | 1.20× | −43% | −81% |

That is a **real risk-management effect** — diversified dip-buying survives crashes far
better than concentrated leveraged (or even plain) tech beta.

## Why it is NOT a deployable "beat everywhere" strategy
The **same** oversold rotation over the modern tech decade is a catastrophe:

| signal (base menu, K3) | 2010–2026 continuous ratio vs QQQ-DCA |
|---|--:|
| dip_in_uptrend | 0.06× |
| below_ma50_uptrend | 0.08× |
| pure_reversal | 0.18× |

In a tech-dominated decade, continually rotating *out* of the compounding winner into
"whatever's cheap" is a permanent drag. The per-era wins are isolated windows that each
restart the DCA from zero; the honest **continuous** run loses 0.06–0.45×.

## The synthesis, and why it also fails
Natural fix: a **trend gate** — run vol-targeted leveraged NASDAQ when QQQ is above its
200d MA (capture the tech decade), switch to the diversified-oversold/cash sleeve when it
breaks (survive the bust). Per era it looks like a winner: **1.26× dot-com, 1.17× 2000–10,
1.24× GFC, 1.05–1.27× most eras, with half the crash drawdown.**

It dies on **phase-robustness** — the full 1999–2026 continuous ratio by rebalance day:

| rebalance on | ratio vs QQQ-DCA | maxDD |
|---|--:|--:|
| month-end | 1.08× | −44% |
| day 4 | 0.93× | −58% |
| day 9 | 0.98× | −56% |
| day 14 | 0.44× | −60% |

A 0.44×→1.08× swing on *which day you trade* is timing-luck, not signal — the exact
signature that killed the trend-switch and momentum-rotation strategies. The continuous
full-period ratio is a **tie** (0.99–1.08×), and the 46 regime switches are trading noise.

## The honest bottom line
Tested four independent ways (stock selection, leveraged momentum rotation, oversold/value
rotation, regime-gated synthesis): **no robust, non-overfit strategy substantially beats
QQQ-DCA in every period including dot-com, because the requirement is self-contradictory.**
Beating QQQ-DCA in the dot-com bust requires being *diversified and defensive*; beating it
in 2010–2021 requires being *concentrated in leveraged tech*. Those are opposite bets and
no causal signal separates the regimes cleanly enough to survive phase-robustness.

- **QQQ-DCA is itself a recency-biased, concentrated bet on tech leadership** — it "wins
  every period" only because the sample *is* the tech era. That cuts both ways.
- **VOLT** maximizes that same tech bet with vol-managed leverage: wins the modern sample,
  and in a bust **protects** (drawdown ~half) rather than out-returns. It is honest
  risk-managed leverage, **not** alpha.
- The **only** durable, phase-robust edge found here is defensive: diversified
  oversold-in-uptrend buying cuts crash drawdowns roughly in half in every bust.
