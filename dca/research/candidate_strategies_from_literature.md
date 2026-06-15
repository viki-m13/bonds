# Candidate Strategy Specs — mapped to the SUMMIT harness

Companion to [`stock_picking_literature_survey.md`](stock_picking_literature_survey.md).
Each spec turns a **survivor** from the survey into something runnable on *this* repo's
infrastructure, with an honest expectation and a pass/fail bar. These are **deliberately
non-overlapping** with the momentum-internal ideas already in
[`literature_enhancements.md`](literature_enhancements.md) (FIP, panic-deploy, turn-of-month,
52-week-high gate, conviction sizing, residual-momentum overlay) — see that file for those.

## Harness contract (recap)

```python
import data, protocol
P = data.build_panel()                    # open/high/low/close/volume/member
                                          # ~5647 days x 725 tickers, PIT S&P 500 mask
scores = ...                              # DataFrame dates x tickers; row d uses info
                                          # through CLOSE of day d only (exec next open)
card = protocol.evaluate_signal(scores, "name", k=3, every=10, cost_bps=5, sell=None)
# headline: card["overall"]["win_qqq"], ["win_spy"], ["med_vs_qqq"], ["worst_vs_qqq"]
```

**Causality (non-negotiable):** cross-sectional ranks/z-scores within row `d` are fine; no
full-sample stats, no `shift(-1)`, ML fit walk-forward only. **Pass bar** (from
`RESEARCH_PROTOCOL.md`): `win_qqq ≥ 85%` with `med_vs_qqq` clearly positive, **and** it must
beat the `random_control` (survivorship check). For *tail* ideas, the real target is
**improving `worst_vs_qqq` without lowering the median.**

---

# Section A — OHLCV-native (testable today, no new data)

## A1. Long-term reversal as a price-only "value/cheapness" sleeve — the bear-sleeve, formalized

**Survey basis.** Value is a top survivor (§1.1) and is **negatively correlated with
momentum** (Asness-Moskowitz-Pedersen 2013, Sharpe ~1.45 combined). True value needs
fundamentals (Section B), but the **DeBondt-Thaler long-term reversal** (3–5yr losers beat
3–5yr winners) is a *price-only* cheapness proxy and is the academic cousin of SUMMIT's
existing "deep drawdown-from-ATH among healthy names" bear sleeve. This spec *formalizes and
literature-anchors* that sleeve so it can be tested as a clean signal rather than lore.

**Signal (causal, OHLCV-only).** For each name in `member` on day `d`:
```
ltr_d   = - (close_d / close_{d-756} - 1)        # 3y reversal: cheap = down a lot
health  = close_d > rolling_mean(close, 200)_d   # long-term trend intact (avoid value traps)
cheap_d = close_d / rolling_max(close, 1260)_d    # distance below 5y high (lower = cheaper)
score   = zscore_xs(ltr_d)  where health, else -inf
```
Use **only inside the bear regime** (`SPY < SPY_200dma`), matching the survey's finding that
the discounted-quality precursor only works below the 200-dma (`literature_review.md` §8).

**Harness mapping.** Build `scores` = bull-momentum sleeve on green days, `ltr`/`cheap`
sleeve on red days (the existing regime switch), and A/B it vs the current bear-sleeve
formulation via `evaluate_signal(..., name="ltr_bear_sleeve")`.

**Honest expectation / risk.** Long-term reversal has **decayed in large caps** and tilts
low-momentum — so as an *always-on* selector it will lose to QQQ (consistent with the file's
existing negatives). It should *only* help as the **bear-regime sleeve**, buying healthy
names at a discount for the rebound. **Pass bar: improve `worst_vs_qqq` (the 2010–2013 /
post-crash transition windows) without lowering `med_vs_qqq`.**

## A2. Static value-proxy × momentum blend — test the regime switch's reason for existing

**Survey basis.** The most reliable combination result in the literature is value+momentum
**blended, not selected** (AMP 2013), because their crashes don't coincide. SUMMIT
implements this as a *regime switch*; the literature's default is a *static blend*. Worth
knowing which wins on this panel.

**Signal.** `blend_d = w·zscore_xs(momentum_d) + (1-w)·zscore_xs(ltr_d·health)`, swept over
`w ∈ {0.6, 0.7, 0.8, 0.9}`, **always on** (no regime gate).

**Harness mapping.** `evaluate_signal(blend_scores, f"valmom_w{w}", k=3)` across the sweep;
compare `worst_vs_qqq` and `med_vs_qqq` against both the pure-momentum baseline (66% win_qqq)
and the regime-switch SUMMIT.

**Honest expectation.** Likely **worse median** than the regime switch (the static value tilt
drags in bull markets where QQQ-beta dominates) but possibly **better worst-window**. The
honest question this answers: *is SUMMIT's regime switch actually beating a dumb static blend,
or just adding complexity?* A negative result here is still valuable (it validates the switch).

## A3. MAX / lottery **junk-veto** (not a selector)

**Survey basis.** Bali-Cakici-Whitelaw (2011): high max-daily-return stocks underperform by
**>1%/mo** — a robust *negative* premium. **Critical nuance:** `literature_review.md` proved
that demoting high-vol/high-beta *leaders* hurts here (forward winners ARE high-vol). So MAX
must be used **only** to veto **non-mega-cap lottery junk**, never to demote mega-cap leaders.

**Signal.** `max5_d = max(daily_return over last 21d)`. Veto a candidate iff
`max5_d > 99th xs-pctile` **AND** `dollar_volume_d < median` (i.e., a small, jumpy name) —
demote it below the next mega-cap momentum name. Never applies to the top dollar-volume decile.

**Harness mapping.** Apply as a score penalty on the momentum top-N, then `evaluate_signal`.
Expect a **near-zero to tiny positive** effect (the mega-cap tilt already excludes most
lottery junk) — its value is *insurance against the universe-broadening case*, not alpha.

**Honest expectation.** Marginal. Include only if it doesn't lower the median. This is the
*correctly-scoped* version of the MAX/anti-lottery idea that the file otherwise (rightly)
lists as a negative.

---

# Section B — Fundamentals-gated (the survey's #1 unrealized edge)

> **This is the headline recommendation of the whole research effort.** The two most
> defensible *non-price* factors in the entire literature — **profitability/quality (QMJ)**
> and **composite value** — are **counter-cyclical** (negative market beta, strong in
> recessions; §1.3, §6) and are therefore the *ideal complements* to SUMMIT's pro-cyclical
> momentum engine. They are blocked **only** by the absence of point-in-time fundamentals on
> the panel. Acquiring that data unlocks more defensible edge than any further price-only
> signal mining.

## B1. Quality-Minus-Junk sleeve (Asness-Frazzini-Pedersen 2019)

**Data needed (PIT, point-in-time / as-first-reported, lagged ≥ the real filing date):**
gross profits, total assets, ROE/ROA, gross margin, accruals, leverage, earnings
variability, shares outstanding. Sources to evaluate: SEC EDGAR financial-statement datasets
(free, PIT-correct if aligned to filing dates), Sharadar/SF1 (paid, PIT), or
WRDS/Compustat-PIT (institutional).

**Signal.** Composite QMJ z-score within `member`: `quality = zscore(profitability) +
zscore(growth) + zscore(safety) - zscore(accruals)`. Use as the **bear-regime sleeve**
(replace/augment A1) and/or as a tie-breaker within momentum winners in bulls.

**Why it's the right complement.** QMJ has a **negative market beta** and **performs in
crises** — precisely SUMMIT's weak windows (momentum-crash transitions). Long-only reachable
(long leg only), low turnover (high capacity, cheap under buy-and-hold).

**Pass bar.** Improve `worst_vs_qqq` materially; hold or improve median. Validate the quality
sleeve beats the price-only A1 bear sleeve.

## B2. Composite value sleeve (not bare P/B)

**Data needed:** book equity, earnings, cash flow, sales (PIT). **Signal.** `value =
mean(zscore(B/M), zscore(E/P), zscore(CF/P), zscore(S/P))` — composite travels better than
P/B (which intangibles distort; §1.1). Use blended with momentum (the AMP static blend, now
with *real* value) and/or as the bear sleeve.

## B3. Investment / net-issuance discipline (Cooper-Gulen-Schill; q-factor)

**Data needed:** total assets (for asset growth), shares outstanding (for net issuance).
**Signal.** Penalize high asset-growth and net share *issuers*; favor net *repurchasers*.
Low-turnover, robust, long-only. Use as a **veto/tilt** on the momentum book (avoid
empire-builders that are momentum-hot but fundamentally diluting).

---

# Section C — Methodology upgrade (applies to all future signal searches)

## C1. Deflated-Sharpe / multiple-testing discipline in the protocol

**Survey basis.** Part 4.2 — with enough trials, a zero-edge strategy produces great
backtests. This repo has already tried **many** signals (FIP, compression, volume, ML,
chronos, MAX, low-vol, residual mom, cadence, ETF universes…). That trial count must be
*paid for* statistically.

**Concrete addition to `protocol.py`:**
- Track a **trials counter** (number of distinct signals/parameterizations evaluated).
- Report a **Deflated Sharpe Ratio** (Bailey-López de Prado 2014) on the win-rate /
  vs-benchmark series, discounting for the number of trials, skew, and kurtosis.
- Keep the **`random_control`** (already present) as the empirical false-discovery floor.
- Adopt **`t > 3`** (Harvey-Liu-Zhu), not `t > 2`, as the significance bar for any new
  always-on selector, given the search intensity.

This doesn't add return — it protects against *believing* a lucky one. Highest-leverage,
zero-data change.

---

# Priority order to backtest

| # | Spec | Data | Targets | Expected verdict (honest prior) |
|---|------|------|---------|-------------------------------|
| 1 | **C1 deflated-Sharpe / trials discipline** | none | trust | Should-do regardless; cheap, protects all conclusions. |
| 2 | **B1 QMJ quality bear-sleeve** (needs B-data) | PIT fundamentals | worst-window | Highest expected edge; the survey's #1 unrealized complement. |
| 3 | **A1 long-term-reversal bear-sleeve (formalized)** | none | worst-window | Modest; literature-anchors the existing bear sleeve; price-only stand-in for B1/B2. |
| 4 | **B2 composite value + AMP static blend** | PIT fundamentals | worst-window | Real value finally testable; counter-cyclical hedge. |
| 5 | **A2 static value-proxy × momentum blend** | none | diagnostic | Likely validates the regime switch (negative result still useful). |
| 6 | **B3 investment/issuance veto** | shares/assets | tail/quality | Low-turnover quality guard on the momentum book. |
| 7 | **A3 MAX junk-veto** | none | insurance | Marginal; only adopt if median unharmed. |

**Cross-cutting honesty caveat.** Every published effect cited here is **historical
full-sample** and decays out-of-sample (~26–58%, McLean-Pontiff). Adopt a candidate only if
it improves the **worst-3y window** without lowering the **median**, beats the
**random-pick control**, and clears the **deflated-Sharpe / `t>3`** bar on this PIT panel.
The point of this whole exercise is *not* to find more signals — it is to find the few that
are honest enough to survive contact with the out-of-sample.
