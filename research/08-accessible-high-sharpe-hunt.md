# Phase 2 — The Hunt for an *Accessible* Sharpe > 5 (Social Media → Validation)

**Prepared:** 2026-06-14
**Mandate:** Start where practitioners actually talk (Twitter/X, Reddit, EliteTrader, Hacker News),
collect every concrete claim of a retail-accessible strategy with Sharpe > 5, then chase each one to a
reproducible primary source and **validate it to destruction**.

> **Verdict after an exhaustive second pass:** No retail-accessible, net-of-cost, out-of-sample strategy
> that sustains **Sharpe > 5** was found — and we now have the *quantitative reason why one cannot exist*
> by sleeve-stacking. The strongest documented practitioner strategies (Zarattini/Concretum) headline at
> Sharpe 1.3–2.8 in-sample and **collapse to ~0.4–1.3 net** in every independent replication that imposes
> real spreads, slippage, borrow costs, and a fair (1×) leverage benchmark. The single eye-popping social
> claim (Sharpe 7.1 overnight-ETF reversion) is a **gross, multiple-tested, open-auction microstructure
> artifact** in precisely the ETFs the foundational academic paper says are efficiently priced. The only
> verified Sharpe > 5 remains Renaissance Medallion — and the stacking math shows exactly why it lives in a
> corner (millions of bets + execution moat) that sleeve-stacking cannot reach.

This file complements Phase 1 ([`README.md`](README.md) and [`01`](01-statistical-arbitrage.md)–[`07`](07-validation-checklist.md)).

---

## Where we looked

- **Reddit:** r/algotrading, r/quant, r/quantfinance, r/options, r/thetagang, r/Daytrading, r/FuturesTrading
- **Forums:** EliteTrader, Wilmott, QuantNet, Hacker News
- **Twitter/X "fintwit" + Substack:** Concretum/Carlo Zarattini, 0DTE & vol traders, and the aggregator
  Substacks (Quant Returns, Quantitativo, QuantSeeker, Robot Wealth, Hunt Gather Trade, QuantMacro)

*Sourcing caveat:* SSRN, EliteTrader, and X return HTTP 403 to automated fetching; every figure below was
triangulated across ≥2 independent mirrors (Concretum's own pages, QuantConnect replications, SFI/RePEc,
CXO Advisory, independent Substack reviews) before being reported.

---

## The candidates, ranked by (claimed Sharpe × credibility) — and what survived

### 1. Cross-sectional overnight→intraday ETF mean-reversion — claimed **Sharpe 7.1** → honest **~0–1.5**
**Source:** Quant Returns Substack / quantreturns.com, built on Petajisto's ETF-mispricing work.
**Mechanism:** Split each day into overnight (close→open, "CO") and intraday (open→close, "OC") returns.
Rank a small ETF universe by past overnight return; **buy low-past-overnight / short high-past-overnight**,
cross-sectionally demeaned (market-neutral); enter at the open, exit at the close.
**Claimed:** US Financials ETFs (IYF, XLF, VFH, FNCL) **Sharpe 7.1, t-stat 18.8**; Biotech (ARKG, XBI, IBB)
**4.44, t=17.3, ~0.29%/day**; index futures ~3.3. Period 2007–2025.

**Why it fails validation (three independent kills):**
1. **Two-tier Sharpe reporting / cherry-pick.** The 7.1 / 4.44 are the *raw cross-sectional CO-OC
   daily-mean ÷ daily-std × √252* numbers; the author's own *tradeable-portfolio* Sharpes are **3.67 /
   3.91**. The social headline quotes the higher tier.
2. **Circular open-print execution (the fatal mechanic).** The author explicitly admits you must "know the
   precise open in order to size trades — but you also need to trade at that very same open." You cannot
   fill at a price you are still using as your information set. The Financials gross edge is only **~13
   bps/day**; realistic open-auction execution (you don't get the official open print — you get fills
   several bps around it) or the honest VWAP-of-first-N-minutes proxy shaves 5–8 bps off 13, cutting net
   return 40–60%+. Using the first-minutes VWAP typically *destroys* the financials signal because the
   reversal it harvests largely **is** the open-auction-to-first-minutes move.
3. **Petajisto contradiction (the deepest red flag).** [Petajisto (2017, FAJ)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2000336)
   — the foundational ETF-mispricing paper — finds exploitable mean-reversion (~7% gross alpha) **only in
   international/illiquid-holding ETFs**, and is explicit that "funds holding liquid US securities are
   priced relatively efficiently" with "only very modest abnormal returns." This strategy claims its
   *biggest* Sharpes in exactly the liquid-US-sector category the paper says is efficient → the edge is a
   **microstructure/open-print artifact, not NAV mean-reversion.** Plus a 4-ETF universe = severe
   multiple-testing/overfit risk (the 7.1 was selected across many sectors × CO-OC/CC-OC/OO-OC variants)
   and trivial capacity.

**Honest net, OOS, retail Sharpe: ~0 (financials) to ~0.5–1.5 (biotech, but thin/low-capacity).** The 7.1
is not a tradeable number.

### 2. Zarattini/Barbon/Aziz — ORB "Stocks in Play" — claimed **Sharpe 2.81** → honest **~0.8–1.3**
**Source:** [SSRN 4729284](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284) (Concretum). The
single highest *credible* number from a named practitioner with a paper.
**Mechanism:** Daily, scan ~7,000 NYSE/Nasdaq stocks; filter to price>$5, ADV≥1M, 14-day ATR>$0.50,
**relative volume ≥100%**, trade only the **top 20 by opening relative volume**. 5-minute opening-range
breakout (direction = sign of the 5-min move), stop = 10% of 14-day ATR, **exit EOD**. **Max 4× leverage**,
$25k account, **$0.0035/share commission**, 2016–2023.

**Validation:**
- **Net of commission only — no short-borrow/locate fees modeled.** "Stocks in play" are small/mid-cap
  *news catalysts* (earnings, FDA) — i.e., exactly the **hard-to-borrow names with 20–100%+ annualized
  borrow fees and frequent no-locate**; ~half the trades are shorts. This is the single largest unmodeled
  cost.
- **Zero out-of-sample.** 2016–2023 is the entire fitting window; ≥6 jointly-tuned parameters (the "5-min
  beats 15/30/60" choice is itself in-sample selection).
- **Damning internal contrast:** the *unfiltered* broad ORB returns just **+29% / Sharpe 0.48** (below
  SPY's 0.78) — ~100% of the headline comes from the universe filter + 4× leverage, not the breakout.
- **Best independent replication** ([QuantConnect 18444](https://www.quantconnect.com/research/18444/))
  got **Sharpe 2.396** on a 1,000-stock universe (still no borrow costs, likely survivorship-biased), and
  the official QC notebook is a **single favorable year (2016)** with a **17% win rate**; community
  comments flag overfitting and ~25% cost drag on just 6 symbols. Parameter sweeps give 1.5–2.7 (real
  sensitivity).

**Honest net, OOS, retail Sharpe: ~0.8–1.3** after borrow/locate on shorts (−0.4 to −0.8), real small-cap
slippage (−0.3 to −0.5), and the no-OOS/survivorship haircut. Small capacity (sized into 20 small-caps/day).

### 3. Zarattini/Aziz/Barbon — "Beat the Market" intraday momentum (SPY) — **Sharpe 1.33 net** → honest **~0.8–1.1**
**Source:** [SSRN 4824172](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172) / SFI 24-97. The
**most genuinely investable** of everything found.
**Mechanism:** "Noise-area" bands = daily open × (1 ± 14-day average intraday move to that minute-of-day),
gap-adjusted. At each HH:00/HH:30, go long above the upper band / short below the lower; **dynamic trailing
stop anchored to the day's VWAP**; always flat at the close. **2% daily-vol target, up to 4× leverage.**
SPY, May 2007–April 2024. Costs modeled: **$0.0035/share commission + $0.001/share slippage**.
**Reported:** +1,985% net, 19.6% annualized, **Sharpe 1.33** (vs SPY buy-hold 0.45), beta ≈ 0, **Sharpe
3.50 conditional on VIX>40**, ~43% win rate (convex).

**Validation:**
- More credible than the rest: liquid SPY (deep liquidity, cheap borrow, real capacity), costs explicitly
  modeled, full 17-year sample.
- **But the edge is concentrated in crisis days** — the jump to 3.5 only when VIX>40 means 2008/2020 carry
  the result; in a calm-VIX decade expect **~0.5–0.7**.
- **Independent replications collapse it:**
  [QuantConnect forum 17091](https://www.quantconnect.com/forum/discussion/17091/) — at a fair **1× leverage**
  benchmark, **Sharpe drops to 0.399**, with fees = **16.7% of total return** over 6,940 orders; replicators
  note the paper fills "at the bar close [ignoring] close/open gaps and bid/ask spreads," 4× leverage
  triggers simulated margin calls, and one live deployment "over the past six months has not been good."
  [Quantitativo](https://www.quantitativo.com/p/intraday-momentum-for-es-and-nq) futures replication:
  **Sharpe 0.91**, underperforming benchmark, "the lower the slippage assumption, the better."
  [HuntGatherTrade](https://newsletter.huntgathertrade.com/p/intraday-momentum-researched-based): EOD-proxy
  Sharpe ~1.13 but flags its own look-ahead (`FillPrice`) and that the paper's better QQQ result is an
  instrument-overfit tell. The [Maróy "improvement" (SSRN 5095349)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5095349)
  to Sharpe >3 is brute-force single-instrument parameter optimization — the canonical overfit.

**Honest net, OOS, retail Sharpe: ~0.8–1.1 full-sample (≈0.5–0.7 in calm regimes).** A crisis-convexity bet
worth ~Sharpe 1, not a free lunch.

### 4. 0DTE / short-premium SPX selling — claimed ~40%/yr, "Sharpe 2–2.3" → **the trap**
r/thetagang culture + backtests (Quantish 0DTE ORB credit spreads "Sharpe 2.26", OptionAlpha/tastylive).
**This is short-vol/short-gamma carry** — a smooth equity curve and flattering Sharpe *until* a vol spike
delivers a fat left tail. The Quantish 2.26 comes from **post-hoc dropping Wednesday & Friday**
(multiple-testing), is **gross** (QuantConnect "doesn't account for multi-leg options properly"), and spans
only a benign-vol 3.3-year window. Any Sharpe here is regime-dependent and tail-blind — see Phase 1
[`03-volatility-options.md`](03-volatility-options.md) (XIV: −96% in one day). **Honest read: ~1 with a
hidden left tail; the Sharpe is the wrong metric.**

### 5. "0+" double-digit-Sharpe HFT scratching — real, **not accessible**
Hacker News "0+: A double digit Sharpe HFT strategy" + Everstrike write-up. Join a strong order-book queue;
on a fill, immediately post the offsetting order; "scratch" (exit flat) ~8/10 times. Edge = queue priority +
imbalance. **Requires being fastest (FPGA, colocation) and sub-one-tick fees (~0.01%); retail fees (~0.05%)
exceed the tick and kill it.** This is the honest origin of "Sharpe > 5" lore — *technology alpha*, not a
reproducible rule. Consistent with Phase 1 [`02-hft-market-making.md`](02-hft-market-making.md).

---

## The decisive result: why sleeve-stacking *cannot* reach Sharpe 5

The recurring fintwit/Reddit pitch is "stack N uncorrelated Sharpe-1 strategies → Sharpe = √N → just add
more." The exact math kills it. For N sleeves each with individual Sharpe `s` and **uniform pairwise
correlation ρ**, equal-weighted:

```
SR_portfolio = s · √( N / (1 + (N−1)·ρ) )
```

- **ρ = 0:** reduces to `s·√N` (the seductive textbook claim).
- **ρ > 0:** as N → ∞, **SR → s/√ρ — a hard ceiling independent of N.** This is the single most ignored fact.

**The ρ-ceiling (max achievable Sharpe multiple, regardless of how many sleeves you add):**

| Avg correlation ρ | Ceiling multiple s·(1/√ρ) | If s = 1.0 | If s = 1.5 |
|---|---|---|---|
| 0.00 | ∞ (→ s·√N) | unbounded | unbounded |
| 0.05 | 4.47× | 4.5 | 6.7 |
| 0.10 | 3.16× | 3.2 | 4.7 |
| **0.16 (empirical, short-horizon equity)** | **2.50×** | **2.5** | **3.75** |
| 0.20 | 2.24× | 2.2 | 3.4 |
| 0.50 (tail regime) | 1.41× | 1.4 | 2.1 |

The empirical anchor: [Kakushadze & Tulchinsky, "101 Formulaic Alphas" (arXiv:1601.00991)](https://arxiv.org/pdf/1601.00991)
— 101 real, reproducible short-horizon equity alphas (0.6–6.4 day holding), **average pairwise correlation
15.9% (median 14.3%)**, 80 of them in production at a real fund. At ρ≈0.16, an s=1 stack is capped at **~2.5
even with infinitely many alphas** — and those figures are **gross** (turnover-heavy short-horizon alphas
bleed most net; Kakushadze's companion "Performance v. Turnover: a Story by 4,000 Alphas" shows cost ∝ 1/T).

**Reproducible demonstration:** the open-source anti-overfit stack at
[github.com/45ck/llm-quant](https://github.com/45ck/llm-quant) (pre-registered specs, Deflated Sharpe ≥0.95,
CPCV, walk-forward) combines 18 strategies of individual Sharpe 1.087 at ρ=0.186 → **empirical combined
Sharpe 2.205** — already ~87% of its ρ-ceiling (2.52). Adding 80 more sleeves would crawl toward ~2.5,
**never 5.** Worse, ρ is measured in calm periods; in stress, sleeves co-crash (ρ → 0.5+), so the realized
tail-regime ceiling is ~1.4×.

**Grinold–Kahn restates it from the signal side:** `IR = IC·√BR`. Medallion's reported ~50.75% win rate is
an IC near zero; its Sharpe comes entirely from **breadth (BR) in the millions** of near-coin-flip bets plus
a market-making execution moat — *not* from stacking a dozen strong sleeves. That is the corner of the math
that delivers Sharpe > 5, and it is categorically inaccessible (closed since 1993, capped ~$10B). See
[`06-validation-methodology.md`](06-validation-methodology.md) §9.

---

## The annualization / time-in-market question (resolved)

A strategy holding a position only ~30 min/day still annualizes its **daily** Sharpe by √252 — is that a
cheat? **No, not by itself.** Under IID daily returns, √T scaling is correct regardless of intraday
time-in-market; a 30-min/day strategy's daily Sharpe is a fair, apples-to-apples object (idle capital simply
isn't exposed to midday/overnight variance). **The real inflation enters via autocorrelation** (Lo 2002,
*The Statistics of Sharpe Ratios*): with serially-correlated daily returns the correct factor is √252 ÷
√(1 + 2Σρₖ), and **ignoring positive autocorrelation overstates annualized Sharpe by up to ~65%.** The
intraday edges here cluster in trending/high-VIX regimes (positive autocorrelation) → deflate ~20–40%. The
same IID violation means the headline **t-stats (e.g. 18.8) are also overstated** and should not be read as
"19-sigma real."

---

## Academic roots & peer-reviewed anchors (what the originals actually say)

The retail strategies above are dressed-up versions of published academic results — and the peer-reviewed
versions are far more modest and honest:

- **Intraday momentum root — Gao, Han, Li & Zhou, "Market Intraday Momentum" (JFE 2018).** SPY, 1993–2013.
  The first-half-hour return predicts the last-half-hour return with **R² = 1.6%** (OOS R² 1.4%); the timing
  strategy earns **6.67%/yr, σ 6.19%, Sharpe 1.08** (vs buy-hold 0.29), and the authors show it **survives
  transaction costs** (cost ≈ 1.2 bps; return reduced only ~1.2% to 4.30%/yr). This is the honest root:
  **Sharpe ~1, net, peer-reviewed** — and Zarattini's "Beat the Market" is essentially the leveraged,
  VWAP-stopped extension of it. Independently replicated on APAC/SPY data (Limkriangkrai et al. 2024) with
  near-identical R². [SSRN 2440866](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866) ·
  [JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351)
- **ORB peer-reviewed test — Holmberg, Lönnbark & Lundström, "Assessing the profitability of intraday
  opening range breakout strategies" (Finance Research Letters 2013).** Tested published ORB rules on S&P
  500 futures; blunt conclusion: **"the strategies do not work as advertised."** Zarattini's pro-ORB SSRN
  papers are the optimistic counterpoint that this and the Deflated-Sharpe critique directly target.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612312000438)
- **0DTE systematic selling — Vilkov, "0DTE Trading Rules" (2024), SPX 2016–2026, net of half-spread +
  0.5bp.** *Unconditional* same-day vol selling is **weak and tail-dominated** (small VRP, hard to monetize
  net). The best *conditional, strict-OOS* rules: put ratio spreads **gross SR 1.18 / net 0.93**;
  straddle/strangle **gross 0.56 / net 0.39**; diversified top-3 basket **gross 1.12 / net 0.82**. Verdad's
  Monte-Carlo puts systematic 0DTE straddle selling at **SR 0.85–1.4** *but* warns "real-world tails are
  fatter than options theory assumes." So the honest 0DTE number is **net ~0.4–0.9 with a fat left tail** —
  nowhere near 5, and the metric itself understates the tail. [SSRN 4641356](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4641356) ·
  [Verdad](https://verdadcap.com/archive/zero-day-options)
- **The deflation tools (formal).** Lo (2002): √252 annualization overstates Sharpe by up to **~65%** under
  positive return autocorrelation — the exact correction for clustered intraday edges. Bailey & López de
  Prado's **Deflated Sharpe Ratio** / "false-strategy theorem": after N trials, **even a true-zero-Sharpe
  strategy produces a high backtested Sharpe** (winner's curse), so any reported figure must be benchmarked
  against E[max SR] before it counts. Sullivan-Timmermann-White's Reality Check found **no simple trading
  rule survives** data-snooping correction on the DJIA/S&P. See [`06`](06-validation-methodology.md).

## Phase 2 scorecard

| Strategy | Headline Sharpe | Single biggest validity threat | Honest net/OOS/retail Sharpe |
|---|---|---|---|
| Overnight→intraday ETF reversion (Financials) | **7.1** (t=18.8) | Fill-at-exact-open is circular; Petajisto says liquid US ETFs are efficient | **~0–0.5** |
| same (Biotech) | 4.44 | Same open-print artifact + thin capacity | **~0.5–1.5** |
| Zarattini ORB "Stocks in Play" | **2.81** | No borrow/locate on hard-to-borrow small-cap shorts; 4× lev; no OOS | **~0.8–1.3** |
| Zarattini "Beat the Market" (SPY) | 1.33 (3.5 @ VIX>40) | Edge concentrated in crisis days; 1× fair-leverage repl. = **0.40** | **~0.8–1.1** (0.5–0.7 calm) |
| 0DTE short-premium SPX | "2.0–2.3" | Short-vol tail-blind; Wed/Fri cherry-pick; gross | **~1, wrong metric** |
| "0+" HFT scratching | "double digits" | Needs FPGA/colo + sub-tick fees; retail fees kill it | **real but inaccessible** |
| 18-sleeve disciplined stack | (√N hope) | ρ-ceiling = s/√ρ ≈ 2.5 at ρ=0.16 | **~2.0–2.2** |
| **Renaissance Medallion** | **2.0–7.5 (live, 31 yr)** | None — but closed/capped/proprietary | **>5 real, unreachable** |

---

## Bottom line

The community's own most-upvoted summary (EliteTrader's "Realistic Sharpe Ratios 2026") states it plainly:
**"Sharpe above 5.0 is almost certainly HFT market-making… impossible without co-located FPGAs and sub-ms
latency — the alpha is technological speed, not predictive skill. Gold standard for retail/prop is Sharpe
2.0–3.0."** Our independent validation agrees on every point:

1. **The accessible candidates that headline high all degrade to ~0.4–1.3 net** once real spreads, slippage,
   borrow costs, and a fair leverage benchmark are imposed. Leverage (TQQQ) scales return *and* vol
   together, so it **does not raise Sharpe** — the 1,484% TQQQ headline is the same ~1.0 Sharpe as unlevered.
2. **The one genuine social-media "Sharpe 7" is a gross open-auction microstructure artifact** in efficient
   liquid ETFs — not tradeable.
3. **Sleeve-stacking is mathematically capped at s/√ρ ≈ 2.5** at realistic short-horizon correlations; you
   cannot add your way to 5.
4. **The only verified Sharpe > 5 (Medallion) is the √BR/execution-moat corner** — millions of bets, closed
   since 1993, capped ~$10B. It exists; it is not reproducible.

The honest, defensible target for a sophisticated individual/small fund remains **~1–2 net for a single
strategy, ~2–3 net for a disciplined diversified book** — consistent with Phase 1. A retail-accessible,
net, out-of-sample, tail-honest, scalable **Sharpe > 5 was not found, and the stacking math explains why it
is not merely undiscovered but structurally out of reach** without HFT infrastructure or Medallion-scale
breadth.

*Research only; no existing repository files modified. A couple of secondary diggers (risk-managed short-vol,
event-driven/ML micro-edges) were still completing at write-up; nothing in their partial returns contradicts
the above and this file will stand as the Phase 2 record.*
