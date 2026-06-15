# Stock-Picking Strategies — An Honest, Validated Literature Survey

**Scope.** A cited, adversarial survey of the empirical literature on equity-selection
strategies, filtered through a strict honesty bar: *does the edge survive out-of-sample,
survive replication scrutiny, and survive transaction costs — and is it reachable
long-only?* Compiled 2026-06 from primary sources (links inline; full list at bottom).

**Why this document exists.** The repo already has
[`literature_review.md`](literature_review.md) and
[`literature_enhancements.md`](literature_enhancements.md), which are excellent but
(a) **momentum-centric**, because SUMMIT's mandate is concentrated momentum DCA, and
(b) candidly note their citations were reconstructed *from memory* after a prior
web-research agent stalled. This survey is the broader, **properly-sourced** companion:
the whole factor zoo and beyond, every load-bearing number checked against the source.

**How to read the verdicts.** Each strategy is tagged:

- **ROBUST** — replicates under hostile methodology (value-weighted, NYSE breakpoints,
  excluding microcaps), persists out-of-sample, and survives realistic costs.
- **DECAYED** — real in-sample, materially weaker (or gone) after publication / post-2003.
- **DISPUTED** — replication or mechanism is actively contested; sign depends on method.
- **SPURIOUS / LORE** — fails the multiple-testing bar or has thin academic support.

A recurring theme: **most published "edge" lives in microcaps, illiquid names, and the
short leg** — exactly where a long-only, liquid-universe retail book (like SUMMIT's
PIT S&P 500 panel) cannot reach. Separating "true but unreachable" from "true and
implementable" is the single most important honesty filter here.

---

## Part 0 — The honesty bar: meta-evidence every strategy must clear

Before any individual factor, four bodies of evidence define what "validated" even means.

### 0.1 Out-of-sample decay — McLean & Pontiff (2016)

> **McLean & Pontiff, "Does Academic Research Destroy Stock Return Predictability?",
> *Journal of Finance* 71(1), 2016, 5–32.**
> ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365) ·
> [SSRN 2156623](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623))

Replicated **97** published cross-sectional predictors and tracked them after their
original sample windows. Findings (the canonical decay numbers):

- **~26% lower** returns out-of-sample (original-sample-end → publication date). This is
  an *upper bound* on the pure data-mining / in-sample-overfit component.
- **~58% lower** returns **post-publication**.
- The gap between the two (~32 points) is attributed to **arbitrage**: investors learn
  from the publication and trade the mispricing away. Predictors that are cheaper to
  arbitrage (high-liquidity, low-idio-vol names) decay more.

**Implication.** Halve any published premium as a baseline prior before you believe it
live. Decay is *not* full disappearance — the residual ~42% is consistent with a real
(behavioral or risk) component — but it is large and reliable.

### 0.2 The replication crisis — two camps, both worth taking seriously

**Pessimist camp:**

> **Hou, Xue & Zhang, "Replicating Anomalies," *Review of Financial Studies* 33(5),
> 2020, 2019–2133.**
> ([SSRN 3275496](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275496) ·
> [global-q PDF](https://global-q.org/uploads/1/2/2/6/122679606/houxuezhang2020rfs.pdf))

Of **452** anomalies, with microcaps controlled via **NYSE breakpoints and
value-weighted** returns, **~65% fail** to clear even a single-test `|t| > 1.96`. The
**trading-frictions** category is worst — **96% fail**. At a multiple-testing hurdle
(`t > 2.78`) the failure rate rises to **~82%**. Even survivors have economic magnitudes
"much smaller than originally reported." Their core point: *most anomalies were artifacts
of equal-weighting and microcap over-representation.*

> **Harvey, Liu & Zhu, "…and the Cross-Section of Expected Returns," *RFS* 29(1), 2016,
> 5–68.** ([Duke PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF) ·
> [SSRN 2249314](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314))

Catalogued **316 factors** across 313 papers and argued that with that much data-mining,
a new factor needs **`t > 3.0`** (not 1.96) to be credible. This is *the* multiple-testing
discipline cited throughout this document.

**Optimist camp** (the necessary counterweight — do not cherry-pick the pessimists):

> **Jensen, Kelly & Pedersen, "Is There a Replication Crisis in Finance?", *JF* 78(5),
> 2023, 2465–2518.** ([Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249) ·
> [SSRN 3774514](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514) ·
> [code](https://github.com/bkelly-lab/ReplicationCrisis))

Using a **Bayesian, multiple-testing-aware** hierarchical model and a fresh dataset of
**93 countries**, the *majority* of factors replicate, cluster into **13 themes**, and the
evidence is *strengthened* (not weakened) by the sheer number of correlated factors. They
argue HXZ's "failures" largely reflect low power against small true effects, not absence.

> **Chen & Zimmermann, "Open Source Cross-Sectional Asset Pricing," *Critical Finance
> Review* 2022** ([SSRN 3604626](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3604626)),
> and **"Publication Bias and the Cross-Section of Stock Returns"** ([SSRN 2802357](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2802357)).

Hand-reproduce **~200** predictors; for the 161 that were clearly significant in the
original papers, **98%** reproduce `|t| > 1.96`. They argue publication bias is modest and
post-publication decay (~26% OOS, echoing McLean-Pontiff) is *too small* to be explained by
pure mining — most anomalies are real, just smaller.

**Honest reconciliation.** The disagreement is mostly about **weighting and universe**, not
about whether *any* alpha exists. Where pessimists and optimists **agree**: equal-weighted,
microcap-heavy, high-turnover, short-leg-dependent anomalies are fragile; **value-weighted,
large-cap, low-turnover, theory-backed factors (value, momentum, profitability/quality,
investment, low-risk) survive both treatments.** That intersection is the trustworthy core.

### 0.3 Transaction costs — Novy-Marx & Velikov (2016)

> **Novy-Marx & Velikov, "A Taxonomy of Anomalies and Their Trading Costs," *RFS* 29(1),
> 2016, 104–147.** ([SSRN 2535173](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2535173) ·
> [NBER w20721](https://www.nber.org/papers/w20721))

- Anomalies with **< ~50% monthly turnover** generally keep a significant **net** spread;
  most **higher-turnover** ones do not.
- Execution costs ran **~20–57 bps** per trade for mid-turnover strategies (their sample).
- **Size, value, and profitability** have the **highest capacity** for new capital.
- The most effective mitigation is a **buy/hold spread** (strict entry, lenient exit) —
  which is *exactly* the structure of a never-sell DCA book.

**Real-world corroboration:** Frazzini, Israel & Moskowitz, "Trading Costs" (AQR/SSRN),
using ~$1T of live AQR trades, find real costs are *lower* than academic models for large
liquid names but that high-turnover anomalies still erode badly at scale.

### 0.4 Does stock-picking pay at all? — SPIVA, Fama-French, Bessembinder

> **SPIVA U.S. Year-End 2024** (S&P Dow Jones Indices,
> [PDF](https://www.spglobal.com/spdji/en/documents/spiva/spiva-us-year-end-2024.pdf)).
> ~**65%** of active large-cap U.S. funds underperformed the S&P 500 in 2024; over **10
> years ~85%** underperform; over **20 years ~90%+**. The ~65% 1-year figure is close to
> the 24-year SPIVA average (~64%). Persistence scorecards show top-quartile status rarely
> persists beyond chance.

> **Fama & French, "Luck versus Skill in the Cross-Section of Mutual Fund Returns," *JF*
> 65(5), 2010.** After fees, the cross-section of fund alphas looks like what you'd get from
> **luck alone** — very few managers have skill exceeding costs.

> **Bessembinder, "Do Stocks Outperform Treasury Bills?", *JFE* 129(3), 2018, 440–457.**
> ([SSRN 2900447](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2900447))
> Since 1926, **~57.8%** of CRSP stocks had lifetime buy-and-hold returns **below one-month
> T-bills**; the **best ~4%** of firms account for the **entire** net dollar wealth creation
> of the U.S. market (the rest collectively matched T-bills). Returns are **massively
> positively skewed**.

**The central tension for any stock-picker.** Bessembinder is double-edged:

- *Against* concentrated picking: if you don't hold the rare extreme winners, you
  underperform the index — and most stocks are duds. Diversification + indexing is the
  base-rate-correct default, and SPIVA shows even professionals mostly fail.
- *For* concentrated picking: *because* a few names drive everything, any signal that
  reliably tilts you **toward** the right tail (the high-skew, high-growth, momentum-leading
  mega-caps) is enormously valuable, and a **never-sell** book lets winners compound
  unbounded. SUMMIT's mega-cap-momentum tilt is, in effect, a bet on capturing this skew.

This is the honest frame for everything below: **the bar is the index, net of costs, and
the prize is the right tail.**

---

## Part 1 — The core factor zoo (the survivors and the casualties)

Rubric per factor: canonical cite → original premium → out-of-sample / replication →
net-of-cost → long-only reachability → **verdict**.

### 1.1 Value (HML, B/M, earnings yield) — **ROBUST but DECAYED; reachable long-only**

- **Cite.** Fama & French (1992, 1993); Basu (1977) for E/P.
- **Original.** HML ~0.3–0.5%/mo; the second pillar of the 3-factor model.
- **OOS / replication.** Survives HXZ value-weighting (it *is* a value-weighted factor) and
  is one of Jensen-Kelly-Pedersen's 13 themes. **But** it suffered a brutal **2017–2020**
  drawdown that reopened the "is value dead?" debate. AQR's
  [*Is (Systematic) Value Investing Dead?*](https://www.aqr.com/Insights/Research/Alternative-Thinking/Is-Systematic-Value-Investing-Dead)
  argues no — the drawdown was driven by **cheapening of cheap stocks** (valuation spread
  widening), not a vanished premium — and value rebounded sharply in 2021–2022.
- **Net of cost.** Low turnover → among the **highest-capacity** factors (Novy-Marx-Velikov).
- **Long-only.** Yes — large-cap value indices/ETFs capture most of the long-leg premium.
- **Verdict: ROBUST core factor, but premium is smaller and lumpier than 1990s papers
  implied; needs a long horizon and stomach for multi-year underperformance.** Price-to-book
  specifically has weakened as intangibles distort book value; composite value (B/M + E/P +
  CF/P + sales) travels better.

### 1.2 Size (SMB) — **DECAYED / DISPUTED standalone; useful only as a control**

- **Cite.** Banz (1981); Fama-French (1992).
- **Original.** Small-minus-big premium.
- **OOS / replication.** **Largely vanished after ~1980**, concentrated in January and in
  microcaps, and fragile to delisting bias. HXZ: weak.
- **The rehabilitation.** Asness, Frazzini, Israel, Moskowitz & Pedersen, *"Size Matters,
  If You Control Your Junk," JFE 2018* — size **re-emerges and is robust** *once you control
  for quality* (small caps are junk-heavy; clean small caps do carry a premium).
- **Long-only.** Pure small-cap tilt is *not* a reliable standalone retail edge; small-cap
  *quality* is better.
- **Verdict: DECAYED as a standalone signal; valuable mainly (a) as a control and (b)
  inside a "small + quality" interaction. For a large-cap book it is the *wrong sign* — see
  SUMMIT's mega-cap tilt, which beats cap-weighted growth benchmarks.**

### 1.3 Profitability / Quality — **ROBUST; reachable long-only; needs fundamentals**

- **Cites.** Novy-Marx, *"The Other Side of Value: The Gross Profitability Premium," JFE
  2013*; Fama-French **5-factor (2015)** adds **RMW** (profitability) and **CMA**
  (investment); Asness, Frazzini & Pedersen, **"Quality Minus Junk," *Review of Accounting
  Studies* 2019**
  ([SSRN 2312432](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432)).
- **Original.** Gross profitability has "roughly the same predictive power as B/M"
  (Novy-Marx). **QMJ** earns an **information ratio > 1** in the U.S. and across **24
  countries**, with a **negative market beta** and strong performance **in recessions/crises**
  — a genuine challenge to risk-based stories.
- **OOS / replication.** Among the **best survivors** of HXZ-style scrutiny; profitability is
  one of the few additions that improved the Fama-French model materially.
- **Net of cost.** Low turnover, high capacity (Novy-Marx-Velikov name profitability
  explicitly).
- **Long-only.** Yes for the long leg; **but requires point-in-time fundamentals**
  (gross profits, ROE, accruals) — *not available in an OHLCV-only panel.*
- **Verdict: ROBUST, arguably the most defensible "new" factor since momentum. The
  counter-cyclical (crisis-resilient) profile makes it the natural complement to momentum.
  Gated for this repo only by the missing PIT fundamentals.**

### 1.4 Investment / Asset growth — **ROBUST-ish; reachable long-only; needs fundamentals**

- **Cite.** Cooper, Gulen & Schill, *"Asset Growth and the Cross-Section of Stock Returns,"
  JF 63, 2008, 1609–1651* ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01370.x)).
- **Finding.** **High asset growth → low future returns** (over-extrapolation /
  empire-building). Robust in the original 1968–2003 sample, **extended to 2016** by the
  authors; it is the economic basis of Fama-French **CMA** and the q-factor "investment" leg.
- **Net of cost / long-only.** Low turnover; the long leg (low-investment, disciplined
  firms) is reachable but again **needs fundamentals**.
- **Verdict: ROBUST within the q-factor / FF5 framework; one of the more theory-grounded
  anomalies. Fundamentals-gated.**

### 1.5 Low-volatility / Betting-Against-Beta — **DISPUTED; mostly a microcap/leverage artifact**

- **Cites.** Baker, Bradley & Wurgler (2011); **Frazzini & Pedersen, "Betting Against
  Beta," *JFE* 2014** ([Stern PDF](https://pages.stern.nyu.edu/~lpederse/papers/BettingAgainstBeta.pdf)).
- **Original.** BAB (long low-beta levered, short high-beta) reported Sharpe ratios "above
  0.7" across many asset classes (~0.78 US equities; ~0.81 on Treasuries).
- **The damaging critique.** **Novy-Marx & Velikov, "Betting Against Betting Against Beta,"
  *JFE* 2022** ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X21002051)):
  BAB's alpha is largely an artifact of its **non-standard (rank-based) beta construction**,
  which makes it almost an **equal-weighted microcap portfolio** — *"$1.05 of every dollar
  invested goes to stocks in the bottom 1% of market cap."* Adjusted properly, much of the
  alpha is **profitability/investment exposure in disguise**, and the implementable
  (value-weighted, large-cap) version is far weaker.
- **Long-only.** The pure premium needs **leverage** (lever up low-beta) — *unavailable to
  retail long-only*. A long-only low-vol tilt exists but is a **bet against the benchmark's
  beta**: it *underperforms* in high-beta-growth-led markets.
- **Verdict: DISPUTED, and actively HARMFUL for a high-beta-benchmark mandate.** SUMMIT's own
  panel independently confirms this: low-vol / anti-lottery *selection* loses to QQQ because
  the forward winners are high-vol, high-beta names. **Do not use low-vol as a selector here.**

### 1.6 Accruals (Sloan 1996) — **DECAYED (largely gone post-2003)**

- **Cite.** Sloan, *"Do Stock Prices Fully Reflect Information in Accruals…?", TAR 1996.*
- **Original.** Low-accrual firms outperform high-accrual firms (earnings quality).
- **OOS.** **Green, Hand & Soliman, "Going, Going, Gone? The Demise of the Accruals
  Anomaly," *Management Science* 2011**
  ([SSRN 1501020](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1501020)): the hedge
  return **attenuated after ~2003–04**, coincident with **hedge-fund AUM** flowing into the
  trade. A textbook McLean-Pontiff arbitrage story.
- **Verdict: DECAYED. Of historical interest; not a live standalone edge. Fundamentals-gated
  anyway.**

### 1.7 Net share issuance / buyback, NOA — **ROBUST-ish; fundamentals-gated**

- Net issuance (firms issuing shares underperform; net repurchasers outperform) is one of
  the **better-surviving** anomalies in HXZ and overlaps the investment factor. Reachable
  long-only conceptually but **needs shares-outstanding / financing data**.
- **Verdict: ROBUST-ish, low-turnover; fundamentals-gated.**

### Core-factor scoreboard

| Factor | Verdict | Survives HXZ? | Survives cost? | Long-only? | OHLCV-only? |
|---|---|---|---|---|---|
| Momentum (see Part 3) | **ROBUST** | Yes (VW) | Yes if low-turnover | Partial | **Yes** |
| Profitability / Quality (QMJ) | **ROBUST** | Yes | Yes | Yes | No (fundamentals) |
| Value (composite) | **ROBUST/decayed** | Yes | Yes (high cap.) | Yes | No (fundamentals) |
| Investment / asset growth | **ROBUST-ish** | Yes | Yes | Yes | No (fundamentals) |
| Net issuance | **ROBUST-ish** | Yes | Yes | Yes | No (fundamentals) |
| Size (standalone) | **DECAYED** | No | — | — | partial |
| Low-vol / BAB | **DISPUTED/harmful here** | No (microcap/leverage) | Needs leverage | No (pure) | Yes |
| Accruals | **DECAYED** | Marginal | No | — | No |

---

## Part 2 — Fundamental / accounting stock-pickers

**Blanket caveat for this repo:** every signal in Part 2 needs **point-in-time fundamental
data** (financial statements aligned to the date they were public, to avoid look-ahead).
SUMMIT's panel is **OHLCV + FRED macro only**, so these are **future work** pending a PIT
fundamentals feed — flagged per item.

### 2.1 Piotroski F-score — **ROBUST in its niche; small/illiquid; needs fundamentals**

- **Cite.** Piotroski, *"Value Investing: The Use of Historical Financial Statement
  Information," JAR 2000.*
- **Original.** Within **high book-to-market** firms, a 9-point fundamental health score;
  long high-F / short low-F earned **~23%/yr** (1976–1996); high-F value beat average value
  by **~7.5%/yr**.
- **The honesty flag.** The edge is **concentrated in small, illiquid, low-share-price, low
  analyst-coverage** firms — exactly the corner that (a) HXZ down-weights and (b) is hard to
  trade net of cost. It is **not** a large-cap phenomenon.
- **Verdict: ROBUST but niche. Genuinely useful for a small-cap value book; weak for a
  liquid large-cap mandate. Fundamentals-gated.**

### 2.2 Mohanram G-score (2005) — growth-firm analog of Piotroski; **DISPUTED/weaker**, fundamentals-gated.

### 2.3 Beneish M-score (1999) — earnings-manipulation detector. Useful as a **short/avoid
red-flag screen**, not a long alpha source. **LORE-adjacent for return-prediction**;
legitimate for fraud-risk filtering. Fundamentals-gated.

### 2.4 Greenblatt "Magic Formula" (earnings yield + ROIC) — **DISPUTED / fragile**

- Independent backtests (Alpha Architect; Gray & Carlisle, *Quantitative Value*, 2012) find
  the published results **highly unstable to small methodology changes** — the ranking of
  earnings-yield vs ROIC, rebalance timing, and universe choices swing performance
  dramatically. The *quality-cheap* intuition is sound and overlaps value+profitability, but
  the specific "formula" is **over-fit marketing**, not a robust recipe.
- **Verdict: DISPUTED. The components (value + quality) are robust; the exact formula is
  fragile. Fundamentals-gated.**

### 2.5 Quality-Minus-Junk — see §1.3 (the rigorous version of "quality"). **ROBUST.**

**Part 2 takeaway.** The robust, theory-backed fundamental edges are **profitability/quality
and investment/issuance** (Part 1). The famous *recipes* (F-score, Magic Formula) are either
niche-only (F-score in small caps) or fragile (Magic Formula). **None is implementable on an
OHLCV-only panel** — all are flagged as the highest-value *data* upgrade for this repo.

---

## Part 3 — Momentum family & factor combination (the OHLCV-native survivors)

This is the part most reachable on a price/volume-only panel, and where the repo's existing
work is strongest. Summarized here with **verified citations**; deep, panel-tested detail
lives in [`literature_review.md`](literature_review.md) and
[`literature_enhancements.md`](literature_enhancements.md).

### 3.1 Cross-sectional momentum (12-1) — **ROBUST; partially reachable long-only**

- **Cite.** Jegadeesh & Titman (1993). 12-month formation, skip the most recent month,
  decile spreads historically ~**1%/mo**.
- **OOS / replication.** Among the **most robust** anomalies — survives HXZ value-weighting
  and is a Jensen-Kelly-Pedersen theme — but the **long-short** form **decayed post-2000/2010**
  and crashes violently in rebounds (see §3.2). **Long-only large-cap** retains roughly
  **half** the spread and is the workhorse of SUMMIT.
- **Net of cost.** High-ish turnover hurts long-short net returns (Novy-Marx-Velikov), **but
  a never-sell DCA book pays the half-spread once per lot** — the cost objection largely
  dissolves under buy-and-hold (the buy/hold-spread mitigation, taken to its limit).
- **Verdict: ROBUST and the single most useful OHLCV-only selector for this mandate.**

### 3.2 Momentum crashes & volatility management — **real risk; the fix is timing, not selection**

- **Cites.** Daniel & Moskowitz, *"Momentum Crashes," JFE 2016*; Barroso & Santa-Clara,
  *"Momentum Has Its Moments," JFE 2015* (vol-scaling lifts Sharpe ~**0.53 → 0.97**);
  Moreira & Muir, *"Volatility-Managed Portfolios," JF 2017*.
- **The honest caveat.** Cederburg et al., *"Do Volatility-Managed Portfolios Work?"* find
  the monthly-variance version is **fragile out-of-sample**. The **slower** (6-month realized
  vol / 200-dma regime) version is sturdier.
- **Reachability.** The scaling rules all **de-lever/short** — unavailable long-only. The
  *forecast* is usable as **deploy-timing** (route DCA lots away from panic states), which is
  exactly the §2 idea in `literature_enhancements.md`.
- **Verdict: the crash is REAL and the dominant tail risk of any momentum book. ROBUST
  finding; long-only translation = regime-aware deployment, not vol-scaled selection.**

### 3.3 Residual / idiosyncratic momentum — **ROBUST (long-short); anti-beta tax long-only**

- **Cite.** Blitz, Huij & Martens, *J. Empirical Finance* 2011 — residual momentum ~doubles
  the information ratio (~0.48 vs 0.25 monthly) and roughly **halves** crash risk.
- **Reachability caveat.** Its edge carries an **anti-beta tilt** that *fights a high-beta
  growth benchmark* (QQQ) — SUMMIT found it dilutes the edge if used always; best as a
  **transition-regime overlay**.
- **Verdict: ROBUST in long-short; conditionally useful long-only.**

### 3.4 Time-series (absolute) momentum — **ROBUST but DISPUTED magnitude**

- **Cite.** Moskowitz, Ooi & Pedersen, *"Time Series Momentum," JFE 2012* (the crash-
  protective "smile"). **Challenge:** Huang et al. (2020) *"Time-Series Momentum: Is It
  There?"* argue the pooled significance is weaker than claimed once specified carefully.
- **Long-only use:** only as a **per-name trend-intact gate** ("don't add to a name that has
  rolled over"), since the sizing/shorting kernel is unavailable.
- **Verdict: ROBUST direction, DISPUTED magnitude; usable as a gate.**

### 3.5 Factor momentum — **ROBUST and important conceptually**

- **Cites.** Ehsani & Linnainmaa, *"Factor Momentum and the Momentum Factor," JF 2022*
  ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13131)) — most factors are
  **positively autocorrelated** (avg factor earns **6 bps** after a down year vs **51 bps**
  after an up year); **factor momentum subsumes much of individual-stock momentum**. Gupta &
  Kelly, *"Factor Momentum Everywhere," JPM 2019* (AQR).
- **Verdict: ROBUST. Reframes momentum as "factors trend," which is why multi-factor timing
  via own-recent-performance has *some* (modest, contested — see §3.6) basis.**

### 3.6 Value + Momentum, and the factor-timing debate

- **Combination — ROBUST.** Asness, Moskowitz & Pedersen, *"Value and Momentum Everywhere,"
  JF 2013* ([AQR](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere)):
  value and momentum are **negatively correlated**; a 50/50 combo reaches **Sharpe ~1.45**,
  beating either alone in every market studied. This is the single most reliable
  "combine-two-cheap-things" result in the literature, and the template for SUMMIT's
  **bull-momentum / bear-quality regime switch.**
- **Timing — DISPUTED / mostly DON'T.** Asness, *"The Siren Song of Factor Timing"* (JPM
  2016): valuation-spread timing is **"deceptively difficult"** and mostly disappoints net of
  the value exposure it sneaks in. Arnott, Beck & Kalesnik counter that *some* timing
  (valuation + momentum of the factor itself) adds value. **Honest read: contrarian factor
  timing is weak; the robust "timing" is the negative value/momentum correlation doing the
  work for you in a static blend.**

**Part 3 takeaway.** Momentum (cross-sectional, OHLCV-native) is the **most defensible
implementable selector**; its **only serious enemy is the crash**, and the honest long-only
fix is **regime-aware deployment + a counter-cyclical quality sleeve** (value/quality being
negatively correlated with momentum). This is precisely SUMMIT's architecture — the
literature endorses it.

---

## Part 4 — Machine learning & the overfitting reckoning

### 4.1 Where ML alpha actually lives — Gu, Kelly & Xiu (2020)

- **Cite.** *"Empirical Asset Pricing via Machine Learning," RFS 33(5), 2020, 2223–2273*
  ([Xiu PDF](https://dachxiu.chicagobooth.edu/download/ML.pdf)). **94** stock characteristics
  × **8** macro series; trees and **neural nets** dominate linear models, roughly **doubling**
  the out-of-sample predictive performance of regression methods, with large long-short
  decile spreads.
- **The honest qualifier.** The gains **concentrate in small, illiquid, microcap** names; on
  a **value-weighted** (large-cap-dominated) basis the economic edge shrinks substantially.
  Confirmed by **Avramov, Cheng & Metzker, "Machine Learning vs. Economic Restrictions,"
  *Management Science* 2023**: excluding microcaps/distressed/hard-to-arbitrage names and
  applying realistic costs **collapses** much of ML's paper profit.
- **Reachability.** For a **liquid large-cap long-only** book, ML's incremental alpha over a
  simple momentum/quality tilt is **small and fragile**. SUMMIT's own walk-forward LightGBM
  test independently reproduced this (OOS IC ≈ 0, loses to a single momentum column).
- **Verdict: ML alpha is REAL but lives in the corners retail can't trade; for liquid
  large-caps it is mostly DECAYED/OVERFIT once costs and value-weighting are imposed.**

### 4.2 The overfitting math — Bailey & López de Prado

- **Cites.** Bailey & López de Prado, *"The Deflated Sharpe Ratio," 2014*
  ([SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)); Bailey,
  Borwein, López de Prado & Zhu, *"The Probability of Backtest Overfitting,"* 2014
  ([SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)); Harvey &
  Liu, *"Backtesting" / "Evaluating Trading Strategies."*
- **The point.** If you try **N** strategies and keep the best, its Sharpe is inflated by
  selection. The **Deflated Sharpe Ratio** discounts the observed Sharpe by the **number of
  trials, skewness, kurtosis, and sample length**; the **PBO** measures how often the
  in-sample winner underperforms out-of-sample. With enough trials, an *expected Sharpe of
  zero* still yields impressive-looking backtests.
- **Verdict (methodology, not a strategy): MANDATORY discipline.** Any signal search in this
  repo should report a **deflated** Sharpe / multiple-testing-aware hurdle, use **purged &
  embargoed** cross-validation (López de Prado, *Advances in Financial Machine Learning*), and
  treat `t > 3` (Harvey-Liu-Zhu) — not `t > 2` — as the bar. SUMMIT's
  [`RESEARCH_PROTOCOL.md`](../RESEARCH_PROTOCOL.md) random-pick control and IS/OOS window
  grid are the repo's existing implementation of this discipline.

---

## Part 5 — Niche, newer & alternative-data signals

| Signal | Cite | Verdict | Honest flag |
|---|---|---|---|
| **PEAD** (post-earnings drift) | Bernard-Thomas 1989/90 | **ROBUST but cost-eaten** | Costs absorb **70–100%** of long-short profit; value-weighted return is **0.04%/mo in liquid vs 2.43%/mo in illiquid** stocks — a microcap effect. Needs earnings data. |
| **Analyst revisions / earnings momentum** | Chan-Jegadeesh-Lakonishok | **ROBUST-ish** | Overlaps PEAD & price momentum; needs estimate data; decayed somewhat. |
| **Short interest / days-to-cover** | Boehmer-Jones-Zhang; Rapach et al. | **ROBUST (short side)** | A *negative/avoid* signal — high short interest predicts low returns. Edge is in the **short leg**; long-only use = a **veto**, not a buy. Needs short-interest data. |
| **Options-implied** (IV skew, put-call parity deviations, variance risk premium) | Xing-Zhang-Zhao; Cremers-Weinbaum | **ROBUST-ish** | Real but small; needs **options data**; capacity-limited. |
| **Insider trading (Form 4)** | Cohen-Malloy-Pomorski, "Decoding Inside Information" 2012 | **ROBUST** for *routine-vs-opportunistic* split | Needs insider-filing data; modest capacity. |
| **13F / institutional & hedge-fund clones** | Frazzini-Lamont; Gompers-Metrick | **DISPUTED** | 45-day-stale data; "smart money" cloning mostly fails to beat costs. |
| **Supply-chain / customer momentum** | Cohen & Frazzini, "Economic Links and Predictable Returns," JF 2008 | **ROBUST (slow-diffusion)** | Elegant and real; needs customer-supplier linkage data. |
| **MAX / lottery effect** | Bali-Cakici-Whitelaw 2011 | **ROBUST (negative premium)** | High max-daily-return stocks **underperform by >1%/mo**. *Avoid* lottery names — but note SUMMIT found forward mega-cap winners are themselves high-vol; use MAX as a microcap junk filter, **not** to demote high-beta leaders. |
| **Idiosyncratic-vol puzzle** | Ang-Hodrick-Xing-Zhang 2006 | **DISPUTED** | Overlaps MAX; sign sensitive to method. |
| **Turn-of-the-month / Dash-for-Cash** | Ariel 1987; Etula et al. 2020 | **ROBUST but small** | Pure **execution-timing** edge (when to deploy a lot), partially arbitraged. OHLCV-native. |
| **Halloween / Sell-in-May** | Bouman-Jacobsen 2002 | **DISPUTED** | Maberly-Pierce critique: driven by a few outliers (1987, 1998); fragile. Treat as **LORE**. |
| **Sentiment / text (news, 10-K tone, social)** | Tetlock 2007 | **DISPUTED / decaying** | Real in-sample; needs text pipeline; heavily arbitraged now. |
| **ESG as alpha** | mixed | **SPURIOUS as alpha** | After controlling for quality/profitability, ESG adds **~no** independent return; it is a *constraint/preference*, not an edge. |

**Part 5 takeaway.** Most niche signals are either (a) **short-side / avoid** signals
(short interest, MAX), (b) **data-gated** (options, insider, supply-chain, PEAD), or (c)
**execution-timing** (turn-of-month). The only **OHLCV-native, long-side-usable** items are
turn-of-month deployment and MAX-as-a-junk-filter — both small. Seasonality lore
(Halloween) and ESG-as-alpha do **not** clear the bar.

---

## Part 6 — Synthesis: what is honest, validated, and profitable

### 6.1 The intersection that survives *every* filter

A signal is trustworthy only if it survives **all four**: out-of-sample (McLean-Pontiff),
replication (HXZ value-weighted ∩ Jensen-Kelly-Pedersen), costs (Novy-Marx-Velikov), and the
implementation constraint. That intersection is small and boring — which is the point:

1. **Momentum** (cross-sectional, low-turnover, value-weighted/large-cap).
2. **Profitability / Quality (QMJ)** — counter-cyclical, the natural momentum hedge.
3. **Value** (composite, not bare P/B) — negatively correlated with momentum.
4. **Investment / asset-growth & net issuance** (the q-factor discipline).
5. **Combination over selection**: value+momentum+quality blended beats any one
   (Asness-Moskowitz-Pedersen Sharpe ~1.45), because their crashes don't coincide.

Everything else is either decayed, disputed, microcap-only, short-side-only, data-gated, or
lore.

### 6.2 Ranked for (a) long-short institutional

| Rank | Strategy | Why |
|---|---|---|
| 1 | **Multifactor blend: value + momentum + quality + investment** (q-factor / FF5 + UMD) | Highest, most durable risk-adjusted return; crashes diversify; survives HXZ. |
| 2 | **Quality / QMJ** | Best standalone survivor; crisis-resilient; high capacity. |
| 3 | **Momentum (residual-momentum variant)** | High IR, half the crash risk of plain UMD. |
| 4 | **Value (composite)** | High capacity, negatively correlated with momentum. |
| 5 | **Investment / net issuance** | Theory-backed, low turnover. |
| — | *Avoid as standalone:* BAB (microcap/leverage artifact), accruals (decayed), size (decayed), most niche/ML (corner-dwelling). |

### 6.3 Ranked for (b) long-only retail, liquid universe (this repo's reality)

Constraints: **no shorting, no leverage, OHLCV + FRED only (today), beat QQQ *and* SPY
net of cost, never-sell DCA.**

| Rank | Signal | Reachable now (OHLCV)? | Role |
|---|---|---|---|
| 1 | **Cross-sectional multi-horizon momentum** | **Yes** | Core bull selector (SUMMIT's engine). |
| 2 | **Mega-cap / dollar-volume tilt** | **Yes** | Closes the gap to cap-weighted growth benchmarks; under-published but decisive here. |
| 3 | **Regime switch (SPY 200-dma + breadth) → counter-cyclical sleeve** | **Yes** | The long-only translation of vol-managed momentum + value/quality hedge. |
| 4 | **Quality / profitability tilt (QMJ)** | **No — needs fundamentals** | Highest-value *data upgrade*; the proven momentum complement. |
| 5 | **Value (composite) counter-cyclical sleeve** | **No — needs fundamentals** | Negatively correlated hedge for the bear sleeve. |
| 6 | **Turn-of-month deployment + MAX junk-filter** | **Yes** | Small execution/quality refinements. |
| — | *Proven negatives here:* low-vol/anti-lottery **selection**, volatility-compression breakouts, pure volume signals, foundation-model re-ranking, broad ML on large caps. (See `literature_review.md`.) |

### 6.4 The three honest meta-conclusions

1. **The index is a brutal benchmark and most active selection loses to it** (SPIVA
   ~85%/10y). Any stock-picker's first job is to clear *that* bar net of cost — most don't.
2. **A few extreme winners drive everything** (Bessembinder). The defensible edge is not
   "predict every stock" but **reliably tilt toward the right tail and let it compound** — a
   never-sell, momentum-led, mega-cap book is a coherent bet on exactly this.
3. **Diversified, low-turnover, multi-factor tilts beat single-signal heroics.** The robust
   recipe is *momentum × quality × value, combined not selected, costs minimized via
   buy-and-hold, validated with deflated-Sharpe / multiple-testing discipline.* Glamorous
   niche and ML signals mostly fail to clear the honesty bar once value-weighted and
   cost-adjusted.

### 6.5 The single highest-value next step for *this* repo

Everything ranked #1–3 above is already implemented in SUMMIT. The biggest *unrealized*
edge is **#4–5: a profitability/quality + composite-value sleeve**, which is **gated only by
the absence of point-in-time fundamentals**. Acquiring a PIT fundamentals feed (gross
profits, ROE, accruals, asset growth, shares outstanding) would unlock the two most
defensible non-price factors in the entire literature — and they are precisely the
**counter-cyclical** complements that would attack SUMMIT's known weakness (momentum-crash
transition windows). See [`candidate_strategies_from_literature.md`](candidate_strategies_from_literature.md)
for harness-mapped specs.

---

## Sources

**Meta / validation**
- McLean & Pontiff (2016), *JF* — https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623
- Hou, Xue & Zhang (2020), *RFS* "Replicating Anomalies" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275496 · https://global-q.org/uploads/1/2/2/6/122679606/houxuezhang2020rfs.pdf
- Harvey, Liu & Zhu (2016), *RFS* — https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314
- Jensen, Kelly & Pedersen (2023), *JF* "Is There a Replication Crisis in Finance?" — https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514 · https://github.com/bkelly-lab/ReplicationCrisis
- Chen & Zimmermann — Open Source Cross-Section https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3604626 · Publication Bias https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2802357
- Novy-Marx & Velikov (2016), *RFS* "Taxonomy of Anomalies and Their Trading Costs" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2535173 · https://www.nber.org/papers/w20721
- SPIVA U.S. Year-End 2024 — https://www.spglobal.com/spdji/en/documents/spiva/spiva-us-year-end-2024.pdf
- Fama & French (2010), *JF* "Luck versus Skill"
- Bessembinder (2018), *JFE* "Do Stocks Outperform Treasury Bills?" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2900447

**Core factors**
- Fama & French (1992, 1993, 2015 five-factor)
- AQR, "Is (Systematic) Value Investing Dead?" — https://www.aqr.com/Insights/Research/Alternative-Thinking/Is-Systematic-Value-Investing-Dead
- Asness, Frazzini, Israel, Moskowitz & Pedersen (2018), *JFE* "Size Matters, If You Control Your Junk"
- Novy-Marx (2013), *JFE* "The Other Side of Value: Gross Profitability"
- Asness, Frazzini & Pedersen (2019), *RAS* "Quality Minus Junk" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432
- Cooper, Gulen & Schill (2008), *JF* "Asset Growth…" — https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01370.x
- Frazzini & Pedersen (2014), *JFE* "Betting Against Beta" — https://pages.stern.nyu.edu/~lpederse/papers/BettingAgainstBeta.pdf
- Novy-Marx & Velikov (2022), *JFE* "Betting Against Betting Against Beta" — https://www.sciencedirect.com/science/article/abs/pii/S0304405X21002051
- Sloan (1996), *TAR*; Green, Hand & Soliman (2011), *Mgmt Sci* "Going, Going, Gone?" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1501020

**Fundamental recipes**
- Piotroski (2000), *JAR* F-score; Mohanram (2005) G-score; Beneish (1999) M-score
- Gray & Carlisle (2012), *Quantitative Value*; Alpha Architect Magic-Formula analysis — https://alphaarchitect.com/2011/06/07/909/

**Momentum family**
- Jegadeesh & Titman (1993), *JF*
- Daniel & Moskowitz (2016), *JFE* "Momentum Crashes"; Barroso & Santa-Clara (2015), *JFE*; Moreira & Muir (2017), *JF*; Cederburg et al. "Do Volatility-Managed Portfolios Work?"
- Blitz, Huij & Martens (2011), *JEmpFin* residual momentum
- Moskowitz, Ooi & Pedersen (2012), *JFE* time-series momentum; Huang et al. (2020) critique
- Ehsani & Linnainmaa (2022), *JF* "Factor Momentum and the Momentum Factor" — https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13131 ; Gupta & Kelly (2019), *JPM* "Factor Momentum Everywhere"
- Asness, Moskowitz & Pedersen (2013), *JF* "Value and Momentum Everywhere" — https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere ; Asness (2016), *JPM* "The Siren Song of Factor Timing"

**Machine learning & overfitting**
- Gu, Kelly & Xiu (2020), *RFS* — https://dachxiu.chicagobooth.edu/download/ML.pdf · https://academic.oup.com/rfs/article/33/5/2223/5758276
- Avramov, Cheng & Metzker (2023), *Mgmt Sci* "Machine Learning vs. Economic Restrictions"
- Bailey & López de Prado (2014) "Deflated Sharpe Ratio" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 ; Bailey, Borwein, López de Prado & Zhu (2014) "Probability of Backtest Overfitting" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- López de Prado, *Advances in Financial Machine Learning* (2018)

**Niche / alternative data**
- Bernard & Thomas (1989/1990) PEAD; Cohen & Frazzini (2008), *JF* "Economic Links and Predictable Returns"
- Bali, Cakici & Whitelaw (2011), *JFE* "Maxing Out" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1262416 ; Ang, Hodrick, Xing & Zhang (2006), *JF*
- Cohen, Malloy & Pomorski (2012), *JF* "Decoding Inside Information"; Boehmer, Jones & Zhang (2008) short selling; Cremers & Weinbaum (2010) put-call parity; Xing, Zhang & Zhao (2010) IV skew
- Ariel (1987), *JFE*; Etula, Rinne, Suominen & Vaittinen (2020), *RFS* "Dash for Cash"; Bouman & Jacobsen (2002), *AER* "Halloween Indicator" + Maberly & Pierce critique; Tetlock (2007), *JF* media sentiment

*Where a URL is omitted, the work is a widely available journal article locatable by the
title/author/year given; the load-bearing quantitative claims above were verified against
the primary sources linked.*
