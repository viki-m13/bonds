# HFT & Market-Making — Validated Sharpe Audit

**Central finding:** The highest Sharpe ratios in finance are **genuinely real** in this category —
but they are measured at the firm level on capital that is microscopic relative to the tech cost,
gross of the full latency arms-race spend, and structurally inaccessible without co-location and
custom hardware. **"High Sharpe that exists but cannot be scaled or accessed" is the correct mental
model here, NOT "overfit backtest."** The overfit-backtest risk lives almost entirely in the
*retail-accessible* imitations (intraday momentum, naive order-book signals).

---

## Cross-cutting caveat: the annualization problem

Nearly every >5 Sharpe in HFT is an **intraday Sharpe annualized by √252** (or finer). This is
*legitimate only if* daily P&L is approximately i.i.d. and serially uncorrelated — which for genuine
market-making (thousands of near-independent round-trips/day, flat overnight) is closer to true than
for almost any other strategy. That is precisely *why* market-making Sharpes are real: the law of large
numbers over a huge number of small, weakly-correlated bets crushes daily-return volatility, and
annualizing a tiny daily vol produces a large Sharpe. The illegitimate cases annualize a strategy with
autocorrelated returns, fat-tailed jump risk, or hidden short-gamma — there the realized tail blows up
the "Sharpe" the moment a regime breaks. (See [`06`](06-validation-methodology.md) §6, Lo 2002.)

---

## 1. Inventory-based market-making — Avellaneda–Stoikov (2008)

**Mechanism.** Post two-sided limit quotes around a "reservation price" skewed from mid as a function
of inventory and risk aversion; optimal half-spread solves an HJB equation under CARA utility with
Poisson fill intensity declining exponentially in quote distance. Seconds-to-minutes per round-trip,
thousands/day. The original paper reports **P&L and inventory-risk reduction, not a Sharpe ratio** —
a control-theory result, not a backtest. Sharpe figures attributed to "A-S" come from later
student/practitioner backtests and are not robustly comparable. Net profitability is dominated by
**adverse selection** (the model's Achilles heel — fills are assumed independent of future price, which
is false) and fee/rebate structure. Per-name capacity small; scales by breadth.
- [arXiv 1206.4810](https://arxiv.org/abs/1206.4810) · [Stanford MS&E 448](https://web.stanford.edu/class/msande448/2018/Final/Reports/gr5.pdf) · [LLMQuant summary](https://llmquant.substack.com/p/optimal-high-frequency-market-making)

**Verdict.** Credible as the canonical *framework*; any specific high Sharpe quoted for it is a backtest
artifact unless it conditions on order-flow toxicity.

## 2. Adverse-selection theory — Glosten–Milgrom (1985)

Not a tradable strategy — the *foundation* explaining why the spread exists: it equals the
adverse-selection cost of trading against possibly-informed counterparties. In a pure Glosten-Milgrom
world the MM earns **zero economic profit**. Real MMs profit because uninformed (noise) flow dominates
and they detect/avoid toxic flow faster than competitors. **A market-maker's Sharpe is a bet that it can
distinguish noise from informed flow** — a latency/signal problem, not a pricing problem.
- [Columbia abstract](https://business.columbia.edu/faculty/research/bid-ask-and-transaction-prices-specialist-market-heterogeneously-informed-traders) · [Semantic Scholar](https://www.semanticscholar.org/paper/Bid,-ask-and-transaction-prices-in-a-specialist-Glosten-Milgrom/5827ca4a5ac97e717fb5768f313079e813cebe86)

## 3. Order-book / order-flow imbalance (OFI) — Cont, Kukanov & Stoikov

**Mechanism.** OFI = cumulative signed change in best bid/ask queue sizes over a short window; a
near-linear predictor of the next mid-price change. The academic claim is **explanatory power, not
Sharpe**: OFI explains a *majority* of short-interval price changes, average **R² ≈ 65%** in US equity
(TAQ) regressions, predicting "exactly the next ~2 mid-price changes" then decaying. **R²=65% is a
contemporaneous regression statistic, NOT tradable P&L** — net of the spread you must cross and adverse
selection, a huge fraction is unmonetizable. This is the single most over-claimed retail HFT signal.
Capacity tiny (predicts the next tick, not a tradable size move).
- [arXiv 1707.01167](https://arxiv.org/pdf/1707.01167) · [EmergentMind: OFI](https://www.emergentmind.com/topics/order-flow-imbalance-ofi-7dff1686-44cf-4cf4-a602-b24df2b7c56e) · [EmergentMind: OBI](https://www.emergentmind.com/topics/order-book-imbalance-obi)

**Verdict.** Signal genuinely robust (R² real); the leap from R² to a high Sharpe is where most claims
become unjustified backtests.

## 4. Latency arbitrage (Flash Boys / IEX context) — ★ real >5

**Mechanism.** Exploit the few-microsecond gap between a price update on the fastest venue and its
propagation to slower venues / the SIP / dark pools. "Sniping" stale quotes or racing to cancel before
being picked off. The entire strategy is latency — a race measured in millionths of a second.

**Rigorous magnitude (Aquilina, Budish & O'Neill, *QJE* 2022).** Latency-arbitrage races last a
**modal 5–10 microseconds**, occur **~1/minute/symbol** (FTSE 100), are **~20% of trading volume**,
average ~half a tick each, total **~$5 billion/year globally** in equities; eliminating them would cut
the **cost of liquidity by 17%**. Top 6 firms win **>80%** of races. Dark-pool variant ("Sharks in the
Dark"): HFTs on the profitable side **96–99%** of the time, costing other traders **~2.4 bps/trade**.
These are near-pure-profit captures, but the $5bn is split among a handful of firms and is **gross of
the arms-race spend** (Budish's thesis: competition dissipates much of it into fixed technology cost).
Real, measured from exchange message data — not a backtest. Winner-take-all.
- [QJE 2022](https://academic.oup.com/qje/article/137/1/493/6368348) · [Budish summary](https://ericbudish.org/publication/quantifying-the-high-frequency-trading-arms-race/) · [Sharks in the Dark, JEDC 2024](https://www.sciencedirect.com/science/article/pii/S0165188923001926) · [SSRN 4157168](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4157168) · [IEX speed-bump](https://thehedgefundjournal.com/the-sec-approves-the-investors-exchange-speed-bump/)

**Verdict.** Highest-Sharpe activity in existence and **completely real**, but accessible to ~6 firms
globally and structurally a fixed-cost race. The signal is trivial; winning is 100% infrastructure.

## 5. Virtu Financial — "1 losing day in 1,238" — ★ real >5

A diversified global electronic market-making / latency-driven firm. The S-1 disclosed **exactly one
losing trading day out of 1,238** (~2009–2014). A ~99.92% positive-day rate with flat overnight
positioning implies an **annualized Sharpe well into the double digits (commonly cited ~20+)**. Note
the precise number is *an inference* from the disclosed day-count, not a figure Virtu published. The
mechanism: tens of millions of tiny, weakly-correlated round-trips/day make daily P&L vol minuscule,
so √252-annualization yields a huge ratio — and this annualization is *legitimate* because daily P&L is
genuinely close to i.i.d. and flat-overnight. This is **firm-level net trading income** (after fees),
audited, in an SEC filing — the gold-standard evidence that high HFT Sharpes are real. CEO Cifu later
said disclosing it "backfired" by fueling the "rigged markets" narrative. Not reproducible without
Virtu's global colocation/custom-hardware footprint.
- [Wikipedia: Virtu](https://en.wikipedia.org/wiki/Virtu_Financial) · [Fortune on the IPO](https://www.fortune.com/2015/04/06/virtu-financial-ipo) · [Virtu 10-K](https://s2.q4cdn.com/591992113/files/doc_downloads/Virtu_10K.pdf)

## 6. Empirically-measured HFT firm Sharpes — Baron, Brogaard, Hagströmer, Kirilenko — ★ real >5

**Mechanism.** Direct measurement of *actual* HFT accounts in CME E-mini S&P 500 futures (audit-trail
data), split into Aggressive / Passive / Mixed HFTs. Annualized Sharpe ratios are **very high and real**:
the dataset's HFTs average a Sharpe around **~10** (Mixed >10.4, Passive ~8.6, Aggressive ~8.5 in the
JFQA version); an earlier cut reports Aggressive HFTs at **annualized Sharpe 4.29** with 122.1%
annualized return and 90.67% alpha. These are *measured trading revenues* (effectively net of exchange
fees within the data) but do **not** net the firms' fixed tech/colocation cost. **Relative latency rank
drives performance** — firms that upgrade colocation/latency improve P&L; it is winner-take-all and new
entrants rarely displace incumbents. The capacity constraint made explicit: the pie is fixed and
speed-rank-determined.
- [NBER/working paper (PDF)](https://conference.nber.org/confer/2012/MMf12/Baron_Brogaard_Kirilenko.pdf) · [SSRN 2433118 (JFQA)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2433118) · [CityU clean PDF](https://www.cb.cityu.edu.hk/ef/doc/GRU/HFT%202017/Brogaard_HFT_risk_return_20170825.pdf) · [CFTC OCE PDF](https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_riskandreturn0414.pdf)

**Verdict.** The most credible peer-reviewed evidence that HFT Sharpes of ~8–10 are *real and measured,
not backtested* — while simultaneously proving they are non-scalable (returns to a fixed speed race,
concentrated in few firms).

## 7. Crypto market-making, triangular & funding-rate arbitrage

**Mechanism.** (a) cross-exchange/triangular arb (~0.1–0.5% per fill, seconds); (b) funding-rate
cash-and-carry (long spot + short perp, delta-neutral, hours-to-days); (c) crypto MM spread capture on
fragmented books. **Very high but data-quality-poor Sharpes:** funding-rate arb **5–10** typically, with
outliers like **23.55 (Drift)** and **6.50 (ApolloX)** vs HODL 2.89 — from medium/industry, *not* audited
track records. The headline Sharpes are **heavily gross**; net reality includes exchange/withdrawal fees,
slippage, **counterparty/exchange-solvency risk (FTX-type blowups), funding-rate sign flips, stablecoin
de-peg tail** — none captured in a Sharpe computed on smooth funding income. The smoothness *inflates*
Sharpe precisely by hiding short-gamma tail risk. Lower infrastructure barrier than equity HFT → edge
more competed, clean Sharpes less trustworthy. (Full treatment in [`05`](05-crypto-niche.md).)
- [Funding-rate arb risk/return, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2096720925000818) · [Two-tiered funding-rate markets, MDPI](https://www.mdpi.com/2227-7390/14/2/346) · [1Token](https://blog.1token.tech/crypto-fund-101-funding-fee-arbitrage-strategy/)

**Verdict.** Genuine real edge with a *moderate* true Sharpe, but the >10/>20 figures are not credible
ex-ante — they omit fat-tailed counterparty/de-peg risk and are the area most contaminated by
promotional backtests. Treat single-name Sharpes like 23.55 as overfit/marketing.

## 8. Market Intraday Momentum — Gao, Han, Li & Zhou (2018, *JFE*) — accessible, low Sharpe

**Mechanism.** The **first half-hour** SPY return (from prior close) positively predicts the **last
half-hour** return. Go long the last 30 min if the first 30 min was positive, short if negative. ~30-min
hold, once/day — **not HFT**, the retail-accessible contrast case. Timing strategy annualized **Sharpe
≈ 1.08** vs buy-and-hold 0.29; ~6.67% return, ~6.19% vol; predictive slope sig. at 1%, **R² only 1.6%**.
SPY/S&P 500, 1993–2013. Authors state outperformance **survives transaction costs** (trades infrequently),
replicated across 10+ international ETFs. **High capacity**, essentially no infrastructure.
- [SSRN 2440866](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866) · [JFE/ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351) · [Diva replication](https://www.diva-portal.org/smash/get/diva2:1878991/FULLTEXT01.pdf)

**Verdict.** Credible, robust, *low* Sharpe (~1). It demonstrates the report's thesis: **the strategies
you can access have Sharpe ~1, and the strategies with Sharpe ~10 you cannot access.**

## 9. Theoretical ceiling & "corrected Sharpe" skeptics

- **Kearns, Kulesza & Nevmyvaka, "Empirical Limitations on HFT Profitability"** ([arXiv 1007.2593](https://arxiv.org/pdf/1007.2593)):
  an omniscient/perfect-foresight trader bounds *maximum possible* HFT profit; aggregate HFT
  profitability is **far smaller than popular claims** — a hard capacity ceiling.
- **Corrected-Sharpe critiques:** when intraday HFT returns are properly adjusted (autocorrelation,
  non-normality), "efficient" Sharpe ratios fall to **~1.1 (aggressive), ~2.9 (medium), ~1.4 (passive)**,
  industry overall ~1.1–1.9 — a warning that *some* published HFT Sharpes (4.3 to absurd four-/five-figure
  values) are **misspecified annualizations**. ([arXiv 2202.11309](https://arxiv.org/pdf/2202.11309); [Elite Trader deep-dive](https://www.elitetrader.com/et/threads/realistic-sharpe-ratios-in-2026-hft-vs-retail-algos-deep-dive.388680/))

---

## Bottom-line verdicts

| Strategy | Reported Sharpe | Credible? | Scalable? | Accessible? | Real or backtest |
|---|---|---|---|---|---|
| Latency arbitrage (races) | implied very high; ~$5bn/yr global | **Yes** | No (fixed pie) | No (~6 firms) | Real (msg data) |
| Virtu firm-level | ~20+ implied (1 loss/1238 d) | **Yes** | No | No | Real (S-1) |
| Measured HFT (Baron-Brogaard-Kirilenko) | ~8–10 (4.29 aggressive cut) | **Yes** | No (returns-to-speed) | No | Real (CFTC data) |
| OFI / order-book imbalance | R²~65% (no honest Sharpe) | Signal yes, Sharpe no | No | No | Signal real; Sharpe = backtest |
| Avellaneda-Stoikov | no native Sharpe | Framework yes | No | No | Theory |
| Crypto funding-rate arb | 5–10, outliers 23.5 | Edge yes; >10 no | Partially | Yes (lower barrier) | Mostly promotional backtest |
| Intraday momentum (GHLZ) | **1.08** | **Yes** | **Yes** | **Yes** | Real, OOS |

**Synthesis.** The >5 / >10 Sharpe ratios here are genuinely real and verified by regulator-grade data
and audited filings (Virtu, Baron-Brogaard-Kirilenko, Aquilina-Budish) — NOT overfit backtests. But they
are real *only* as the property of a handful of firms winning a fixed-size, winner-take-all latency race,
on tiny strategy capital, gross of enormous fixed tech cost, and non-scalable by construction. The binding
barrier is infrastructure (colocation, FPGA, microwave links), not the signal — the signals (OFI,
imbalance) are simple and public. The genuinely *suspect* high Sharpes are (a) standalone retail
order-book-imbalance backtests and (b) crypto funding-arb's >10/>20 promotional figures, both hiding
adverse-selection / tail risk that a smooth daily-P&L Sharpe deliberately masks.

*Note: several primary PDFs (Virtu overview, GHLZ, Baron-Brogaard-Kirilenko, Kearns) returned unparseable
binary via fetch; figures were cross-confirmed through search snippets and journal/SSRN landing pages.
The exact Virtu→Sharpe number is an inference from the disclosed day-count.*
