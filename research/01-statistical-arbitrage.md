# Statistical Arbitrage & Mean-Reversion — Validated Survey

**Headline:** Across the literature, *credible, net-of-cost, large-capacity* Sharpe ratios for
classic stat-arb cluster around **0.8–1.5**, not above 2. Every Sharpe found above ~2 is
attributable to one or more of: (a) gross-of-cost reporting, (b) tiny-capacity high-frequency
niches, (c) short non-stationary sample windows, (d) inclusion of illiquid small-caps that
cannot actually be traded at the assumed cost. The classic anomalies (pairs, short-term
reversal, index effect) have all measurably **decayed since publication**, several to
statistical insignificance.

---

## 1. Pairs Trading — Gatev, Goetzmann & Rouwenhorst (2006)

**Mechanism.** Form "distance" pairs: over a 12-month formation window, normalize each stock's
cumulative total-return price index; pick the partner minimizing the sum of squared price
deviations. Trade over the following 6-month period: open the spread when it diverges by
**2 historical standard deviations** (long the loser, short the winner), close at convergence
(or period end). Top-5 and top-20 pair portfolios, monthly rebalanced into new cohorts.

**Reported performance.** Average **monthly excess return ~0.81% on a committed-capital basis**
for the top-20 pairs → ~11.16% annualized; Sharpe ~1.22 on the canonical implementation.
Period **1962–2002**, CRSP daily, liquid US common stocks.
- [NBER w7032](https://www.nber.org/papers/w7032) · [GGR PDF](http://stat.wharton.upenn.edu/~steele/Courses/434/434Context/PairsTrading/PairsTradingGGR.pdf) · [Quantpedia: CAGR 11.16%, Sharpe 1.22, vol 5.85%, MaxDD −17%](https://quantpedia.com/strategies/pairs-trading-with-stocks)

**Gross vs net.** GGR's headline figures are stated as surviving "conservative transaction-cost
estimates" — a one-way cost proxy. The committed-capital Sharpe ~1.22 is effectively net of
their assumed costs but does **not** include realistic short-borrow costs or market impact at scale.

**Decay (the defining weakness).** Do & Faff (2010, 2012) extended through June 2008 and confirmed
a "continuation of the declining trend in profitability." After commissions, market impact and
short fees, net risk-adjusted return falls to **~30 bps/month** among well-matched within-industry
pairs — and pairs trading is "**largely unprofitable after 2002**." Strongest in the 1970s–80s,
decayed through the 1990s, brief revivals only in 2000–02 and 2007–09 dislocations.
- [Do & Faff 2012, J. Financial Research](https://onlinelibrary.wiley.com/doi/10.1111/j.1475-6803.2012.01317.x) · [determinants paper](https://www.sciencedirect.com/science/article/abs/pii/S1386418114000809)

**Verdict.** ✅ Credible as a historical phenomenon; ❌ **largely dead net-of-cost since ~2002.**
Sharpe ~1.0–1.2 is honest for the in-sample era. GGR's cost proxy is lighter than modern realistic
frictions (bid-ask + borrow); convergence assumes you can always borrow the winner to short.

---

## 2. Cointegration / PCA Stat-Arb — Avellaneda & Lee (2010)

**Mechanism.** Decompose stock returns into systematic + idiosyncratic via either (a) **PCA** on
the correlation matrix (~15 eigenportfolios) or (b) regression on **sector ETFs**. Model the
residual as **Ornstein–Uhlenbeck mean-reverting**; compute a dimensionless **"s-score"** = distance
from equilibrium in OU std-dev units. Trade contrarian: open when |s-score| crosses ~1.25, close
near zero. Daily signals, market/sector-neutral, **leverage = $2 long / $2 short per $1 equity**.

**Reported Sharpe (net of cost, 1997–2007).**
- **PCA-based: Sharpe 1.44.** ETF-based: **1.1.**
- Decay built into their own results: **2003–2007 PCA Sharpe fell to ~0.9.** Adding daily volume
  ("trading time" vs calendar time) lifted the **ETF strategy to ~1.51 for 2003–2007** — their best
  post-decay number, still below 2.
- [Quantitative Finance 10(7):761–782](https://www.tandfonline.com/doi/abs/10.1080/14697680903124632) · [Berkeley PDF](https://traders.studentorg.berkeley.edu/papers/Statistical%20arbitrage%20in%20the%20US%20equities%20market.pdf) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505)

**Gross vs net.** The 1.44 / 1.1 figures are **after** an assumed slippage/cost charge — which is
exactly why they sit at a believable 1–1.5, not 5. Independent replications (Stanford MS&E) found
the edge worked 2003–2008 then stagnated, attributing decline to crowding.
- [Stanford replication](http://stanford.edu/class/msande448/2017/Final/Reports/gr5.pdf)

**Verdict.** ✅ **Most credible high-quality stat-arb reference.** Sharpe ceiling firmly ~1.5,
decaying to ~0.9 unlevered post-2003. No data-snooping red flags; authors document the decay
themselves. The 2×/2× leverage is essential to reach ~1.4 — unlevered signal Sharpe is lower.

---

## 3. High-Frequency / Intraday Mean Reversion — ⚠️ where the >5 numbers live

**Claim A — Intraday pairs, doubly-mean-reverting OU on oil-company HF data: Sharpe 3.9 and 7.2.**
- 🚩 **FLAG.** Sharpe 7.2 is for **2008 alone** (extreme-vol year); 3.9 for Jun 2013–Apr 2015.
  A handful of oil-company pairs (microscopic capacity), single-year window for the 7.2 figure,
  intraday HF with an optimistic fill model. **Gross-leaning, ultra-low-capacity, sample-specific.
  Not a deployable >5 Sharpe.**
- [Liu, Chang et al., *Quantitative Finance* 17(1):87–100](https://www.tandfonline.com/doi/abs/10.1080/14697688.2016.1184304) · [IDEAS](https://ideas.repec.org/a/taf/quantf/v17y2017i1p87-100.html)

**Claim B — Bowen, Hutchinson & O'Sullivan (2010), FTSE-100 HF pairs, 2007 (the honest counterweight).**
- Excess returns are "extremely sensitive to transaction costs and speed of execution." Just
  **15 bps round-trip cuts excess returns by >50%**, and most profit accrues in the first hour and
  to the fastest executors. **The HF pairs edge is real gross but collapses net unless you are a
  low-latency liquidity provider.**
- [SSRN 1611623](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1611623) · [Semantic Scholar](https://www.semanticscholar.org/paper/972d1b2d384ec6b7547e19bb8ded85410e524c28)

**Claim C — Overnight→intraday reversal ("CO-OC"), Sharpe 4.44.** Buy lowest past overnight
(close-to-open) returns, sell highest, hold intraday; ~0.29% avg daily return, t-stat 17.3.
- 🚩 **FLAG.** **Gross**, daily-rebalanced, **enormous turnover** (flip the entire book twice daily).
  The t-stat reflects signal robustness, not net tradability. Daily round-trip costs annihilate a
  29 bps/day gross edge for anyone paying retail/institutional spreads.
- [Liu, Liu, Wang, Zhou, Zhu, "Overnight-Intraday Reversal Everywhere," SSRN 2730304](https://papers.ssrn.com/sol3/Delivery.cfm/2730304.pdf?abstractid=2730304) · [Della Corte & Kosowski version](https://www.cicfconf.org/sites/default/files/paper_357.pdf)

**Overall HF mean-reversion verdict.** The >2 (and >5) Sharpes are **real but gross, capacity-tiny,
execution-dependent** — essentially *liquidity-provision* returns earned only if you ARE the market
maker. Chan's cited "Sharpe 4.8 gross → 3.5 after 10 bps round-trip" is itself a low-cost assumption;
realistic institutional costs and impact at any scale push these toward 1 or below.
- [epchan](http://epchan.blogspot.com/2008/12/enduring-profitability-of-mean.html)

---

## 4. ETF Arbitrage & Index-Inclusion Effects — DECAYED TO NEAR-ZERO

**Mechanism.** Buy a stock between S&P 500 addition announcement and effective date (forced
index-fund demand → temporary price pop), reverse afterward; symmetric for deletions.

**Effect size and its collapse.** Addition abnormal return: **+3.4% (early 1980s) → ~+7–8.8% peak
(late 1990s/2000) → +5.2% (2000s) → ~+1.0% and statistically INSIGNIFICANT in the 2010s.** Drivers
of disappearance: more index "migrations" from S&P MidCap (already priced), vastly improved
liquidity provision around rebalances, predictability/front-running of changes.
- [Greenwood & Sammon, "The Disappearing Index Effect," NBER w30748](https://www.nber.org/system/files/working_papers/w30748/w30748.pdf) · [Petajisto 2011, JEF](https://www.petajisto.net/papers/petajisto%202011%20jef%20-%20hidden%20cost%20for%20index%20funds.pdf) · [Bennett, Stulz, Wang](https://www.ecgi.global/sites/default/files/working_papers/documents/bennettstulzwangfinal.pdf)

**Verdict.** ✅ Was a real, large effect; ❌ **effectively dead as a standalone arb** — no Sharpe
worth quoting today. ETF creation/redemption (premium/discount) arbitrage persists but is an
**authorized-participant, near-zero-margin, infrastructure-gated** business, not accessible alpha.
APs strategically omit illiquid names ([JFQA, ETF Sampling and Index Arbitrage](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/etf-sampling-and-index-arbitrage/EE6BA16F9C54C1E01DD726FF23796FC7)).

---

## 5. Ornstein–Uhlenbeck Mean-Reversion Modeling (methodology layer)

**Mechanism.** Model a tradable spread/residual as OU: `dX = θ(μ−X)dt + σ dW`. Estimate θ, μ, σ by
MLE; derive **optimal entry/exit/stop-loss thresholds** via optimal double-stopping that maximizes
expected discounted P&L net of transaction costs (Leung & Li framework). Used *inside* pairs/stat-arb.

**Reported Sharpes.** Realistic portfolio implementations: **~0.8** (e.g., 0.815 vs SPY's 0.612 over
5 yrs, QuantConnect). Cherry-picked "best pairs": ~1.9–2.4 in- AND out-of-sample (CCI/HCP: 2.326
in-sample, 2.425 OOS).
- [Leung & Li, *Optimal Mean Reversion Trading*](https://www.worldscientific.com/doi/10.1142/9789814725927_0002) · [Huang & Martin, arXiv 1602.05858](https://arxiv.org/pdf/1602.05858) · [arXiv 1411.5062 (costs + stop-loss)](https://arxiv.org/pdf/1411.5062) · [QuantConnect optimal pairs](https://www.quantconnect.com/research/15294/optimal-pairs-trading/)

**Verdict.** ⚠️ The ~2.3 OOS numbers are **survivorship/selection artifacts** — the *best 9 pairs*
chosen from a search. Honest portfolio-level OU Sharpe is **~0.8–1.0**. OU framing improves entry/exit
discipline (and explicitly incorporates costs/stop-loss), but does not manufacture a >2 portfolio
Sharpe. Treat any single-pair >2 OOS claim as data-snooping.

---

## 6. Lead-Lag / Cross-Sectional Short-Term Reversal

**Mechanism.** Lehmann (1990) / Lo & MacKinlay (1990) contrarian: each week, long past losers /
short past winners, dollar-neutral; profits from short-horizon overreaction + liquidity provision.

**Gross results & the cost reality.** Lehmann/Lo-MacKinlay found large, significant weekly contrarian
profits. Quantpedia/De Groot-Huij-Zhou large-cap implementation: **CAGR 16.25%, Sharpe ~1.09 net
(1990–2009), but MaxDD −52.9%.** BUT standard reversal net of costs is "**indistinguishable from zero
or even negative**" for any universe including small/illiquid caps (Avramov, Chordia, Goyal) — the
largest reversals live in the most expensive-to-trade names. It survives ONLY when (a) restricted to
large-caps, (b) turnover-limited, (c) using residual reversal. Lo-MacKinlay warned a chunk of
contrarian profit is **lead-lag/bid-ask microstructure bias** (partly illusory).
- [Quantpedia STR](https://quantpedia.com/strategies/short-term-reversal-in-stocks) · [De Groot, Huij, Zhou](https://repub.eur.nl/pub/25718/AnotherLook_2011.pdf) · [Avramov et al. via ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378426611002263) · [NY Fed SR 513](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr513.pdf)

**Verdict.** ✅ Real microstructure phenomenon (liquidity-provision premium); ❌ **net Sharpe ≈ 0 for
the naive version; ~0.8–1.1 only with aggressive large-cap/turnover engineering, and decaying.** It is
"**the most trading-cost-constrained anomaly**" (Frazzini-Israel-Moskowitz). Gross >2–4 HF reversal
numbers are the same liquidity-provision returns — gross, not net.

---

## Consolidated Skeptical Scorecard

| Strategy | Best reported Sharpe | Gross/Net | Credible net today | Primary red flag |
|---|---|---|---|---|
| GGR pairs (distance) | ~1.2 | Net (light proxy) | ~0 post-2002 | Decay; borrow costs ignored at scale |
| Avellaneda-Lee PCA | **1.44** (97–07) | Net | ~0.9–1.1, decaying | Crowding; needs 2× leverage |
| Avellaneda-Lee ETF+volume | 1.51 (03–07) | Net | ~1.0 | Same |
| HF intraday pairs (oil) | **3.9 / 7.2** | **Gross** | ≪ gross; 15bps cuts >50% | 🚩 1-yr window, tiny capacity |
| Overnight→intraday reversal | **4.44** | **Gross** | fraction of it | 🚩 daily full-book turnover |
| OU optimal pairs (portfolio) | ~0.8 | Net | ~0.8–1.0 | Single-pair >2 = selection bias |
| Index-inclusion effect | (was +7–8% event) | n/a | ~0, insignificant 2010s | **Disappeared** |
| Short-term reversal (naive) | gross large | **Gross** | ≈0 or negative net | 🚩 illiquid costs, lead-lag bias |
| Short-term reversal (large-cap residual) | ~1.1 | Net | ~0.8–1.1, decaying | −53% drawdown; capacity-limited |

**Bottom line.** No credible, scalable, net-of-cost, out-of-sample stat-arb Sharpe above ~1.5 survives
scrutiny. The most defensible reference is **Avellaneda & Lee at 1.44 net (1997–2007), decaying to
~0.9**. Every Sharpe >2 found is gross, single-period, low-capacity, execution-gated, or a selection
artifact; the >5 figures (7.2 oil pairs; 4.44 overnight reversal) are gross liquidity-provision returns.
Pervasive decay is the strongest cross-cutting finding.
