# Validation Methodology — Why High-Sharpe (>5) Claims Are Almost Always Misleading

A research methodologist's evidence base for validating claimed Sharpe ratios, with citable claims and a
practical rubric (the rubric itself is in [`07-validation-checklist.md`](07-validation-checklist.md)).

## Executive summary

A Sharpe ratio is a sample statistic with a wide confidence interval, sensitive to the number of strategies
tested, the annualization method, return autocorrelation, return smoothing, and database biases — every one
of which biases the *reported* number **upward**. The literature establishes that (a) the maximum Sharpe
selected from many trials is inflated even when true skill is zero; (b) published anomaly returns decay
26–58% out of sample; (c) autocorrelation can overstate annualized Sharpe by up to ~65%; (d) illiquid/smoothed
returns mechanically inflate Sharpe; and (e) the few funds that genuinely sustain high Sharpe (Medallion) do
so only at tightly **capped capacity** and are **closed to outsiders**. A claimed Sharpe >5 that is gross,
uncapped, in-sample, annualized by √T from intraday data, and lacking a live track record should be treated
as a near-certain artifact until proven otherwise.

---

## 1. Deflated Sharpe Ratio & selection bias (Bailey & López de Prado, 2014)

When you select the best Sharpe from N trials, the maximum is inflated even if every strategy is pure noise.
The DSR deflates the observed Sharpe by the expected maximum under the null, correcting for skewness,
kurtosis, and sample length. The expected-maximum (threshold) Sharpe under the null:

> **SR₀ = √V[ŜR] × [ (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)) ]**

where γ ≈ 0.5772 (Euler–Mascheroni), e ≈ 2.718, Φ⁻¹ the inverse normal CDF, N = number of trials, V[ŜR] the
cross-sectional variance of trial Sharpes. The DSR test statistic:

> **DSR = Φ( (ŜR\* − SR₀)·√(T−1) / √(1 − γ̂₃·SR₀ + ((γ̂₄−1)/4)·SR₀²) )**

with γ̂₃ = skewness, γ̂₄ = kurtosis, T = sample length. Negative skew and excess kurtosis *reduce*
significance. *J. Portfolio Management* 40(5):94–107.
- [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) · [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) · [formulas (Wikipedia)](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio)

**Implication:** Always ask **N** (number of trials/configs tested). The threshold SR₀ that a noise strategy
clears rises with √(ln N); a high Sharpe with high N and negative skew can deflate to insignificant.

## 2. Backtest overfitting & Minimum Backtest Length (Bailey, Borwein, López de Prado, Zhu, 2014)

High simulated Sharpe is *easily* achievable by trying a modest number of configurations. **With only 5 years
of data, trying more than ~45 independent model configurations almost guarantees a strategy with in-sample
annualized Sharpe of 1.0 but expected out-of-sample Sharpe of 0.** The Minimum Backtest Length (years) must
grow with the number N of independent configurations to hold the expected maximum in-sample Sharpe constant.
*Notices of the AMS*, May 2014.
- [PDF](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf) · [SSRN 2308659](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659) · [AMS](https://www.ams.org/notices/201405/rnoti-p458.pdf)

**Implication:** Demand the ratio of backtest length to number of configurations tried. Short backtest + many
trials = overfit by construction. Most proposals never report N.

## 3. Multiple testing & haircut Sharpe ratios (Harvey & Liu 2015; Harvey, Liu & Zhu 2016)

Reported Sharpe ratios must be "haircut" for data mining. The haircut is **nonlinear** — highest Sharpes only
mildly penalized, marginal Sharpes penalized heavily; a flat 50% haircut is "a serious mistake." Worked
example: a strategy with 20 years of monthly returns (240 obs) and annualized Sharpe 0.75, tested among N≈200
strategies, is haircut to **≈0.32 (≈60% reduction)**. Controls: Bonferroni (strictest), Holm, BHY (controls
FDR, most lenient). Harvey-Liu-Zhu document **≥316 factors** tested in the cross-section and recommend a
critical **t-stat ≈ 3.0** (vs the naive 2.0), closer to **3.18** accounting for unpublished tests.
- [SSRN 2345489 (Backtesting)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489) · [programs](https://people.duke.edu/~charvey/backtesting/)

**Implication:** Convert any claimed Sharpe to a t-stat (t ≈ SR·√(years)) and require it to clear ~3.0, not 2.0.

## 4. Post-publication decay (McLean & Pontiff, 2016)

Studying **97 predictors** of cross-sectional stock returns: portfolio returns are **26% lower out-of-sample**
and **58% lower post-publication**. The implied **32%** (= 58 − 26) is attributed to publication-informed
trading; the remaining 26% an upper bound on data mining. Decay is largest for predictors with the highest
in-sample returns and concentrated in high-idiosyncratic-risk, low-liquidity stocks. *J. Finance*, 2016.
- [SSRN 2156623](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623) · [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)

**Implication:** Discount a high backtested Sharpe on a *known* signal ~26–58%. Strongest in-sample results
decay most.

## 5. Data-snooping & White's Reality Check (Sullivan, Timmermann & White, 1999)

When you mine a universe of trading rules, the best performer's significance must be tested against the
*whole* search universe via White's Reality Check bootstrap. Applying ~7,846 technical rules to ~100 years of
daily DJIA data, the best in-sample rule's apparent profitability was **not significant** after the bootstrap
adjusted for the full search space; out-of-sample profitability collapsed. *J. Finance* 54(5):1647–1691.
- [SSRN 160330](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=160330) · [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00163)

**Implication:** The relevant null is "best of many," not "this one in isolation." Ask whether significance was
tested against the full universe of rules/parameters searched (Reality Check / SPA test).

## 6. Annualization & autocorrelation (Lo, 2002)

Monthly Sharpe **cannot** be annualized by √12 except under IID returns. **The annual Sharpe of a hedge fund
can be overstated by as much as 65% due to serial correlation in monthly returns.** Correct annualization:
SR(q) = η(q)·SR, where η(q) = q / √(q + 2·Σ(q−k)ρ_k) — which is < √q when autocorrelation is positive. Lo
also gives the Sharpe estimator's standard error ≈ √((1 + SR²/2)/T) under IID. *Financial Analysts Journal*
58(4).
- [SSRN 377260](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=377260) · [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios)

**Implication:** Be deeply skeptical of high-frequency/intraday Sharpes annualized by √(periods/year) (√252,
√(390·252)). Positive P&L autocorrelation (common in momentum/mean-reversion) means the true annualized number
is lower. (This is the legitimacy test for the HFT Sharpes in [`02`](02-hft-market-making.md): genuine
market-making P&L is near-IID, so its annualization is *valid*; autocorrelated strategies' is not.)

## 7. Return smoothing & illiquidity (Getmansky, Lo & Makarov, 2004)

Illiquid holdings and smoothed (stale, broker-quoted, or deliberately smoothed) returns understate volatility
and serial-correlation-inflate the Sharpe. Across **908 hedge funds (TASS)**, an MA(2) smoothing model shows
the strongest smoothing in illiquid styles — **convertible arbitrage, fixed-income arbitrage, emerging
markets** exhibit the highest serial correlation; liquid styles show little. Smoothing biases volatility
*down* and Sharpe *up*. *J. Financial Economics* 74(3):529–609.
- [MIT PDF](http://web.mit.edu/Alo/www/Papers/JFE2004Pub.pdf) · [NBER w9571](https://www.nber.org/papers/w9571) · [SSRN 387578](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=387578)

**Implication:** A smooth, low-vol equity curve in illiquid assets is a **red flag, not a virtue**. Test P&L
for first-order autocorrelation; if ρ₁ is high and positive, recompute Sharpe on de-smoothed returns.
Mark-to-model returns can manufacture an arbitrarily high Sharpe. (This is exactly what inflates convertible-arb
and crypto-carry Sharpes — see [`04`](04-futures-cta-arbitrage.md) §6, [`05`](05-crypto-niche.md) §1.)

## 8. Survivorship & backfill (instant-history) bias

Hedge-fund databases overstate performance because dead funds are dropped (survivorship) and new entrants'
prior track records are added only after good performance (backfill / instant history). Common estimates:
**survivorship bias ~2–4%/yr** and **backfill bias ~1–5%/yr** of overstated returns (magnitudes vary by
database/period). Funds typically begin reporting only after a strong incubation period.
- [database biases overview](https://breakingdownfinance.com/finance-topics/alternative-investments/hedge-fund-database-biases/) · [fund-of-funds evidence](https://people.duke.edu/~dah7/fof.pdf)

**Implication:** A track record that *starts* with a strong run is suspect — ask when reporting began vs when
trading began, and whether early months were live or backfilled.

## 9. Sharpe vs capacity vs decay — who actually achieves high Sharpe

Genuinely high Sharpe is real but **structurally capacity-constrained**. Adding capital lowers Sharpe long
before it dents dollar profit, so the only way to *defend* an extreme Sharpe is to cap AUM and refuse outside
capital. High Sharpe is therefore almost definitionally non-scalable.

- **Renaissance Medallion:** ~39% net returns after fees (5% mgmt + 44% performance) and ~66% gross arithmetic
  mean over 1988–2018; reported Sharpe "exceeding 2.0" net, with higher gross/peak figures cited. AUM **capped
  ~$10–15B**, **closed to outsiders in 1993**, last external investor bought out ~2005. $100 in 1988 → ~$398.7M
  by 2018; firm right on only ~50.75% of trades.
- **HFT / market-makers (Virtu):** near-perfect daily win rates and very high implied Sharpe, but on tiny
  per-trade edge at enormous volume — capacity-limited by microstructure, not replicable at size by allocators.
- **Multi-strategy platforms (Citadel etc.):** flagship net ~19%/yr over ~two decades, platform Sharpe
  estimated ~2; DE Shaw composite ~16%/yr — *blended, levered, multi-pod* Sharpes, not single-strategy.
- [Cornell-Capital: Medallion the ultimate counterexample](https://www.cornell-capital.com/blog/2020/02/medallion-fund-the-ultimate-counterexample.html) · [why Renaissance closed Medallion](https://youngandcalculated.substack.com/p/why-renaissance-closed-medallion)

**Implication:** A claimed Sharpe >5 that is also *open to capital / scalable* is internally contradictory —
the world's best documented fund sits ~2 net and is *closed*. Extreme Sharpe + open capacity ≈ red flag.

## 10. Top funds — what's publicly known and the caveats

| Fund | Publicly cited figure | Caveat |
|---|---|---|
| **Renaissance Medallion** | ~39% net / ~66% gross 1988–2018; Sharpe "exceeding 2.0" net | Reconstructed by outsiders. **Closed, capped ~$10–15B, employee-only.** Net is the investable reality. |
| **Two Sigma** | Publishes Sharpe-methodology pieces, not its own ratio | No audited public Sharpe; stresses Sharpe's estimation error. |
| **DE Shaw** | Composite ~16%/yr (5-yr) | Multi-strategy, levered, fees opaque; not a single-strategy Sharpe. |
| **Citadel** | Wellington ~19%/yr net over ~20 yrs; platform Sharpe est. ~2 | Multi-pod, heavily risk-managed and levered. |

None publish audited Sharpe ratios; figures are journalist/third-party estimates. Even the best **net** numbers
cluster around Sharpe ~2, achieved with leverage, diversification across hundreds of uncorrelated bets, and
capped/closed capacity — *not* a single scalable strategy at Sharpe >5.
- [Two Sigma – Making Sense of the Sharpe Ratio](https://www.twosigma.com/articles/making-sense-of-the-sharpe-ratio/)

---

## Source list

- Bailey & López de Prado, *The Deflated Sharpe Ratio* — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) · [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- Bailey, Borwein, López de Prado & Zhu, *Pseudo-Mathematics and Financial Charlatanism* (AMS 2014) — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659) · [PDF](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf)
- Harvey & Liu, *Backtesting* (JPM 2015) — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489) · [programs](https://people.duke.edu/~charvey/backtesting/)
- McLean & Pontiff, *Does Academic Research Destroy Stock Return Predictability?* (JF 2016) — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623)
- Sullivan, Timmermann & White, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (JF 1999) — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=160330)
- Lo, *The Statistics of Sharpe Ratios* (FAJ 2002) — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=377260)
- Getmansky, Lo & Makarov, *An Econometric Model of Serial Correlation and Illiquidity in Hedge Fund Returns* (JFE 2004) — [MIT PDF](http://web.mit.edu/Alo/www/Papers/JFE2004Pub.pdf) · [NBER](https://www.nber.org/papers/w9571)
- Hedge-fund database biases — [overview](https://breakingdownfinance.com/finance-topics/alternative-investments/hedge-fund-database-biases/)
- Medallion — [Cornell-Capital](https://www.cornell-capital.com/blog/2020/02/medallion-fund-the-ultimate-counterexample.html)
- Two Sigma, *Making Sense of the Sharpe Ratio* — [link](https://www.twosigma.com/articles/making-sense-of-the-sharpe-ratio/)

*Confidence note: primary-source statistics (DSR formulas, MinBTL 5yr/45-config, Harvey-Liu 0.75→0.32,
McLean-Pontiff 26%/58%, Lo's 65%, Getmansky's 908-fund TASS, Medallion ~39% net / ~66% gross / Sharpe ~2 net /
capped ~$10B closed) are confirmed from the cited papers. Database-bias magnitudes are well-established ranges.
Fund figures other than Medallion are third-party estimates, flagged as such.*
