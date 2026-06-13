# Options & Volatility Arbitrage — Validated, Adversarial Survey

**Scope:** variance risk premium, dispersion, put-write/covered-call, skew/term-structure, delta-hedged
options, VIX roll, gamma scalping.

## Executive verdict up front

Almost every strategy here is a variant of **selling insurance**: collecting a small, steady premium
(the variance/volatility risk premium, VRP) in exchange for taking the other side of crash risk. The
empirical existence of the VRP is one of the most robust facts in asset pricing (Bakshi–Kapadia,
Coval–Shumway, Carr–Wu, Bollerslev). **That is not in dispute.** What *is* in dispute is whether the
reported Sharpe ratios mean anything.

**The core problem — stated bluntly:** The Sharpe ratio is the **wrong metric** for short-volatility
strategies because it assumes symmetric, roughly-normal returns. Short-vol returns are violently
non-normal: high negative skew, high kurtosis, fat left tails. A short-vol book can print Sharpe 2–3 for
3–5 years and then lose 50–100% in **days** (XIV: −96% in one session, 5 Feb 2018). The Sharpe ratio is
computed over the benign period and is mechanically inflated by the *absence* of the very event that
defines the strategy's true risk. This is the **"peso problem"** (the disaster is in the sample's
expectation but not its realized history) and the **"picking up pennies in front of a steamroller"**
profile. Treat every Sharpe below with suspicion proportional to the strategy's left-tail exposure.

---

## 1. Variance Risk Premium Harvesting / Systematic Short Volatility

### 1a. Short ATM straddle (Coval–Shumway / Quantpedia implementation)
**Mechanism.** Monthly, sell 1-month ATM S&P 500 straddle (at bid), buy 15% OTM puts as crash insurance
(at ask), deploy cash + premium into the index. **Reported Sharpe 1.16**; CAGR ~26%; vol ~19%; max
drawdown −24.07% (Quantpedia). Source paper: Coval & Shumway (1999/2001), "Expected Option Returns,"
*J. Finance* 56(3), framed 1986–1995. Quantpedia models bid/ask explicitly (better than most) but likely
still understates real slippage + margin costs. **Tail risk (critical):** Coval–Shumway's headline is
that **zero-beta ATM straddles lose ~3% per week on average** — being *long* straddles is a steady bleed,
so *shorting* them earns the premium but inherits the mirror-image crash exposure. Quantpedia flags "very
abnormal return distribution," "strong serial correlation in large negative days," and historical seller
losses approaching **−800%** on un-hedged variants.
- [Quantpedia VRP Effect](https://quantpedia.com/strategies/volatility-risk-premium-effect) · [Coval & Shumway SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=189840) · [paper PDF](https://business.baylor.edu/don_cunningham/Option%20Returns.pdf)

**Verdict.** Real premium, but Sharpe overstates quality. The −24% drawdown is the *hedged* version over
a benign window (sample ends 1995, before 1998/2008/2018/2020). Unhedged it is uninsurable.

### 1b. Global VRP composite (Fallon, Park & Yu, FAJ 2015)
34 standardized short-volatility return series across equities, rates, FX, commodities, each scaled to
1% monthly vol, combined. **Reported Sharpe: 0.6 equities, 0.5 FI, 0.5 FX, 1.5 commodities, 1.0 global
composite** (vs ~0.4 for equity beta). 20 years from 1995. **Tail risk (critical):** authors explicitly
caution short-vol is "not a free lunch" — "occasional but substantial tail risks" — and crucially found
**diversification does NOT reduce tail risk**: "the tail risk in the diversified portfolio was similar to
the average tail risk of the individual strategies, indicating correlations rise in bad times." This is
the killer caveat: cross-asset vol-selling all blows up together.
- [AlphaArchitect summary](https://alphaarchitect.com/the-variance-risk-premium-is-pervasive/) · [FAJ abstract](https://www.tandfonline.com/doi/abs/10.2469/faj.v71.n5.4) · [ETF.com / Swedroe](https://www.etf.com/sections/index-investor-corner/swedroe-volatility-strategy?nopaging=1)

**Verdict.** Most credible academic VRP source here. The 1.0 composite Sharpe is real but the authors
themselves neutralize the diversification story.

### 1c. Short VXX / inverse VIX ETPs (the steamroller, documented)
Hold short VXX (or long XIV/SVXY), harvesting negative roll yield as ~1-month VIX futures roll down the
contango curve (~80–84% of days), ~3–7% **monthly** roll yield in calm regimes. **Tail risk (THE
canonical disaster):** **5 Feb 2018 "Volmageddon": XIV lost ~96% in a single session and was terminated;
SVXY (1×) lost >80% in one day.** Mechanism (CFA Institute / FAJ): VIX spike → ETPs must *buy* VIX futures
to rebalance → their concentrated buying pushes futures higher → AUM craters → forced to buy more =
**self-reinforcing feedback loop**. The products' own size created the move that destroyed them.
- [CFA Institute "Volmageddon"](https://rpc.cfainstitute.org/research/financial-analysts-journal/2021/volmageddon-failure-short-volatility-products) · [Six Figure Investing](https://www.sixfigureinvesting.com/2019/02/what-caused-the-february-5th-2018-volatility-spike-xiv-termination/)

**Verdict.** Empirical proof that **Sharpe is the wrong metric**. Pre-2018, short-XIV showed Sharpe ~2+ —
then went to zero. **Any backtest ending before Feb 2018 is a textbook benign-period selection artifact.**
Capacity is self-limiting and *dangerous*: scale creates the crash.

---

## 2. Volatility Dispersion Trading (short index vol, long single-name vol)

**Mechanism.** Sell index volatility, buy single-stock volatility on constituents, vega-neutral. Profits
from the **correlation risk premium** — index IV embeds a premium for realized correlation spiking in
crashes; single-name IV does not. Buraschi–Trojani–Vedolin implementation (Quantpedia): monthly, buy puts
on stocks with highest analyst-forecast **disagreement**, sell index puts; 20 single-name + 1 index.
**Reported Sharpe:** BTV implementation (Quantpedia) **0.82**, 15.39% p.a. *after* costs, vol 13.86%,
**max drawdown −43.49%**, 1996–2007. Other marketed claims cite **Sharpe 2.47, ~23.5% p.a.** — treat with
extreme skepticism (gross, in-sample, idealized execution). **Gross vs net (critical):** dispersion is
execution-cost-dominated — hundreds of single-name options with wide idiosyncratic spreads. The profitable
version is the "**dirty**" version (a liquid subset, not all 500 names) because "theory dies in backtests."
The clean 2.47 is not net-achievable. **Tail risk:** short index vol = short correlation; in a crash,
correlations → 1, index vol explodes vs single-name, the trade loses on **both legs simultaneously**.
- [Quantpedia Dispersion](https://quantpedia.com/strategies/dispersion-trading) · [IBKR "Dirty Version"](https://www.interactivebrokers.com/campus/ibkr-quant-news/dispersion-trading-in-practice-the-dirty-version/) · [CQF](https://www.cqf.com/blog/quant-finance-101/what-is-dispersion-trading)

**Verdict.** Genuine premium (correlation risk premium is real), but marketed high Sharpes are
gross/in-sample fantasies. Net Sharpe ~0.8 with −43% drawdown is the honest figure.

---

## 3. Put-Write / Covered-Call Systematic Indices (CBOE PUT, BXM)

### 3a. CBOE S&P 500 PutWrite (PUT)
Sell 1-month ATM S&P 500 puts, fully cash-secured (collateral in T-bills), monthly roll. **Sharpe 0.65**
vs 0.49 for S&P 500; ~10.3% return (Jul 1986–Oct 2008) vs 8.77%; vol **9.91% vs 15.39%**; **max drawdown
−32.7% vs −50.9%**. Higher Sortino too. **Tail risk:** CBOE/Bondarenko explicitly state PUT has "more
negative skewness than the S&P 500" — outperforms in quiet/falling markets, underperforms in sharp
rallies, and −32.7% still arrived in 2008. The modestly-better Sharpe is bought with extra left-tail skew.

### 3b. CBOE BuyWrite (BXM) — and the Israelov critique (the decisive finding)
BXM: 8.5% return, vol 10.7%, **max drawdown −35.8%**, beta 0.62, **Sharpe 0.54** — essentially tied with
S&P 500's 0.56 since 1986. **Israelov & Nielsen, "Covered Calls Uncovered" (AQR, FAJ 2015)** decompose
covered-call returns into: (1) **equity exposure** — most of risk/return; (2) **short-volatility exposure**
— Sharpe **~1.0**, but **<10% of risk**; (3) **equity timing (reversal)** — ~25% of risk, **~zero return**
(uncompensated). The headline BXM Sharpe (~0.54) is **dominated by passive equity beta**, not the vol
premium. The actual vol-selling alpha (Sharpe ~1.0) is small in the package and diluted by an
uncompensated equity-timing bet. Naively buying BXM/PUT is mostly owning equities with capped upside —
you are NOT cleanly harvesting the VRP.
- [BXM Wikipedia](https://en.wikipedia.org/wiki/CBOE_S%26P_500_BuyWrite_Index) · [PUT Wikipedia](https://en.wikipedia.org/wiki/CBOE_S%26P_500_PutWrite_Index) · [Bondarenko/CBOE PutWrite PDF](https://cdn.cboe.com/resources/education/research_publications/PutWriteCBOE19_v14_by_Prof_Oleg_Bondarenko_as_of_June_14.pdf) · [AQR Covered Calls Uncovered](https://images.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/Covered-Calls-Uncovered.pdf)

---

## 4. Options Skew / Term-Structure Arbitrage

**Mechanism.** Trade relative-value mispricings across the IV surface — butterflies (strike-skew
curvature), risk reversals (put/call skew), calendar spreads (term structure). **Reported Sharpe (HANDLE
WITH EXTREME CARE):** a Kalman-filter academic study (Bedendo & Hodges, FDIC working paper) reports
**Sharpe ~3.5 (butterfly) to ~6 (risk reversal).** These are almost certainly **gross, in-sample,
frictionless**; no net/out-of-sample confirmation could be extracted (binary PDF). Sharpe 6 in a tradable
multi-leg option structure does not survive bid/ask. Skew/butterfly trades are maximally cost-sensitive
(multiple legs each crossing a spread, frequent curvature rehedging). **Net Sharpe is a small fraction of
gross.**
- [Bedendo & Hodges (FDIC WP)](https://www.fdic.gov/system/files/2024-08/mbedendo-shodges.pdf) · [Rival Systems skew](https://www.rivalsystems.com/news/archive/2025/12/trading-opportunities-in-volatility-skew-strategies-for-professional-options-desks/)

**Verdict.** The Sharpe 3.5–6 figures are **not credible as net, live numbers — flag as in-sample
Kalman-filter artifacts.** Treat any skew-arb pitch quoting Sharpe >2 as gross/in-sample until proven
otherwise.

---

## 5. Delta-Hedged Option Returns (Bakshi–Kapadia 2003, *RFS*)

Buy an option, continuously delta-hedge with the underlying to strip out direction, leaving pure VRP
exposure. **Finding (foundational, not a strategy Sharpe):** delta-hedged **long** option portfolios on
the S&P 500 **systematically underperform zero** (lose money), and lose *more* when volatility is high —
proving a **negative market volatility risk premium**: option buyers overpay, sellers are compensated.
Continuous delta-hedging incurs heavy transaction costs; discrete hedging adds path-dependent error.
Delta-hedging removes *direction* but **NOT gamma/vega/jump risk** — a short-vol delta-hedged book still
gets destroyed by a gap move (the crash scenario).
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=267106) · [Oxford RFS](https://academic.oup.com/rfs/article-abstract/16/2/527/1579962) · [Bakshi-Kapadia individual-equity VRP](https://people.umass.edu/~nkapadia/docs/Bakshi_Kapadia_JoD_Fall_2003.pdf)

**Verdict.** Gold-standard academic validation that the premium *exists* and is *negative* (sellers earn
it). The theoretical engine under strategies 1–3; not a Sharpe to trade directly.

---

## 6. VIX Futures Calendar / Roll-Yield Strategies

**Mechanism (Simon & Campasano, "The VIX Futures Basis").** Short VIX futures when basis is in
**contango** (daily roll > +0.10 pts), long when **backwardation** (< −0.10). Hold ~5 trading days;
**hedge with E-mini S&P 500 futures**; require ≥10 days to maturity. In-sample CAGR ~19.67% (2007–2011);
related Cheng paper cites Sharpe **0.36**; Quantpedia's broader implementation cites **0.63**. **OOS
(critical):** Quantpedia notes out-of-sample results turn "slightly negative" — "deteriorating alpha,
warranting caution." The S&P hedge mitigates *some* curve risk, but the short-contango leg is the same
Volmageddon trade — a curve inversion (contango→backwardation overnight) inflicts large losses, and in a
*gap* event the regime-filter switch fires too late.
- [Quantpedia VIX term structure](https://quantpedia.com/strategies/exploiting-term-structure-of-vix-futures) · [Simon & Campasano](https://arxiv.org/pdf/2103.02016) · [QuantConnect replication](https://www.quantconnect.com/research/15261/exploiting-term-structure-of-vix-futures/)

**Verdict.** Modest *honest* Sharpe (0.36–0.63); the literature itself admits OOS decay. More credible than
the inverse-ETP version (regime filter + S&P hedge) but still short the same tail.

---

## 7. Gamma Scalping

Hold **long gamma** (e.g., long straddle), repeatedly delta-hedge — the mirror image of short-vol;
profitable when **realized vol > implied vol**. No credible standalone systematic Sharpe exists — it is a
market-maker / discretionary technique whose expected return is the **negative** of the VRP (you pay the
premium to be long vol). Net profitability requires realized vol to exceed implied by *more* than
cumulative hedging-cost drag. It is **long tail risk** (makes money in crashes); the "risk" is steady theta
bleed in calm markets — the inverse peso problem.
- [Volatility Box gamma scalping](https://volatilitybox.com/research/gamma-scalping-explained/) · [MenthorQ](https://menthorq.com/guide/gamma-scalping-and-delta-hedging/)

**Verdict.** Not a "high Sharpe arbitrage" — it's the *purchase* of the insurance the rest of this report
*sells*. Honest profile: negative carry / positive convexity. Useful as a tail hedge, not a Sharpe-maximizer.

---

## Cross-cutting adversarial conclusions

| Strategy | Reported Sharpe | Honest read | Dominant flaw |
|---|---|---|---|
| Short straddle (hedged) | 1.16 | Real premium, hedge-dependent | Benign-period selection; −800% unhedged |
| Global VRP composite | 1.0 | Most credible | Diversification fails in tail (authors admit) |
| Short VXX/XIV | ~2+ (pre-2018) | **Fraudulent ex-ante** | −96% in 1 day; self-feeding crash |
| Dispersion | 0.82 net / 2.47 gross | ~0.8 net is honest | Execution costs; short correlation |
| PutWrite (PUT) | 0.65 | Marginally > S&P | More negative skew; −32.7% DD |
| Covered call (BXM) | 0.54 | Mostly equity beta | Israelov: real VRP Sharpe~1 but <10% of risk |
| Skew/butterfly arb | 3.5–6 | **Not credible net** | In-sample Kalman artifact |
| Delta-hedged (B-K) | n/a | Premium exists, negative | Theory, not a costed strategy |
| VIX roll (filtered) | 0.36–0.63 | Honest, modest | Out-of-sample turns negative |
| Gamma scalping | n/a | Negative carry | It BUYS the premium others sell |

**Five red flags that recur:** (1) **Sharpe is structurally wrong** for short-vol — computed over windows
that exclude the defining disaster (peso problem); demand Sortino, Calmar, max drawdown, skew, kurtosis,
worst-day instead. (2) **Selection of benign periods** — any short-vol backtest ending before Feb 2018 and
not spanning 2008+2020 is suspect. (3) **Gross ≠ net** — options are expensive; dispersion and skew-arb
especially lose most of their edge to costs. (4) **Return smoothing** — stale single-name/illiquid option
marks artificially lower measured vol → inflated Sharpe. (5) **Diversification doesn't save you** —
vol-sellers across assets all blow up together when correlations → 1 (Fallon et al. proved this).

**Bottom line.** The VRP is one of the best-documented premia in finance (Bakshi–Kapadia, Coval–Shumway,
Carr–Wu, Fallon et al. all converge). But the high Sharpes are an artifact of measuring a negatively-skewed
insurance premium with a metric built for symmetric returns. These strategies *do* pick up pennies — the
steamroller is real, periodic, and has already flattened XIV. Size and risk-manage accordingly; never judge
them by Sharpe alone.

*Note: three PDFs (CBOE Bondarenko, AQR Covered Calls, FDIC skew) returned as binary and could not be
parsed directly — figures sourced from indexing results and corroborating pages. The skew-arb Sharpe 3.5–6
should be treated as the flagged in-sample Kalman-filter figure, not an independently verified net number.*
