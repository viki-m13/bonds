# DCA strategies — a cited literature & evidence review

*Compiled 2026-06-15 via a multi-source, fact-checked web research sweep (deep-research
harness: 5 search angles × parallel agents, primary-source fetch + adversarial
verification). This document **supersedes** the earlier `literature_review.md`, whose
citations were openly "from memory of the canonical papers" because the original
web-research agent stalled. Effect sizes below were extracted from primary-source PDFs
where possible; figures corroborated only via abstracts/secondary summaries are flagged
inline. Cross-references to this project's own empirical results (`results_*.md`) are noted
as **[panel]**.*

> **The mandate.** SUMMIT is a concentrated biweekly/monthly **DCA stock-selection**
> strategy: every contribution buys ~2 S&P-500 mega-cap momentum leaders (regime-switching
> to discounted-quality names below the 200dma), never sells, and beats QQQ-DCA in ~93% of
> rolling windows. "Best/most-profitable DCA strategy" therefore has **two layers**:
> **(A)** how/when to schedule contributions, and **(B)** what to buy with each
> contribution. The honest literature verdict, up front:
>
> * **Layer A is mostly a behavioral/risk story, not an alpha story.** No contribution-
>   scheduling trick (value averaging, buy-the-dip, valuation/indicator-scaling, cadence,
>   seasonality) reliably beats just investing steadily. Lump-sum beats DCA ~2/3 of the time.
> * **Layer B is where the documented edges live** — and they are exactly the ones SUMMIT
>   monetizes: cross-sectional momentum (long-only), the extreme concentration of long-run
>   equity wealth in a few mega-cap winners (the case for a size tilt + never-selling), and
>   regime-conditional behavior. The literature **confirms** most of SUMMIT's empirical
>   findings, **challenges** one or two, and points to a small number of honest,
>   not-yet-tested directions (chiefly point-in-time fundamental **quality**).

---

## Part A — Contribution scheduling (the "DCA" question proper)

### A1. DCA vs lump-sum: the one robust result

Investing a windfall all at once ("lump sum") beats spreading it via DCA **about two-thirds
of the time**, because markets drift up and DCA's un-invested cash forgoes that drift.

* **Vanguard (2012/2023).** Lump-sum beat 12-month DCA in **~66%** of historical windows
  (US, UK, AU); 60/40, rolling 10-yr, lump-sum end-values **~2.3% higher** on average. The
  2023 update (1976–2022) ties this directly to stocks beating cash ~76% of the time.
  ([2023 PDF](https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf),
  [2012 PDF](https://static.twentyoverten.com/5980d16bbfb1c93238ad9c24/rJpQmY8o7/Dollar-Cost-Averaging-Just-Means-Taking-Risk-Later-Vanguard.pdf))
* **Constantinides (1979, JFQA)** proves DCA is **theoretically suboptimal** for *any*
  rational investor — a pre-committed sequential rule is dominated by an optimal dynamic
  policy. ([Cambridge Core](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/note-on-the-suboptimality-of-dollarcost-averaging-as-an-investment-policy/0C483B96429655B24F34FB628CF9CEEB))
* **Williams & Bacon (1993)** (S&P 1926–91) and **Knight & Mandell (1993)**
  ("Nobody Gains from Dollar-Cost Averaging") reach the same conclusion empirically and via
  Monte Carlo. ([Knight-Mandell](https://www.semanticscholar.org/paper/Nobody-gains-from-dollar-cost-averaging-analytical,-Knight-Mandell/37c959833f3a97af40447504f97f2c3adc335c2b))

**Why DCA is still defensible (the honest steelman):**

* **Risk-reduction, not return.** Cho & Kuvvet (2015, *J. Financial Planning*): under
  mean-variance analysis DCA *lowers* risk and can be optimal for a sufficiently
  risk-averse investor — it buys lower volatility/drawdown at the price of lower expected
  return. ([NSU](https://nsuworks.nova.edu/hcbe_facarticles/769/))
* **Behavioral/regret minimization.** Statman (1995, *JPM*): DCA is "not rational but
  perfectly normal" — prospect theory, regret aversion, self-control.
* **One genuine, narrow rational case — directly relevant to SUMMIT.**
  **Brennan, Li & Torous (2005, *Review of Finance*)** show DCA *can* help an uninformed
  investor **adding individual stocks** when prices **mean-revert** — but the benefit is for
  **idiosyncratic single-stock mean reversion, NOT a diversified index.**
  ([Oxford](https://academic.oup.com/rof/article-abstract/9/4/509/1604943),
  [UCLA PDF](https://www.anderson.ucla.edu/documents/areas/fac/finance/dollarcostave.pdf))
  → SUMMIT DCAs into *individual* names, so this is the one strand of the DCA-scheduling
  literature that actually applies to it; DCA-ing into a total-market index has no such
  rational basis.

**Caveat to retire from the pitch:** Hayley (2010) shows the ubiquitous claim "DCA buys
more shares when cheap, so you get a lower average price and higher returns" is a
**cognitive error** — it benchmarks against a straw-man "buy equal *shares*" plan; a lower
average price does **not** imply higher expected return.
([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1473046))

**Verdict:** **Validated.** Lump-sum > DCA ~2/3 of the time by ~2%; DCA is risk-reduction /
behavioral insurance only. For a cash-flow investor (paycheck contributions), the question is
moot — you are *structurally* a DCA-er, which is exactly SUMMIT's setting.

### A2. Value averaging (Edleson) — weak / largely debunked

Edleson's value averaging (VA) — contribute variable amounts so portfolio *value* grows by a
fixed step (buy more after drops, sell after gains) — appears to beat DCA on **IRR**
(Marshall 2000). But **Hayley (2014)** shows that edge is largely an **artifact of IRR
itself**: VA mechanically front-loads dollars into periods that turned out higher-returning,
biasing IRR upward; on a properly capital-/risk-adjusted basis (charging the cash-reserve
opportunity cost) the advantage shrinks or vanishes. VA also forces **selling** (capital-gains
tax + turnover) and demands an unmodeled cash buffer.
([Hayley PDF](https://openaccess.city.ac.uk/id/eprint/6298/1/VALUE%20AVERAGING%207%20October%202014.pdf))
**Verdict: weak/debunked** — and the forced-selling is anti-thetical to SUMMIT's never-sell,
tax-deferral engine **[panel: `results_rebalance.md`]**.

### A3. Tactical DCA — buy-the-dip, CAPE-scaling, "SmartDCA"

* **Buy-the-dip — the famous negative result.** Maggiulli (2019) gave an investor **perfect
  foresight of every market bottom** and it **still underperformed steady DCA ~70% of the
  time** (cash drag); **missing the bottom by 2 months → loses 97% of the time.**
  ([Of Dollars and Data](https://ofdollarsanddata.com/even-god-couldnt-beat-dollar-cost-averaging/))
  **Verdict: debunked.**
* **CAPE-conditioned DCA.** Luskin (2017, *JFP*): starting from **CAPE > 30**, DCA beat
  lump-sum by ~5% over the next ~24 months — i.e., valuation conditions *which* of LS/DCA is
  better, it is not a money-printing overlay. AQR's "Market Timing: Sin a Little" finds pure
  CAPE timing adds only a **modest** Sharpe bump and overfits easily.
  ([Luskin](https://jonluskin.com/wp-content/uploads/2020/10/JAN17_Contribution_Luskin.pdf),
  [AQR](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Market-Timing-Sin-a-Little.pdf))
  **Verdict: weak/suggestive** (few, overlapping high-CAPE regimes → small effective n).
* **"SmartDCA" price-level scaling** (Calvet et al. 2023, arXiv preprint) proves buying more
  when price is lower lowers average cost — but this is just the **harmonic-vs-arithmetic-mean
  inequality** and assumes free reserve capital. **Verdict: tautological**, same flaw as VA.

### A4. Crypto DCA — almost entirely practitioner lore

Essentially **no peer-reviewed work** on crypto DCA; the liquid BTC history is ~3 cycles
(one path that rose ~100,000×), so every "scaled DCA beats HODL" claim is single-path,
in-sample, usually cost-free. The only conclusions that transfer are the equity ones:
lump-sum beat DCA ~66% of BTC windows but DCA cut drawdown (Calmar ~identical, Amdax,
2012–23, with realistic fees). Indicator-scaled variants (Fear&Greed<15, RSI, 200-week-MA,
MVRV-Z) are **data-mined to ~3 cycles** with hindsight; the "buy Mondays = +14.36%" claim is
**debunked** (River: real advantage ≈ $55 on $2,610 ≈ noise).
**Verdict: practitioner lore**; discount heavily.

### A5. Cadence & seasonality of contributions

* **Frequency (daily/weekly/biweekly/monthly):** no exploitable terminal-wealth difference at
  fixed total dollars + horizon; the only signed effect is the "invest sooner" tilt
  (~0.38%/yr), i.e. the lump-sum result at micro scale. **[panel: SUMMIT is cadence-robust —
  ~93–94% win, ~18–20× at daily/weekly/biweekly/monthly, `results_cadence.md`]** — which is
  the *correct* prediction from this literature: cadence shouldn't matter much, and for SUMMIT
  it doesn't.
* **Turn-of-the-month (TOM):** McConnell & Xu (2008, *FAJ*) — historically *all* the market's
  excess-over-T-bill return accrued in a ~4-day month-end window (31 of 35 countries; liquidity
  mechanism, Etula et al. 2020 *RFS*). **But** the tradable US edge **decayed after ~1990** and
  doesn't survive ~5 bps costs. **Verdict: validated historically, decayed today** — scheduling
  contributions just before month-end is a harmless tilt, not alpha.
* **Seasonality ("Sell in May"/Halloween):** Bouman & Jacobsen (2002, *AER*) found a robust
  Nov–Apr premium (36/37 countries; extended to 300+ yrs by Jacobsen-Zhang). **But**
  Dichtl & Drobetz (2014) find it weakened OOS and a sit-in-cash-all-summer rule underperforms
  buy-and-hold after costs; Sullivan-Timmermann-White flag calendar effects as prime
  data-snooping casualties. **Verdict: disputed; never pause contributions for it.**

> **Part A bottom line.** The biggest levers are **savings rate and time-in-market**, not
> scheduling. If you have cash, invest it (lump-sum); if behavior requires DCA, keep the window
> short. *No contribution-scheduling overlay reliably generates alpha.* SUMMIT's value is **not**
> in its schedule — it is in **what it buys** (Part B), and the literature on cadence-robustness
> matches its own panel result.

---

## Part B — Security selection on a contribution stream (where the edge is)

### B1. Cross-sectional momentum — the core selector

* **Base effect — validated, heavily replicated.** Jegadeesh & Titman (1993, 2001):
  12-month formation/3-month hold ≈ **1.31%/mo**; canonical (6,6) ≈ **1%/mo**, t≈3–4;
  **skip the most recent month** (1-month reversal / bid-ask bounce). Profits **persisted in
  the 1990s** OOS, partially reverse at 4–5 yrs (overreaction).
  ([JT 1993 PDF](https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf))
  → **[panel] confirms:** multi-horizon momentum with a skip-month is SUMMIT's core bull
  selector (`results_momentum.md`).
* **Intermediate / "echo" momentum — contested; do NOT rely on it.** Novy-Marx (2012) argued
  months 12–7 carry the signal, but **Goyal & Wahal (2015, *JFQA*) "Is Momentum an Echo?"**
  find the echo exists **only in the US** and **not in 37 other countries** — likely
  data-snooping (≈55 ways to define the windows).
  ([Goyal-Wahal](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1935601))
  → **[panel] confirms SUMMIT's rejection** of intermediate momentum (it was worse than 12-1
  on the PIT panel); plain multi-horizon momentum is the defensible choice.
* **Residual / idiosyncratic momentum — the best-evidenced enhancement, and an honest
  challenge to SUMMIT.** Blitz, Huij & Martens (2011): rank on the momentum of
  factor-residual returns → **~2× the risk-adjusted profit**, Sharpe roughly **doubles**
  (~0.48 vs ~0.25), much lower crash risk, because it strips the time-varying factor betas
  that drive crashes. ([RePEc](https://ideas.repec.org/a/eee/empfin/v18y2011i3p506-521.html))
  → **Tension:** this is the one enhancement the literature most strongly supports that
  SUMMIT *rejected*. The panel's reason — residual momentum's edge here came from an
  **anti-beta tilt that loses to a high-beta QQQ benchmark** — is internally consistent with
  the low-vol findings (B4) and with SUMMIT being **long-only vs a high-beta benchmark**
  rather than a market-neutral Sharpe-maximizer. The honest framing: residual momentum is
  *Sharpe-superior in long-short space*; SUMMIT optimizes *raw-return win-rate vs QQQ*, a
  different objective. Worth a documented re-test that controls for the beta tilt
  **[panel: `results_ram.md`]**.
* **52-week-high (George-Hwang 2004) & Frog-in-the-Pan (Da-Gurun-Warachka 2014):** both
  validated **in-sample** (52wk spread 1.23%/mo > JT's 1.07%; FIP: +5.94% continuous-info vs
  −2.07% discrete-info) but their edge concentrates in **small/low-coverage/short-leg** names
  and **attenuates in large-cap long-only**, where 52wk-high correlates ~0.8–0.9 with 12-1.
  → **[panel] confirms** these "underperformed plain momentum badly vs a QQQ benchmark."
* **Long-only large-cap reality — the decisive point for SUMMIT's architecture.** The
  long-short academic spread **does not survive** in naive large-cap form: **Sathish Kumar
  (2025)**, S&P 500, survivorship-free, 2006–2024, 10 bps/side, finds classic L/S 12-1 nets
  **−2.79%/yr (Sharpe −0.23, maxDD −81%)** — but the **long leg is +7.9%/yr** while the
  **short/loser leg (−9.1%/yr, +40–100% single-month rebounds) is the killer.** Patton-Weller
  (2017) likewise find funds realize ~**zero** net momentum after costs.
  ([Sathish Kumar](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5367656))
  → **This is a strong, independent validation of SUMMIT's two core choices:** be **long-only**
  (skip the leg that destroys the strategy) and **never sell** (sidestep the turnover/tax that
  Patton-Weller show eats the premium). AQR (Frazzini-Israel-Moskowitz 2018) separately show a
  **low-turnover, long-biased** momentum tilt *can* survive costs — which is precisely SUMMIT's
  shape.
* **Combinations — value+momentum is the strongest validated pairing.** Asness, Moskowitz &
  Pedersen (2013) "Value and Momentum Everywhere": corr ≈ **−0.5**, 50/50 Sharpe ≈ **0.80**;
  QMJ (B5) de-risks momentum's junk-rally tail. These are honest, untested overlays for SUMMIT
  (it has no value/quality leg today).

### B2. Crash protection, volatility management & regime switching

* **Momentum crashes are real and partly forecastable.** Daniel & Moskowitz (2016, *JFE*):
  in panic states (post-bear, high-vol, market rebound) the loser leg's optionality explodes
  (2009: losers **+163%** vs winners **+8%**; up-beta −1.51 vs down-beta −0.70). Their
  **dynamic (vol-scaled) momentum roughly doubles the Sharpe.**
  ([NBER w20439](https://www.nber.org/papers/w20439))
* **Volatility-scaling is the most robust result in this whole area.** Barroso & Santa-Clara
  (2015, *JFE*): scale momentum to constant ~12% vol → **Sharpe 0.53 → 0.97**, **kurtosis
  18.2 → 2.7**, **worst month −79% → −28%**, **maxDD −97% → −45%**. It works because it
  predicts momentum's *own* persistent variance (a risk transform), not returns.
  ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429))
* **But standalone 200dma / breadth *market-timing* is fragile.** Faber (2007) 10-month-SMA
  TAA mainly **reduces drawdown** (≈46% → <10%), not raw return, and is sensitive to the
  cash-yield regime. **Zakamulin (2014, 2016)** shows MA-timing outperformance largely
  evaporates after removing **look-ahead bias** and applying ~50 bps costs — "indistinguishable
  from buy-and-hold." **Huang et al. (2020, *JFE*)** show even diversified time-series momentum
  is statistically fragile under bootstrap. Market **breadth** (% > 200dma) as a timing signal
  is **practitioner-only**, not academically validated.
  ([Zakamulin 2016](https://smallake.kr/wp-content/uploads/2016/04/SSRN-id2743119.pdf))
* **Regime-conditional factor behavior is, however, well-supported for momentum specifically**
  (Daniel-Moskowitz's state-dependent betas; Barroso's variance-timing).
  → **Key reconciliation for SUMMIT:** SUMMIT does **not** use the 200dma+breadth as an
  in/out *timing* switch (the fragile use Zakamulin demolishes). It uses it as a **regime
  switch that changes *what* it buys** (leaders above, discounted-quality below) while
  **never pausing the contribution stream** — the use the regime-conditional literature
  *does* support, and the DCA-native way to avoid Faber's cash-drag and whipsaw.
  **[panel: `regime.py`, `results_bear.md`, `eda_parabolic.md`].** A documented, optional
  **vol-scaled defer** (à la Barroso/Daniel-Moskowitz) is the one crash-protection idea with
  enough evidence to be worth the existing optional "panic-defer" buffer
  **[panel: `literature_enhancements.md`]**.

### B3. Size, concentration & the mega-cap tilt — the deepest theoretical backbone

* **The naive small-cap premium is weak/dead; "size + quality" is the real result.** Banz
  (1981) found ~0.4%/mo (smallest-vs-rest) but **concentrated in microcaps & January**.
  Horowitz-Loughran-Savin (2000) and van Dijk (2011) find it ~zero/negative post-1980.
  **Asness et al. (2018) "Size Matters, If You Control Your Junk":** raw SMB α = 14 bps
  (t=1.23) → **49 bps (t=4.89) only after hedging quality** — i.e., size works *long-short,
  quality-hedged*, not as a naive long-only small tilt.
  ([AFIMP PDF](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2015/05/Size-Matters-if-You-Control-Your-Junk.pdf))
* **Equal-weight lags cap-weight in concentration regimes — the benchmark-relative trap.**
  RSP vs SPY: cap-weighted won 5-yr **74.0% vs 49.6%**, 10-yr **250.9% vs 207.5%** (through
  mid-2026); the Magnificent-7 are ~34% of the cap-weighted index but ~1.4% equal-weighted.
  Equal-weight is structurally a mid-cap/anti-momentum bet that **loses when leadership
  concentrates.**
* **Bessembinder (2018) — why a size tilt + never-selling is rational against a cap-weighted
  benchmark.** Of ~26,000 US stocks 1926–2016: **only 42.6% beat one-month T-bills over their
  life; the modal lifetime return is −100%; the best 4.3% of firms (~1,092) created *all* net
  wealth above T-bills; the top ~90 firms created over half.** Global (63,000 stocks): **2.4%**
  of firms. Concentration is **rising** (firms for half of net wealth: 90→83→72→46 across
  updates). ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2900447))
  → **This is the literature SUMMIT's design rests on.** Because (a) a cap-weighted benchmark
  *is* a bet on the mega-cap right tail, an equal-weight picker structurally lags it in
  mega-cap eras — fixed by SUMMIT's **dollar-volume size tilt [panel: §6.2, the decisive
  ingredient]**; and (b) since you cannot know ex-ante which few names compound into the right
  tail and all wealth comes from them, **forced winner-selling caps the exact tail that
  produces returns** — the mathematical case for **never-selling [panel: `results_add_to_existing.md`,
  `results_rebalance.md`]**. *Caveat:* concentration is regime-dependent — when breadth
  broadens (e.g. RSP > SPY YTD-2026), equal-weight/small briefly win; Bessembinder justifies
  *not fighting* cap-weight, not a guarantee mega-caps win every window (consistent with
  SUMMIT's honest 16/244 losing windows, all the peak-AAPL-concentration QQQ era).

### B4. Low-volatility / lottery — why these tilts *hurt* vs a high-beta growth benchmark

* **Low-vol/low-beta is Sharpe-superior but raw-return-inferior.** Baker-Bradley-Wurgler
  (2011): lowest-vol quintile $1→$10.12 vs highest $1→$0.58 (1968–2008, real) — Sharpe story.
  Frazzini-Pedersen (2014) "Betting Against Beta": BAB Sharpe **0.78**, CAPM α **0.73%/mo
  (t=7.39)** — **but only by levering the low-beta leg to β=1 ($1.40 long / $0.70 short).**
  Unlevered low-beta has **β<1 → captures <100% of upside → must lag a β≥1 growth index in a
  rally.** Robeco/Blitz concede low-vol indices "**significantly lagged** their parent indices
  since autumn 2019" through the growth-led rally.
  ([BAB PDF](https://w4.stern.nyu.edu/facdir/lpederse/papers/BettingAgainstBeta.pdf))
  → **[panel] confirms decisively:** vol-scaling/low-vol *selection* hurts SUMMIT because QQQ
  is high-beta growth; this is mechanical, not a fluke (`results_improvements.md`, `results_ram.md`).
* **The lottery/MAX effect and the right-tail tension — resolves SUMMIT's bear sleeve.**
  Bali-Cakici-Whitelaw (2011): high-MAX (lottery) stocks underperform by **~1.0–1.2%/mo**
  (VW raw −0.93%, t≈−3.2; 4-factor α −1.06%/mo); subsumes the IVOL puzzle. Kumar (2009) gives
  the gambling-clientele mechanism; Bali-Brown-Murray-Tang (2017) show the **beta anomaly is
  lottery-demand in disguise.** **But** Bessembinder shows aggregate wealth *is* the extreme
  right tail — so a **permanent anti-lottery exclusion fights a never-sell winner strategy.**
  The reconciliation (different statistical objects): MAX is a **1-month mean-reversion** signal
  on *currently overpriced* pops, whereas multibaggers are identified by **years of compounding**.
  Crucially, the multibagger-characteristics work (Yartseva 2025, 464 ten-baggers) finds future
  winners start as **cash-generative, reasonably-valued small caps — often near 12-month lows,
  with positive FCF the single strongest predictor — NOT lottery tickets**; extreme volatility
  predicts *losses*. ([BCW PDF](https://pages.stern.nyu.edu/~rwhitela/papers/max%20jfe.pdf))
  → **[panel] validates the bear sleeve:** buying *discounted, long-term-healthy* names below
  the 200dma (not chasing lottery vol) is exactly what this literature says distinguishes future
  winners from lottery losers (`eda_parabolic.md`, `results_bear.md`). It also flags the clean
  upgrade: separate winners from junk by **fundamentals (FCF/valuation)** → B5.

### B5. Quality — the best-evidenced *untested* direction for SUMMIT

Quality is one of the strongest-evidenced factor families, and SUMMIT currently has **no
point-in-time fundamental data** — making this the clearest honest gap.

* **Novy-Marx (2013) gross profitability** (gross profits/assets): 3-factor **α 0.52%/mo
  (t=4.49)**, Fama-MacBeth slope t≈5.5 — "**as much power as book-to-market**," is **the other
  side of value** (negatively loaded on HML), **orthogonal to momentum**, low-turnover. Combined
  profitability+value Sharpe **0.85** vs market 0.34.
  ([JFE PDF](https://oldschoolvalue-files.s3.amazonaws.com/pdf/Novy-Marx_Gross-Profitability-Anomaly_JFE_2013.pdf))
* **Piotroski (2000) F-score** (9 fundamental signals) lifts a high-B/M book by **≥7.5%/yr**;
  high-minus-low F-score spread **~23%/yr (t=5.59)**, 18 of 21 years.
  ([PDF](https://www.anderson.ucla.edu/documents/areas/prg/asam/2019/F-Score.pdf))
* **Asness-Frazzini-Pedersen (2019) Quality-Minus-Junk**: abnormal **71–97 bps/mo (t up to
  9.0)** US, **89–112 bps/mo** globally, **information ratio > 1**, positive in **23 of 24
  countries**; quality is only *modestly* priced (why it persists), and a **low price of
  quality predicts high future QMJ returns** (a timing signal). QMJ's "safety" pillar overlaps
  low-beta/BAB, so quality is partly a *bundle* of profitability+low-risk+growth+payout.
  ([QMJ PDF](http://www.econ.yale.edu/~shiller/behfin/2013_04-10/asness-frazzini-pedersen.pdf))
* **Binding requirement:** all three need **point-in-time fundamentals** (proper reporting
  lags, no restated Compustat) or look-ahead/restatement bias inflates results; the raw
  Piotroski score has shown post-publication decay.
  → **[panel] flagged this as future work** with proper PIT fundamentals; the price-only
  "uptrend-intact" proxy in the bear sleeve is a stand-in. **The honest, evidence-backed
  upgrade path:** add a **gross-profitability / QMJ-safety overlay** (low-turnover, momentum-
  orthogonal, and exactly what separates compounders from lottery junk per B4). This is the
  single most defensible not-yet-tested idea in this review.

### B6. Machine learning & time-series foundation models — thin in liquid large caps

* **ML cross-sectional rankers: real science, oversold as large-cap trading.** Gu-Kelly-Xiu
  (2020, *RFS*): NN OOS monthly R² peaks ~**0.40%** (stock-level), and — contrary to the
  common prior — is **higher in large caps** (top-1000 NN3 R² 0.70%). **But** the headline
  long-short Sharpe **2.45 collapses to 1.69 ex-microcap and 1.35 value-weighted**, before
  costs; Freyberger-Neuhierl-Weber show for the largest firms only ~7 of 36 signals survive
  (SR→1.81); the variable-importance is **dominated by momentum/reversal/liquidity** anyway.
  ([GKX NBER](https://www.nber.org/papers/w25398)) Avramov-Cheng-Metzker (2023, *Mgmt Sci*)
  make it explicit: NN signals **lose >50% of their risk-adjusted return** once microcaps /
  distressed / no-rating firms are excluded, and degrade further under costs; Azevedo et al.
  (2023) find ML **break-even costs of just 5–21 bps**. ML alpha lives in the hard-to-arbitrage
  tail. ([Avramov-Cheng-Metzker](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3450322))
  → **[panel] consistent:** walk-forward LightGBM had OOS IC ≈ 0.002 (t≈0.5), learned a
  defensive beta/vol tilt, and **lost to a single momentum column** (`results_ml.md`). The
  literature predicts exactly this: ML's tradable edge over simple momentum is thin in liquid
  large caps after value-weighting and costs.
* **Time-series foundation models: no cross-sectional alpha claim exists.** Chronos, TimesFM,
  Moirai, TimeGPT, Lag-Llama are **all evaluated purely on forecasting-error** (WQL/MASE/CRPS)
  — **none claims stock-selection alpha, Sharpe, or trading returns.** (TimeGPT only lists
  "finance" as a forecasting *domain*.) ([Chronos](https://arxiv.org/abs/2403.07815))
  → **[panel] validated:** Chronos-bolt re-ranking **underperformed the momentum control it
  re-ranks** at every k (`results_chronos.md`). There is no published basis to expect
  otherwise — forecasting accuracy ≠ cross-sectional alpha.

---

## Part C — Honest backtesting (governance the evidence demands)

* **Survivorship/delisting bias is first-order, especially in free data.** Shumway (1997) and
  Shumway-Warther (1999): missing delisting returns inflate backtests; the standard fixes are
  **−30% (NYSE/AMEX) / −55% (Nasdaq)** imputed delisting returns. The **entire Nasdaq "size
  effect" was a delisting-bias artifact.** Free feeds (Yahoo) drop dead tickers — a 10-year
  lookback can be **missing ~75% of then-trading stocks**; survivorship inflation runs
  ~**1–1.6%/yr** broadly, far more in small/distressed segments.
  ([Shumway 1997](https://www.tylergshumway.org/Shumway-DelistingBiasCRSP-1997.pdf))
  → **[panel] directly validates SUMMIT's honesty:** PIT membership masks, delisting-aware
  accounting, the two data-bug fixes (recycled tickers, garbage spikes), and the explicit
  "Yahoo misses ~26% of historical constituents" caveat are exactly the required mitigations.
* **Multiple-testing / data-mining inflation is huge.** Harvey-Liu-Zhu (2016): with **316
  factors** mined, the t>2 bar is invalid → require **t>3.0** (~3.18 with publication bias);
  most published factors are likely false. McLean-Pontiff (2016): anomalies decay **~26% OOS
  and ~58% post-publication.** Bailey-López de Prado: **~45 trials on 5 yrs of data guarantee a
  spurious high in-sample Sharpe**, and overfit strategies can have **negative** expected OOS.
  ([HLZ](https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF))
* **The recommended honest pipeline:** PIT universe + **purged/embargoed** walk-forward →
  **Combinatorial Purged CV** (distribution of OOS Sharpes, not one lucky path) → **Deflated
  Sharpe Ratio** (discount for #trials, skew, kurtosis, length) + **Probability of Backtest
  Overfitting** → **random-portfolio null** (skill vs same-constraint chance).
  ([DSR](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf),
  [PBO](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf))
  → **[panel] SUMMIT already does the hard parts** (leakage audit, PIT, IS/OOS split,
  random-pick survivorship control, phase/cost/k/cadence/N100 sweeps). **The one concrete
  upgrade this literature recommends:** given the *dozens* of variants swept across the
  `results_*.md` log, compute an explicit **Deflated Sharpe Ratio / PBO** that honestly counts
  the trial budget — the most rigorous defense against the charge that 93% win-rate is
  selection bias. This is a cheap, high-credibility addition.

---

## Part D — Scorecard: literature vs SUMMIT's own findings

| SUMMIT finding **[panel]** | Literature verdict | Source |
|---|---|---|
| Mega-cap (dollar-volume) size tilt is decisive vs cap-weighted QQQ | **Strongly supported** — return concentration in mega-cap right tail; equal-weight lags cap-weight in concentration regimes | Bessembinder 2018; RSP/SPY data |
| Never sell (compound winners, defer tax) | **Strongly supported** — all wealth is the right tail; forced selling caps it; turnover/short-leg destroy L/S momentum | Bessembinder 2018; Sathish Kumar 2025; Patton-Weller 2017 |
| Long-only beats running the full long-short | **Strongly supported** — long leg +7.9%/yr, short leg the killer | Sathish Kumar 2025 |
| Multi-horizon momentum w/ skip-month as core selector | **Validated** | Jegadeesh-Titman 1993/2001 |
| Reject intermediate ("echo") momentum | **Supported** — echo doesn't replicate ex-US | Goyal-Wahal 2015 |
| Reject low-vol / risk-adjusted / anti-lottery tilts vs QQQ | **Supported** — Sharpe-superior but raw-return-inferior vs high-beta growth | Baker-Bradley-Wurgler 2011; Frazzini-Pedersen 2014; Bali et al. 2011 |
| Reject 52-wk-high / vol-compression / volume as selectors | **Consistent** — edge is small-cap/short-leg, attenuates large-cap long-only | George-Hwang 2004; Da et al. 2014 |
| Regime *switch what you buy* (not in/out timing); keep buying in bears | **Supported** — regime-conditional momentum real; MA in/out timing fragile; bear "discounted quality" matches multibagger profile | Daniel-Moskowitz 2016; Zakamulin 2016; Yartseva 2025 |
| Reject LightGBM ranker (loses to momentum) | **Consistent** — ML's large-cap tradable edge thin after VW/costs | Gu-Kelly-Xiu 2020; Freyberger et al. 2020 |
| Reject Chronos foundation model | **Validated** — no model even claims cross-sectional alpha | Chronos/TimesFM/etc. papers |
| Honest data caveats + random-pick control | **Exactly the required mitigation** | Shumway 1997/1999; Burns 2006 |
| **Reject residual/idiosyncratic momentum** | **Challenged** — best-evidenced enhancement (Sharpe ~2×); SUMMIT's reason (anti-beta tilt vs QQQ) is defensible but worth a documented beta-controlled re-test | Blitz-Huij-Martens 2011 |

**Net:** the literature **confirms** SUMMIT's load-bearing choices and **most** of its
rejections. The two honest open items are **residual momentum** (re-test controlling for the
beta tilt) and **point-in-time quality** (a genuinely untested, well-evidenced overlay).

---

## Part E — Honest, not-yet-tested directions worth exploring

Ranked by strength of external evidence × fit to SUMMIT's mandate:

1. **Point-in-time fundamental quality overlay (highest-value untested idea).** Add a
   **gross-profitability (Novy-Marx) / QMJ-safety** screen or tilt. It is momentum-orthogonal,
   low-turnover (fits never-sell), the *other side of value*, and is precisely what separates
   compounders from lottery-junk (B4). Requires sourcing PIT fundamentals — the main cost.
2. **Residual-momentum re-test, beta-controlled.** Re-run the rejected residual signal while
   neutralizing the anti-beta tilt (e.g., within-beta-bucket ranking), to check whether its
   ~2× Sharpe survives *without* fighting the high-beta QQQ benchmark.
3. **Vol-scaled contribution/defer in panic states.** The Barroso/Daniel-Moskowitz transform
   is the most robust crash result; SUMMIT's optional panic-defer is the right idea — consider
   a continuous vol-scaled version as an optional tail buffer (accepting a small terminal-return
   cost, as the panel already finds).
4. **Value+momentum blend as an optional sleeve.** AMP-2013's −0.5 correlation / 0.80 Sharpe is
   the most robust factor combination; could lower drawdown without the low-vol penalty.
5. **Deflated-Sharpe / PBO governance number.** Not a return idea — a *credibility* upgrade:
   publish an honest trial-count-adjusted DSR/PBO on the factsheet.

Directions the evidence says **not** to pursue (already rejected, and the literature agrees):
naive low-vol/anti-lottery selection, ML/foundation-model ranking as a primary selector,
buy-the-dip/indicator-scaled contribution timing, value averaging, seasonality timing.

---

## Master reference list

**A — DCA scheduling.** Constantinides 1979 *JFQA*; Vanguard 2012/2023; Williams-Bacon 1993;
Knight-Mandell 1993; Cho-Kuvvet 2015 *JFP*; Statman 1995 *JPM*; Brennan-Li-Torous 2005
*Rev. Finance*; Hayley 2010 (cognitive error) & 2014 (value-averaging IRR bias);
Marshall 2000; Maggiulli 2019; Luskin 2017 *JFP*; AQR "Market Timing: Sin a Little";
Calvet et al. 2023 (SmartDCA, arXiv); McConnell-Xu 2008 *FAJ*; Etula et al. 2020 *RFS*;
Bouman-Jacobsen 2002 *AER*; Dichtl-Drobetz 2014; Amdax/River (crypto).

**B1–B2 — Momentum & crashes.** Jegadeesh-Titman 1993/2001 *JF*; Novy-Marx 2012 *JFE*;
Goyal-Wahal 2015 *JFQA*; Blitz-Huij-Martens 2011 *JEF*; George-Hwang 2004 *JF*;
Da-Gurun-Warachka 2014 *RFS*; Daniel-Moskowitz 2016 *JFE*; Barroso-Santa-Clara 2015 *JFE*;
Moskowitz-Ooi-Pedersen 2012 *JFE*; Faber 2007; Zakamulin 2014/2016; Huang et al. 2020 *JFE*;
Sathish Kumar 2025; Patton-Weller 2017; Frazzini-Israel-Moskowitz 2018;
Asness-Moskowitz-Pedersen 2013 *JF*.

**B3–B5 — Size, low-vol, lottery, quality.** Banz 1981 *JFE*; Horowitz-Loughran-Savin 2000;
van Dijk 2011 *JBF*; Asness-Frazzini-Israel-Moskowitz-Pedersen 2018 *JFE*;
Bessembinder 2018 *JFE* (+2023 global/updates); Baker-Bradley-Wurgler 2011 *FAJ*;
Frazzini-Pedersen 2014 *JFE*; Asness-Frazzini-Pedersen 2014 *FAJ*; Bali-Cakici-Whitelaw 2011
*JFE*; Kumar 2009 *JF*; Bali-Brown-Murray-Tang 2017 *JFQA*; Boyer-Mitton-Vorkink 2010 *RFS*;
Yartseva 2025 (multibaggers); Novy-Marx 2013 *JFE*; Piotroski 2000 *JAR*;
Asness-Frazzini-Pedersen 2019 *RAST* (QMJ).

**B6 — ML & foundation models.** Gu-Kelly-Xiu 2020 *RFS*; Freyberger-Neuhierl-Weber 2020 *RFS*;
Kelly-Pruitt-Su 2019 *JFE* (IPCA); Avramov-Cheng-Metzker 2023 *Mgmt Sci*; Azevedo-Hoegner-Velikov
2023 (AFA); Leippold-Wang-Zhou 2022 *JFE*; Chronos/TimesFM/Moirai/TimeGPT/Lag-Llama (arXiv,
forecasting only).

**C — Honest backtesting.** Shumway 1997 *JF*; Shumway-Warther 1999 *JF*;
Brown-Goetzmann-Ibbotson-Ross 1992 *RFS*; Harvey-Liu-Zhu 2016 *RFS*; Harvey-Liu 2015 *JPM*
(haircut Sharpe); McLean-Pontiff 2016 *JF*; Bailey-Borwein-López de Prado-Zhu 2014
("Pseudo-Mathematics"; PBO/CSCV); Bailey-López de Prado 2014 *JPM* (Deflated Sharpe);
López de Prado 2018 (purged/embargoed CV, CPCV); Burns 2006 (random portfolios).

*(Full URLs are inline in each section above. Where a primary-source PDF would not parse, the
figure was corroborated across the paper's abstract plus ≥2 independent secondary sources and
flagged; treat load-bearing decimals as accurate to sign/order-of-magnitude and verify against
the original table before quoting in print.)*

---

*Not investment advice. Past performance does not guarantee future results. All SUMMIT
**[panel]** results carry the data caveats in `dca/README.md` §3.*
