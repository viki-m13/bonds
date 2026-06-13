# Futures, CTA & Classic Arbitrage — Validated Survey

*All Sharpe figures cited to primary academic or practitioner sources. Skeptical framing throughout:
realistic single-strategy Sharpes here sit in the **0.5–1.5** band; anything above ~2 net at scale should
be treated as suspect, regime-specific, or capacity-constrained.*

---

## 1. Time-Series Momentum / Trend Following (TSMOM)

**Mechanism.** Long/short a diversified basket of liquid futures (equity indices, bonds, FX, commodities).
Signal = sign of the past 12-month excess return; positions scaled to constant *ex-ante* volatility per
instrument, then aggregated. Monthly rebalance.

**Reported Sharpe.** Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, JFE 104(2): **58
instruments**, **1965–2009**, each position scaled to ~40% ex-ante annualized vol before aggregation.
Quantpedia summarizes the diversified TSMOM factor at a **gross Sharpe ≈ 1.31** (20.7% p.a., ~15.7% vol,
max drawdown −33.9%). AQR's practitioner figure for a diversified, **net-of-cost** managed-futures program
is the more conservative **~0.8** — the realistic deliverable, not the 1.3 backtest. AQR notes average
pairwise correlation across ~60 markets of only ~0.08 (the source of the diversification benefit).
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463) · [JFE](https://www.sciencedirect.com/science/article/pii/S0304405X11002613) · [Quantpedia](https://quantpedia.com/strategies/time-series-momentum-effect) · [AQR – Demystifying Managed Futures](https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/Demystifying-Managed-Futures.pdf)

**Net & robustness.** The ~1.31 is **gross**; net of costs/slippage/fees → **0.7–0.8**. Trend is cheap to
trade (liquid futures), so the haircut is modest. Robust OOS (replicated over ~100 years), live in CTA
track records since the 1980s. Caveat: a meaningful chunk of headline performance comes from the
**volatility-scaling/risk-parity overlay** rather than the momentum signal per se.
- [Lancaster ~100yr](https://eprints.lancs.ac.uk/id/eprint/128366/1/Time_Series_Momentum_in_Nearly_100_Years_of_Stock_Returns.pdf) · [ScienceDirect vol-scaling](https://www.sciencedirect.com/science/article/abs/pii/S1386418116301379)

**Capacity.** **High** — among the highest-capacity systematic strategies (tens of billions/manager).
**Tail risk:** positive skew / "long volatility" (profits in crises); main risk is whipsaw/trend reversal
(2009, 2011–2013). **Verdict: validated**, gross ~1.3, net/live **~0.8**. No red flags. The benchmark
*scalable* strategy.

---

## 2. Cross-Sectional Commodity Momentum & Carry

**Mechanism.** *Carry*: long high-carry / short low-carry (for commodities, the slope of the futures
curve / roll yield), dollar-neutral. *Momentum*: long winners / short losers cross-sectionally. Monthly.
Koijen, Moskowitz, Pedersen & Vrugt (2018), *Carry*, JFE: within-asset-class carry **gross Sharpe ≈ 0.7**
avg; "current carry" diversified across all asset classes **≈ 1.10**. Figures are **gross**; roll costs and
wider spreads in less-liquid contracts → net single-sleeve ~0.4–0.7. **Tail risk:** carry is short
volatility / negative skew (earns the premium for bearing crash risk) — opposite profile to trend, which
is why **combining carry + trend** is attractive.
- [SSRN](https://www.ssrn.com/abstract=2298565) · [NBER w19325](https://www.nber.org/system/files/working_papers/w19325/w19325.pdf) · [NYU Stern PDF](https://pages.stern.nyu.edu/~lpederse/papers/Carry.pdf)

**Verdict.** Validated, single-sleeve net **~0.5–0.7**; cross-asset combined carry ~1.1 gross.

---

## 3. Calendar Spread / Inter-Commodity Spread Arbitrage

**Mechanism.** Simultaneous long/short in the same/related underlying across expiries (calendar spread) or
related commodities (crack/crush spreads). Trade mean-reversion of the spread vs cointegration-implied fair
value; intraday-to-weekly. No single canonical academic Sharpe; figures are strategy/dataset-specific
(crude-oil mean-reverting stat-arb and HMM crude models report positive but regime-dependent Sharpes).
**Suspect-claim flag:** an *Attention Factor* paper reports OOS Sharpe **>4** (2.3 net) on large US
equities — treat as research-grade, capacity-limited, not a validated deliverable. Highly cost-sensitive
(small spreads). Capacity **low-to-moderate**. **Tail risk:** spread blowouts (negative WTI Apr 2020,
storage/delivery squeezes) produce sharp non-linear losses despite low day-to-day vol.
- [MDPI – Crude Oil Stat-Arb](https://www.mdpi.com/2227-9091/12/7/106) · [arXiv HMM crude](https://arxiv.org/pdf/2309.00875) · [arXiv 2510.11616 (Attention Factor)](https://arxiv.org/pdf/2510.11616)

**Verdict.** Real but heterogeneous; validated net Sharpes cluster ~1–1.5 for disciplined liquid spreads;
any **>4** claim is suspect/capacity-bound.

---

## 4. Merger (Risk) Arbitrage

**Mechanism.** On announcement of a cash deal, long the target near the offer price (capturing the deal
spread), short the acquirer for stock deals. Earn the spread as compensation for **deal-break risk**;
diversified across many deals. Mitchell & Pulvino (2001), *J. Finance* 56(6), **4,750 mergers 1963–1998**:
abnormal returns **9.25% p.a. gross → ~3.5–4% after transaction costs**. Market beta ~0 in normal/rising
markets but **rises to ~0.5** when the market falls >4% — the defining "**selling insurance**" /
written-put payoff. A later study cited risk-arb **Sharpe ≈ 0.66** (2004–2014). Confirmed live via Credit
Suisse Merger Arb Liquid Index; spread compression over time as capital crowded in.
- [Wiley/J.Finance](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00401) · [AQR reprint](https://www.aqr.com/Insights/Research/Journal-Article/Characteristics-of-Risk-and-Return-in-Risk-Arbitrage) · [Return Stacked](https://www.returnstacked.com/merger-arbitrage/)

**Verdict.** Validated, net Sharpe **~0.5–0.7**. Returns look smooth and high-Sharpe in calm markets, but
the strategy is **short a put** — it crashes precisely when broad markets crash and deals break en masse
(2008, March 2020). Reported Sharpe **understates** true tail risk; do not annualize the calm-period Sharpe
as if Gaussian.

---

## 5. Fixed-Income Relative Value (the LTCM strategies)

**Mechanism.** *On-the-run / off-the-run*: long cheaper off-the-run (less liquid) Treasury, short richer
on-the-run, betting the **liquidity spread** (a few bp) converges; levered heavily because the raw spread
is tiny. *Swap spread arb*: bet on swap-vs-Treasury convergence. *Cash-futures basis trade*: long
cheapest-to-deliver cash Treasury financed in **repo**, short the futures, capturing the basis to delivery.

**These are leverage-manufactured Sharpes:** the unlevered edge is basis points, so the *apparent* Sharpe
is high only because realized vol is low *until it isn't*. Basis traders ran **mean leverage ~21:1** (up to
**~50:1** via repo). In 2019, large basis traders held **60–67% of total hedge-fund Treasury exposure and
73–80% of repo positions**; aggregate hedge-fund UST exposure peaked near **$2.4tn**.
- [OFR WP 21-01](https://www.financialresearch.gov/working-papers/files/OFRwp-21-01-hedge-funds-and-the-treasury-cash-futures-disconnect.pdf) · [CFTC MRAC](https://www.cftc.gov/media/11671/mrac121024_TreasuryCashFuturesBasisTrade/download) · [Fed Notes 2023](https://www.federalreserve.gov/econres/notes/feds-notes/hedge-fund-treasury-exposures-repo-and-margining-20230908.html)

**The blowups.** LTCM ran ~40% annualized 1994–1997 at apparently low vol, then **lost $4.6bn in <4 months
in 1998** (~$30 debt per $1 capital). Russia default → flight-to-quality → spreads **diverged** instead of
converging; $3.65bn Fed-supervised recapitalization. The basis trade re-blew-up in **March 2020** (margin
spiral, Fed intervention required).
- [Fed History – LTCM near-failure](https://www.federalreserve.gov/econres/notes/feds-notes/hedge-fund-treasury-exposures-repo-and-margining-20230908.html) · [Fed History essay](https://www.federalreservehistory.org/essays/ltcm-near-failure) · [Wikipedia – LTCM](https://en.wikipedia.org/wiki/Long-Term_Capital_Management) · [Wikipedia – Treasury basis trade](https://en.wikipedia.org/wiki/Treasury_basis_trade)

**Verdict.** Validated as a real spread, but the high apparent Sharpe is **manufactured by leverage and is
not robust**. The canonical lesson: **low measured volatility hides enormous left-tail / leverage risk —
the Sharpe ratio is the *wrong* metric.** Any "high Sharpe, low vol" fixed-income RV claim is a leverage
artifact, meaningless without leverage, funding terms, and tail/VaR disclosure.

---

## 6. Convertible Bond Arbitrage

**Mechanism.** Long the convertible (long an embedded equity call → long gamma), short delta-equivalent
underlying to be delta-neutral; profit from cheap embedded optionality (implied-vs-realized vol
convergence) and gamma scalping. Hutchinson and hedge-fund-index data: convertible-arb indices returned
**~13.2% p.a. with ~3.4% std dev over 1995–2004** — naively a very high Sharpe, but this is
**smoothed/illiquid-marked** index data that **overstates** the true Sharpe. Suffered a near-catastrophic
2005 dislocation and 2008 forced-deleveraging crash (long liquidity, crushed in redemption spirals).
Classic "Sharpe overstated by autocorrelated returns" case (Goetzmann et al., *Sharpening Sharpe Ratios*).
- [Hutchinson – Convertible Arbitrage](https://www.efmaefm.org/0efmameetings/efma%20annual%20meetings/2005-Milan/papers/114-hutchinson_paper.pdf) · [DCU thesis](https://doras.dcu.ie/17349/1/mark_c._hutchinson_20120703134219.pdf) · [Goetzmann et al. NBER w9116](https://www.nber.org/system/files/working_papers/w9116/w9116.pdf) · [Bocconi BSIC](https://bsic.it/markets-x-corporate-finance-convertible-arbitrage/)

**Verdict.** Validated, but the often-quoted high Sharpe is **inflated by return smoothing**; de-smoothed
realistic Sharpe **~0.5–1.0** with a fat left tail.

---

## 7. Equity Factor Strategies

**Mechanism.** Long/short cross-sectional sorts, dollar-neutral, monthly: Momentum (12-1), Value, Quality,
Low-vol, **Betting-Against-Beta (BAB)** = lever low-beta to beta 1, short high-beta deleveraged to beta 1.
- **Momentum:** standalone long/short ~**0.5** (AQR).
- **BAB** (Frazzini & Pedersen 2014): US **Sharpe 0.78** (1926–Mar 2012); ~0.75 for 1926–2009 — "about
  twice value, ~40% above momentum." Cross-asset BAB Sharpes 0.22–0.51.
- **Value:** standalone ~0.3–0.4 over long US samples.

Single-factor Sharpes are **gross**; BAB and momentum have high turnover, and BAB's alpha concentrates in
**micro/nano-cap** names (Novy-Marx & Velikov critique → net Sharpe materially lower, capacity limited).
Momentum has crash risk (2009); value endured a long ~2007–2020 drawdown.
- [BAB – NYU Stern](https://pages.stern.nyu.edu/~lpederse/papers/BettingAgainstBeta.pdf) · [NBER w16601](https://www.nber.org/system/files/working_papers/w16601/w16601.pdf) · [Novy-Marx & Velikov critique](https://www.sciencedirect.com/science/article/abs/pii/S0304405X21002051)

**Verdict.** Validated, single-factor net Sharpes realistically **~0.3–0.6** (BAB gross ~0.75 but degrades
on costs/capacity).

---

## 8. Factor Timing & Multi-Factor Combination (the legit path to higher Sharpe)

**Mechanism.** Combine multiple **low/negatively-correlated** sleeves (value, momentum, carry, quality,
low-vol, BAB, trend). Factor *timing* is contentious and adds little robust value (Asness: "factor timing is
deceptively difficult"). Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*, J. Finance:
value and momentum are **negatively correlated, averaging ~−0.49** within asset class; a simple
**equal-weighted global value+momentum portfolio across all assets achieves Sharpe ≈ 1.45** — substantially
above either sleeve alone, because of the diversification structure. The 1.45 is **gross**; net depends on
the most cost-intensive sleeve (momentum).
- [Value and Momentum Everywhere – NYU Stern](https://w4.stern.nyu.edu/facdir/lpederse/papers/ValMomEverywhere.pdf) · [CFA Digest](https://rpc.cfainstitute.org/research/cfa-digest/2013/11/value-and-momentum-everywhere-digest-summary) · [Quantpedia](https://quantpedia.com/strategies/value-and-momentum-factors-across-asset-classes)

**Verdict — the central conclusion.** This is the **only legitimate route to a portfolio Sharpe
meaningfully above any single strategy.** Combining *N* uncorrelated sleeves each of Sharpe *s* yields
portfolio Sharpe ≈ *s·√N*. Real-world correlations are positive in crises, so realized multi-strategy
Sharpes top out around **~1.5–2.0**, not the theoretical maximum. A diversified, well-executed
multi-strategy book at **~1.0–1.5 net** is credible and defensible.

---

## Cross-Cutting Validation Summary

| Strategy | Reported Sharpe (gross) | Realistic net/live | Capacity | Key risk |
|---|---|---|---|---|
| TSMOM / trend | ~1.3 (MOP 2012) | **~0.8** (AQR) | High | Whipsaw; vol-scaling drives much of it |
| Carry (commodity/cross-asset) | 0.7–1.1 (KMPV) | ~0.5–0.7 | Mod-high | Negative skew / crash risk |
| Calendar/inter-commodity spread | ~1–1.5 (varies) | ~1–1.5 | Low-mod | Spread blowouts; cost-sensitive |
| Merger arb | 9.25%→3.5% net; SR ~0.66 | ~0.5–0.7 | Moderate | Short-put / crisis tail |
| FI RV / basis / LTCM | "high", leverage-made | unstable | High (systemic) | **Leverage + funding blowup** |
| Convertible arb | apparent >2 (smoothed) | ~0.5–1.0 de-smoothed | Moderate | Stale marks, 2008-style delever |
| Momentum factor | ~0.5 | ~0.3–0.5 | Mod | Momentum crashes |
| BAB | 0.75–0.78 (F&P) | lower | Low (micro-cap) | Funding liquidity |
| Value+momentum combined | **1.45** (AMP) | ~1.0–1.3 | Mixed | Crisis correlation convergence |

**Skeptical bottom line.** Honest single-strategy Sharpes are **0.5–1.5 gross**, typically **0.4–0.9
net/live**. Any standalone claim above ~2 net at scale is suspect — usually explained by (a) return
smoothing (convertible arb, hedge-fund indices), (b) leverage masking tail risk (basis trade/LTCM, where
Sharpe is the wrong metric entirely), (c) micro-cap concentration (BAB), or (d) in-sample/capacity-limited
research models. The only durable path to a high portfolio Sharpe (~1.5) is **combining genuinely
uncorrelated sleeves** — documented cleanly by *Value and Momentum Everywhere* (1.45). These strategies
trade **lower Sharpe for far higher capacity** than HFT.
