# High-Sharpe Quant Strategies — Validated Research Survey

**Prepared:** 2026-06-13
**Scope:** A skeptical, adversarially-validated survey of systematic/quantitative trading
strategies associated with very high risk-adjusted returns (the brief: *find strategies with
Sharpe > 5*), across all asset classes and frequencies — statistical arbitrage,
HFT/market-making, volatility/options, futures/CTA, fixed-income relative value, crypto
microstructure, and niche markets — plus the statistical methodology required to validate
(or debunk) any claimed Sharpe ratio.

**Method:** Six parallel research streams, each fanning out across peer-reviewed papers,
SSRN/arXiv preprints, regulator data, and credible practitioner sources, then applying an
adversarial validation pass: gross vs net, in-sample vs out-of-sample, real vs theoretical,
capacity, tail risk, decay, and data-snooping/overfitting/survivorship/smoothing checks.

> **One-line answer to "what has Sharpe > 5":** Genuinely verified Sharpe > 5 exists in
> essentially **one** place — **professional HFT / electronic market-making** (Virtu's
> ~20+ implied; Baron–Brogaard–Kirilenko's regulator-measured ~8–10) — and it is **real but
> structurally inaccessible and non-scalable** (winner-take-all latency race, ~6 firms,
> tiny strategy capital, gross of enormous fixed tech cost). **Every other** Sharpe > 5 in
> the literature is one of: gross-of-cost, single-period, illiquid/smoothed marks,
> leverage-manufactured, a short-volatility "pennies-in-front-of-a-steamroller" profile that
> Sharpe mismeasures, or a crypto delta-neutral carry whose true risk (exchange blowup) sits
> in an unmeasured left tail. **The best documented investable fund on earth — Renaissance
> Medallion — is ~2.0 net, capped at ~$10B, and closed to outsiders.**

---

## Contents

| File | Domain |
|---|---|
| [`01-statistical-arbitrage.md`](01-statistical-arbitrage.md) | Pairs trading, cointegration/PCA stat-arb, HF mean-reversion, ETF/index-effect, OU modeling, short-term reversal |
| [`02-hft-market-making.md`](02-hft-market-making.md) | Avellaneda–Stoikov, Glosten–Milgrom, order-flow imbalance, latency arbitrage, Virtu, measured HFT firm Sharpes, intraday momentum |
| [`03-volatility-options.md`](03-volatility-options.md) | Variance risk premium, short straddle/VXX, dispersion, put-write/covered-call, skew/term-structure arb, delta-hedged options, VIX roll, gamma scalping |
| [`04-futures-cta-arbitrage.md`](04-futures-cta-arbitrage.md) | Time-series momentum, carry, calendar spreads, merger arb, fixed-income RV/basis trade (LTCM), convertible arb, equity factors, multi-factor combination |
| [`05-crypto-niche.md`](05-crypto-niche.md) | Crypto carry/funding-rate arb, triangular/cross-exchange arb, MEV, AMM LP, crypto momentum/stat-arb, stablecoin delta-neutral, prediction-market & power-spread arb |
| [`06-validation-methodology.md`](06-validation-methodology.md) | Deflated Sharpe, backtest overfitting / MinBTL, multiple-testing haircuts, post-publication decay, data-snooping, annualization/autocorrelation, return smoothing, survivorship, who actually achieves high Sharpe |
| [`07-validation-checklist.md`](07-validation-checklist.md) | Practical rubric for vetting any claimed Sharpe > 5 |
| [`08-accessible-high-sharpe-hunt.md`](08-accessible-high-sharpe-hunt.md) | **Phase 2** — hunting an *accessible* Sharpe > 5 via Twitter/X, Reddit & forums, then validating each claim to destruction (Zarattini/ORB, overnight-ETF reversion, 0DTE, the strategy-stacking ρ-ceiling math) |
| [`09-event-driven-ml-intraday.md`](09-event-driven-ml-intraday.md) | **Phase 2 supplement** — event-driven micro-edges (index rebalance, PEAD, earnings premium, merger arb, SPAC arb) and ML limit-order-book intraday alpha; all decayed-to-0 or thin (~0.3–0.6), none reach 5 |
| [`10-tailhedged-shortvol-crypto-basis.md`](10-tailhedged-shortvol-crypto-basis.md) | **Phase 2 supplement** — can risk-managed/tail-hedged short-vol or crypto basis "fix the tail" into a high Sharpe? No: tail hedges *lower* Sharpe (PUT 0.65→PPUT 0.33); honest nets ~0.8–1.0 and ~1.0–1.8 with correlated left tails |
| [`11-niches-amateur-deep-dig.md`](11-niches-amateur-deep-dig.md) | **Phase 3** — deep dig into niches & amateur work: professional betting/prediction markets (where Sharpe>5 is genuinely REAL but tiny-capacity), accessible market-making (~2–4), fixed-income "infinite" paper arbitrages (~0.6 real), calendar/overnight anomalies, and code-level validation of amateur GitHub/Kaggle backtests (every clean-looking >3 had a bug) |
| [`12-fintwit-screenshot-candidates.md`](12-fintwit-screenshot-candidates.md) | **Phase 3 supplement** — validation of specific X/Twitter-screenshot papers: Factor MAX (net ~0.3–0.6; the chart "2.2" is cumulative return, not Sharpe), X-Trend few-shot trend (gross ~2.7, "5×/10×" is a base-effect artifact, costs set to 0), insider-trading theory (no Sharpe), and the already-covered Zarattini intraday paper |

---

## Master Scorecard

Reported Sharpe is the *headline* claim; "Honest net" is the validated, deliverable figure
after the adversarial pass. ★ marks the only category with verified Sharpe > 5.

| # | Strategy | Reported Sharpe | Gross/Net | Honest net Sharpe | Capacity | Verdict |
|---|---|---|---|---|---|---|
| **HFT / market-making** ||||||
| ★ | Latency arbitrage (races) | very high (~$5B/yr global) | gross of tech | real but ~6 firms | fixed pie | **Real >5, inaccessible** |
| ★ | Virtu (firm-level, S-1) | ~20+ implied (1 loss / 1,238 d) | net trading rev | real | non-scalable | **Real >5, inaccessible** |
| ★ | Measured HFT (Baron-Brogaard-Kirilenko, CME) | ~8–10 (4.29 aggressive cut) | measured P&L | real | returns-to-speed | **Real >5, inaccessible** |
| | Order-flow imbalance (OFI) | R²≈65% (no honest Sharpe) | — | signal real, Sharpe = backtest | tiny | Signal real; standalone Sharpe = overfit |
| | Avellaneda–Stoikov MM | no native Sharpe | — | framework | per-name small | Canonical theory, not a number |
| | Intraday momentum (Gao-Han-Li-Zhou) | **1.08** | net (survives costs) | ~1.0 | **high** | Honest, low Sharpe, accessible |
| **Statistical arbitrage** ||||||
| | Avellaneda–Lee PCA stat-arb | 1.44 (1997–07) | net | ~0.9–1.1, decaying | moderate | Most credible stat-arb; needs 2× leverage |
| | GGR distance pairs | ~1.2 | net (light costs) | ≈0 post-2002 | moderate | Decayed/dead net of borrow + impact |
| | HF intraday pairs (oil) | 3.9 / 7.2 | **gross** | ≪ gross | tiny | 🚩 1-yr window, execution-gated |
| | Overnight→intraday reversal | 4.44 | **gross** | fraction | low | 🚩 daily full-book turnover |
| | Index-inclusion effect | (was +7–8% event) | — | ≈0, insignificant 2010s | — | **Disappeared** |
| | Short-term reversal (naive) | gross large | **gross** | ≈0 or negative net | low | 🚩 illiquid-name costs, lead-lag bias |
| **Volatility / options** ||||||
| | Global VRP composite (Fallon et al.) | 1.0 | net-ish | ~1.0 | moderate | Most credible vol; tail-diversification fails |
| | Short straddle (hedged) | 1.16 | partial | real, hedge-dependent | high | Sharpe mismeasures −800% unhedged tail |
| | Short VXX / XIV | ~2+ (pre-2018) | — | **fraudulent ex-ante** | self-limiting | **−96% in 1 day (Volmageddon)** |
| | Dispersion | 2.47 gross / 0.82 net | both cited | ~0.8 | moderate | Marketed gross is fantasy; short correlation |
| | PutWrite (PUT) / BuyWrite (BXM) | 0.65 / 0.54 | net | ~0.5, mostly equity beta | high | Israelov: true VRP α is <10% of risk |
| | Skew / butterfly arb | 3.5–6 | **gross, in-sample** | not credible net | low | 🚩 Kalman-filter artifact |
| | VIX roll (filtered) | 0.36–0.63 | net | turns negative OOS | low | Honest, modest, OOS decay |
| **Futures / CTA / classic arb** ||||||
| | Time-series momentum / trend | ~1.3 (MOP) | gross | **~0.8** (AQR live) | **high** | Validated; benchmark scalable strategy |
| | Carry (cross-asset) | 0.7–1.1 | gross | ~0.5–0.7 | mod-high | Validated, negative skew |
| | Merger arb | 9.25%→3.5% net; SR ~0.66 | net | ~0.5–0.7 | moderate | "Selling insurance"; crisis tail |
| | FI RV / basis trade / LTCM | "high", leverage-made | — | unstable | high (systemic) | **Leverage masks tail; LTCM/2020 blowups** |
| | Convertible arb | apparent >2 (smoothed) | — | ~0.5–1.0 de-smoothed | moderate | Stale-mark Sharpe illusion |
| | Betting-against-beta (BAB) | 0.78 | gross | lower (micro-cap) | low | Real, capacity-/cost-limited |
| | Value+momentum combined (AMP) | **1.45** | gross | ~1.0–1.3 | mixed | **Legit path to higher Sharpe: diversification** |
| **Crypto / niche** ||||||
| | Crypto carry / cash-and-carry (BTC) | **12.8** BTC, 7.0 ETH | near-gross | tail-blind; → negative by 2025 | OI-limited | 🚩 Sharpe ignores exchange-blowup tail |
| | Funding-rate arb (practitioner) | ~3–5 implied | net pre-tail | cyclical, decaying | moderate | Real carry, counterparty tail dominates |
| | Triangular / indirect arb | 9–14 bps/trade, ~95% win | net | HFT-only | tiny | Valid only co-located |
| | Cross-exchange arb | high historically | often gross | mostly gone | — | **Decayed** |
| | MEV (arb/sandwich) | $675M+ cumulative | net of gas | not a Sharpe strategy | winner-take-most | Real P&L, infra business |
| | Uniswap V3 LP (passive) | often **negative** | net | negative (LVR≈σ²/8) | — | **Value-destructive for ~50% of LPs** |
| | Crypto momentum / stat-arb | ~1–1.5 | gross | <1 net | moderate | Modest, fragile |
| **Reality benchmark** ||||||
| | Renaissance Medallion | "exceeding 2.0" net (higher gross) | net of huge fees | ~2 net | **capped ~$10–15B, CLOSED** | The ceiling for *investable* Sharpe |

---

## Five cross-cutting findings

1. **Verified Sharpe > 5 = professional HFT, and only that.** It is real (regulator-grade
   CME data; Virtu's audited S-1), but it is the property of a handful of firms winning a
   fixed-size, winner-take-all latency race on tiny strategy capital, gross of enormous fixed
   technology cost (FPGA, microwave links, colocation). The signal is trivial and public;
   **the barrier is infrastructure, not alpha.** It cannot be scaled or accessed by ordinary
   participants. See [`02`](02-hft-market-making.md).

2. **The inverse relationship between Sharpe and accessibility is the whole story.** The
   strategies whose Sharpe you can *verify and access* (intraday momentum ~1.0, trend ~0.8,
   stat-arb ~1.0–1.4) are low-Sharpe. The strategies with Sharpe ~10 are inaccessible. There
   is no free lunch in the middle.

3. **Sharpe is the *wrong metric* for the entire short-volatility / carry / convergence
   family** — short straddles, short VXX, dispersion, merger arb, crypto carry, basis trade.
   These collect a smooth premium for bearing a fat negative left tail; Sharpe is computed
   over the benign window that *excludes* the defining disaster (peso problem). XIV printed
   Sharpe ~2+ for years, then **−96% in a single session** (5 Feb 2018). For these, demand
   **Sortino, Calmar, max drawdown, skew, kurtosis, worst-day** — never Sharpe alone. See
   [`03`](03-volatility-options.md), [`04`](04-futures-cta-arbitrage.md),
   [`05`](05-crypto-niche.md).

4. **Pervasive, documented decay.** Pairs trading (dead net post-2002), the index-inclusion
   effect (insignificant by the 2010s), classic short-term reversal (net ≈ 0), crypto carry
   (Sharpe 6.45 in 2020–25 → 4.06 in 2024 → negative in 2025), cross-exchange crypto arb
   (spreads collapsed). McLean & Pontiff quantify the general law: published anomalies decay
   **~26% out-of-sample, ~58% post-publication.** See [`06`](06-validation-methodology.md).

5. **The only legitimate, scalable route to a *high portfolio* Sharpe is combining genuinely
   uncorrelated sleeves.** Mathematically, *N* uncorrelated sleeves each of Sharpe *s* give
   portfolio Sharpe ≈ *s·√N*. Asness–Moskowitz–Pedersen's "Value and Momentum Everywhere"
   reaches a combined Sharpe of **1.45** off two ~0.5–0.7 sleeves precisely because they are
   negatively correlated (~−0.49). Real-world crisis correlation convergence caps the realized
   benefit, so multi-strategy books top out around **~1.5–2.0 net** — exactly where Medallion,
   Citadel, and the best platforms sit. **There is no single-strategy, scalable, net Sharpe > 2.**

---

## Bottom line for the brief

A durable, **net-of-cost, out-of-sample, tail-honest, scalable** Sharpe > 5 **does not exist**
for any participant who is not already a top-tier HFT firm. The honest targets are:

- **~0.8–1.5 net** for a single well-built systematic strategy (trend, stat-arb, VRP, carry).
- **~1.5–2.0 net** for a diversified multi-strategy book combining uncorrelated sleeves —
  the realistic ceiling, and where the best funds on earth actually live.
- **Sharpe > 5** only as the non-scalable property of latency-advantaged market-making, or as
  a measurement illusion (gross / smoothed / leveraged / tail-blind) everywhere else.

---

## Phase 2 addendum — the dedicated hunt for an *accessible* Sharpe > 5

After the Phase 1 survey, a second pass went straight to where practitioners post real strategies
(Twitter/X, Reddit r/algotrading & r/quant, EliteTrader, Hacker News), collected every concrete
retail-accessible Sharpe-5+ claim, and validated each to a reproducible source. Full record in
[`08-accessible-high-sharpe-hunt.md`](08-accessible-high-sharpe-hunt.md). Result:

- The strongest documented practitioner strategies (Carlo Zarattini / Concretum) headline at **Sharpe
  1.3–2.8 in-sample** but **collapse to ~0.4–1.3 net** in every independent replication once real spreads,
  slippage, small-cap short-borrow costs, and a fair 1× leverage benchmark are imposed. **Leverage doesn't
  change Sharpe** — the 1,484% TQQQ ORB headline is the same ~1.0 Sharpe as unlevered.
- The one viral "Sharpe 7.1" (cross-sectional overnight→intraday ETF reversion) is a **gross,
  multiple-tested, open-auction microstructure artifact** — its backtest assumes you both read the open
  price to form the signal *and* fill at that same open (circular), and it claims its biggest edge in the
  liquid US sector ETFs that Petajisto's foundational paper shows are efficiently priced. Honest net: ~0–1.5.
- **Sleeve-stacking is mathematically capped:** combined Sharpe = `s·√(N/(1+(N−1)ρ))` → ceiling **s/√ρ**
  regardless of N. At the empirically measured ρ≈0.16 for short-horizon equity alphas (Kakushadze's 101
  Formulaic Alphas), an s=1 stack maxes at **~2.5 even with infinite sleeves**; a disciplined open-source
  18-sleeve stack lands at 2.2. You cannot add your way to 5.
- Verified Sharpe > 5 remains **only Renaissance Medallion** — the √BR/execution-moat corner (millions of
  near-coin-flip bets, closed since 1993, capped ~$10B). It exists; it is structurally unreachable by
  sleeve-stacking or retail infrastructure.

**Conclusion unchanged and now quantitatively grounded:** a retail-accessible, net-of-cost, out-of-sample,
tail-honest, scalable **Sharpe > 5 does not exist** — and the stacking math explains *why* it is not merely
undiscovered but out of reach without HFT speed or Medallion-scale breadth. Honest targets: **~1–2 net
single-strategy, ~2–3 net for a disciplined diversified book.**

---

## Phase 3 addendum — deep dig into niches & amateur work (where Sharpe > 5 IS real)

A third pass went specifically into under-explored niches and amateur/open-source work — the places a genuine
high Sharpe is most likely to survive. Full record in [`11-niches-amateur-deep-dig.md`](11-niches-amateur-deep-dig.md).
This pass found the **honest answer to "it's out there":**

- **Sharpe > 5 is genuinely REAL in professional betting and prediction-market arbitrage.** Moskowitz (JF
  2021) shows betting contracts have *zero systematic risk* and are truly independent, so the √N mechanism is
  real and un-faked: a 5% edge over 25,000 independent bets = Sharpe 7.9. Bill Benter (~$1B, ~20% on turnover)
  and the Polymarket arb data ($40M extracted; top wallet $2M over 4,049 trades) are real. **You were right —
  it exists.** The validated catch: capacity is trivial (Polymarket's *entire* arb pool ≈ $40M/yr), it decays
  in weeks (53% of strong earners quit within a month), winning gets you banned, and every famous case is
  survivorship-selected. **High Sharpe and scalable/durable capacity are mutually exclusive.**
- **Accessible (non-colocation) market-making** has a real **~2–4** tier (funding-neutral perp + light MM can
  touch 3–5), but > 5 is gated behind the latency war — the same idealized MM backtest shows Sharpe 13 while
  the live textbook model is net-*negative*.
- **Fixed-income "pure arbitrages"** (TIPS-Treasury, CIP basis, negative swap spreads) show "infinite" *paper*
  Sharpe because conditional hold-to-convergence vol ≈ 0 — but the realizable, leveraged, marked-to-market
  Sharpe is **~0.5–0.8**, gated behind dealer balance sheet / SLR, with the LTCM/2020 tail.
- **Calendar/overnight anomalies** mostly decayed; the overnight effect is real gross but *not* capturable net
  in equities — the NightShares ETFs built to harvest it **liquidated** (−6.9% vs S&P +22%).
- **Amateur GitHub/Kaggle work:** we read the *source code* of the highest-Sharpe repos. **Every clean-looking
  Sharpe > 3 had an identifiable bug** — same-bar lookahead (armelf 4.83), hyperopt overfit (freqtrade 26),
  in-sample coefficient fitting (best crypto stat-arb repo). The careful amateurs who do walk-forward + costs
  land at ~2; the one repo that did everything right claimed ~1.

**The deeper, unified conclusion:** honest Sharpe > 5 exists in exactly the corners that **cannot be scaled or
levered** — HFT (infrastructure-gated), Medallion-scale breadth (closed), and tiny-capacity uncorrelated-bet
niches like professional betting (operationally brutal, ban-prone, capacity-capped). The very property that
makes the Sharpe real and high (idiosyncratic, uncorrelated, microstructural) is what caps its capacity. For
a scalable, allocatable, net-of-cost strategy the ceiling remains **~1–2 single / ~2–3 diversified.**

---

*This survey is research only. No existing repository files were modified; all content lives
under `research/`. Figures are attributed to primary sources in each domain file; several
practitioner/fund figures (Medallion, Citadel, DE Shaw) are third-party estimates rather than
audited disclosures and are flagged as such.*
