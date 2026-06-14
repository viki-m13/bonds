# Phase 3 — Deep Dig into Niches & Amateur Work (where a *real* Sharpe > 5 could hide)

**Prepared:** 2026-06-14
**Mandate:** Stop assuming and go look in the places a genuine high Sharpe is most likely to survive —
under-explored niches and amateur/open-source work — and validate anything promising at the code/primary-
source level.

> **The honest, validated answer: a Sharpe > 5 DOES exist — but only in one structural place, and it is the
> opposite of scalable.** Where bets are *numerous AND genuinely uncorrelated* (professional betting,
> prediction-market arbitrage, latency-elite market-making), the √N mechanism delivers real, sometimes huge,
> Sharpe — because there is no hidden beta and no shared factor for the ratio to be lying about. But that
> exact same property — uncorrelated, idiosyncratic, microstructural — is what caps capacity at thousands-to-
> low-millions of dollars and makes the edge operationally brutal, ban-prone, and fast-decaying. **The Sharpe
> is inversely related to capacity and durability.** And on the amateur side, every open-source backtest
> claiming a clean Sharpe > 3 that we read at the code level had an identifiable bug (lookahead, hyperopt
> overfit, in-sample fitting); the *careful* amateurs who do walk-forward + costs land at ~2.

This file extends Phase 1 ([`01`](01-statistical-arbitrage.md)–[`07`](07-validation-checklist.md)) and Phase
2 ([`08`](08-accessible-high-sharpe-hunt.md)–[`10`](10-tailhedged-shortvol-crypto-basis.md)).

---

## 1. Professional betting & prediction markets — the rare niche where Sharpe > 5 is REAL (and the catch)

**Why this is the least-suspicious high Sharpe in finance.** Moskowitz, *Asset Pricing and Sports Betting*
(Journal of Finance 2021), formally establishes that betting contracts have **zero systematic risk** —
outcomes are exogenous to financial markets and to each other. That is exactly the condition under which the
Fundamental Law (`IR = IC·√BR`) lets Sharpe scale with √breadth *without* a hidden-beta illusion.

**The exact math.** A single flat bet at ~even odds has edge `e` (per-bet ROI) and SD ≈ 1, so single-bet
Sharpe ≈ 0.02–0.04 (pathetic). But N *independent* bets/year give annual Sharpe = `√N · e`:

| Edge/bet `e` | Bets/yr N | Annual Sharpe |
|---|---|---|
| 2% | 1,000 | 0.63 |
| 3% | 10,000 | 3.0 |
| **5%** | **10,000** | **5.0** |
| 5% | 25,000 | 7.9 |
| 4% | 40,000 | 8.9 |

**Reading that table is the whole answer:** Sharpe > 5 needs *both* a durable ~4–5% edge *and* tens of
thousands of genuinely independent bets/year — a regime that exists for **syndicates and bots**, not
individuals.

**Real, documented cases:**
- **Bill Benter** (Hong Kong horse racing): ~$1B lifetime over ~14 profitable years, **~18–24% return on
  turnover** via a 120-variable conditional-logit model combined with public odds. Strategy Sharpe plausibly
  well into double digits. The single strongest real high-Sharpe example — and the *least reproducible* (deep
  pari-mutuel pools, no bookmaker to ban him, multi-year modeling monopoly).
  [Bloomberg](https://www.bloomberg.com/news/features/2018-05-03/the-gambler-who-cracked-the-horse-racing-code)
- **Starlizard (Tony Bloom) / Smartodds (Matthew Benham):** football syndicates, Asian-handicap markets;
  court filings allege ~£600M/yr (disputed). [Racing Post](https://www.racingpost.com/news/britain/high-court-case-alleges-tony-blooms-betting-empire-makes-600m-a-year-so-what-do-we-know-about-his-starlizard-syndicate-aNlkE7t8daxQ/)
- **Polymarket arbitrage** (the highest *gross* Sharpe in the whole survey, near-zero per-trade variance ×
  thousands of trades): the IMDEA study (arXiv 2508.03474, Apr 2024–Apr 2025, 86M bets) documents **~$40M
  extracted**; top wallet **$2.0M over 4,049 trades**. [arXiv](https://arxiv.org/abs/2508.03474)

**The validated catch — Sharpe is inversely related to capacity and durability:**
- **Capacity is trivial.** Polymarket's *entire* arbitrage pool is ~$40M/yr across all arbitrageurs; sub-$0.02
  mispricings have tiny depth — you cannot deploy $100M.
- **It decays fast.** Of ~6,600 Polymarket wallets earning >$5k/month, **53% quit after one month; only 2.6%
  lasted >1 year.** 84% of all traders lose; only 0.033% made >$100k.
- **You get banned.** Soft books limit/close winning bettors; the high-Sharpe edge is exactly what triggers
  the ban. Pinnacle ("winners welcome") is the exception but caps stakes.
- **Survivorship.** Every famous number (Benter, Bloom) is a survivor; the loser distribution is unpublished.
- **ROI ≠ Sharpe.** Betting media quotes "yield" (= per-bet edge `e`), not Sharpe. A 5% ROI is Sharpe ~1.1
  over 500 bets but ~7 over 20,000 — always demand the bet count.

**Verdict:** **Sharpe > 5 is mathematically credible and occasionally real here — the genuine answer to "it's
out there."** But it lives in a few-hundred-K-to-low-millions opportunity for elite operators and bots, is
operationally brutal (line-shopping across many accounts, bans, automated execution), and faces
platform/counterparty/tax tails a backtest never shows. **It is the textbook case of a real high-Sharpe edge
that cannot be levered into size** — which is precisely why it stays under-explored and isn't a fund.
[Moskowitz JF 2021](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2635517) · [Polymarket trader distribution](https://www.kucoin.com/news/flash/84-of-polymarket-traders-are-losing-money-0-033-capture-majority-profits)

---

## 2. Accessible (non-colocation) market-making — a real ~2–4 tier; > 5 gated behind speed

Is there a tier between "retail Sharpe 1–2" and "HFT Sharpe 8–10" reachable with good-but-not-elite infra
(cloud/VPS near exchanges, fast WebSocket APIs, millisecond — not microsecond — reaction)? **Yes, ~2–4 net.**
The decisive bracket:

| Source | Net Sharpe | What it is |
|---|---|---|
| hftbacktest "Queue-Based MM" (CRVUSDT) | **13.2** | Idealized backtest — max rebate, no partial fills, perfect queue model. Not live. |
| hftbacktest "MM with Alpha (APT)", real Binance ticks | **3.0–3.6** (→1.5–2.4 on 2025 variants) | Most realistic open-source figure; models latency + queue + 0.005% rebate |
| Avellaneda-Stoikov on **live** BTC (PLOS ONE) | **−0.24** (−0.11 with RL) | The textbook model *loses money* net |

The entire gap from 13.2 → 3 → negative is **fill realism and adverse selection.** The decisive academic
evidence (Baron, Brogaard & Kirilenko, CME E-mini, real account data): median HFT firm Sharpe **4.3**, top
types 8.5–10.5, **returns ranked by latency, new entrants underperform and exit** — substantially
winner-take-all *within the speed game*. But the **median of 4.3 means a sustainable mid-pack exists**; you
just won't be top-decile.

**Where the accessible mid-tier survives (by avoiding the speed war, not winning it):**
- **Funding-rate-neutral + light MM on crypto perps** — plausibly **3–5**, structural (not latency) edge,
  low drawdown, market-neutral; **capacity-limited and crowding-compressed** (see [`05`](05-crypto-niche.md)).
  Best accessible risk-adjusted bet.
- **Mid-frequency market-neutral stat-arb** (minutes-to-hours, no colocation): **~2–3 net**, latency-tolerant.
- **Crypto MM on less-competitive venues** (hftbacktest APT-style): **~2–3.5 live if disciplined**,
  rebate-dependent, regime-fragile.
- **Pod-shop benchmark** (Millennium/Citadel — diversified breadth without FPGAs): **~2.5 typical, ~4.8 in a
  great year** combining hundreds of uncorrelated books — the realistic ceiling for "breadth without speed."

**Verdict:** Accessible non-colocation MM is **~2–4 net**; >5 sustained is mostly gated behind speed. Discount
any backtested MM Sharpe > 3 by assuming live adverse selection + realistic fills roughly halve it.
[Baron-Brogaard-Kirilenko](https://conference.nber.org/confer/2012/MMf12/Baron_Brogaard_Kirilenko.pdf) · [hftbacktest APT](https://hftbacktest.readthedocs.io/en/latest/tutorials/Market%20Making%20with%20Alpha%20-%20APT.html) · [A-S on live BTC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9767337/)

---

## 3. Fixed-income "pure arbitrages" — "infinite" paper Sharpe that nobody can freely capture

The most credible high-Sharpe strategies in academic finance are near-violations of the law of one price.
**But the high paper Sharpe is an artifact of measuring the *conditional, hold-to-convergence* distribution,
where volatility is near-zero** — and capturing the spread requires cheap repo funding + scarce post-2008
balance-sheet (SLR) capacity + the nerve to survive mark-to-market drawdowns to convergence (the LTCM lesson).

| Arbitrage | Mispricing | Paper Sharpe | Realizable / who captures it | Binding constraint |
|---|---|---|---|---|
| **TIPS-Treasury** (Fleckenstein-Longstaff-Lustig 2014) | 54.5 bps avg, 200+ peak; $56bn in 2008 | "near-∞" if held to convergence | high, but repo+balance-sheet players only; MTM/funding tail | primary-dealer repo collateral |
| **CDS-bond negative basis** (Bai-Collin-Dufresne) | −171 bps IG / −322 HY post-Lehman | "250–650 bps guaranteed" p.a. | real but brutal haircut/MTM risk; forced unwinds | repo haircuts, counterparty |
| **Swap-spread / FI arb** (Duarte-Longstaff-Yu) | few-bps, levered | **Sharpe 0.5–0.8 NET** (vol-targeted, costed) | genuinely ~0.6; RV hedge funds w/ repo | leverage → tail/funding risk |
| **CIP / cross-currency basis** (Du-Tepper-Verdelhan) | 24–27 bps avg; JPY 5Y −90 bps | conditional Sharpe **"infinite"** | net only 9–20 bps — *below* the 30 bps bank hurdle (3% SLR × 10% ROE) | Basel III leverage ratio |
| **Negative swap spreads** (Jermann) | ~ −50 bps (30Y) | apparent pure arbitrage | essentially uncapturable frictionlessly | **SLR** / bond-holding cost |
| **Treasury cash-futures basis** | tiny per unit | high at 20–70× leverage | RV funds only; systemically fragile (March 2020) | repo rollover + futures margin |

The unifying framework (Duffie's *slow-moving capital*; Siriwardane-Sunderam-Wallen's *Segmented Arbitrage*,
which finds the 32 arbitrage spreads have only **22% average pairwise correlation** — no single arbitrageur of
last resort) explains why these persist: each is captured by a specialized, capacity-constrained, regulated
intermediary, and the trade consumes scarce balance sheet.

**Verdict:** The "infinite" Sharpe is real *only* as a hold-to-maturity riskless profit; the realizable,
leveraged, marked-to-market Sharpe is **~0.5–0.8** with a fat left tail — and it's gated behind dealer balance
sheet, not available to retail. The clearest illustration in the whole survey of *why* paper Sharpe overstates
deliverable Sharpe.
[Fleckenstein-Longstaff-Lustig](https://www.nber.org/papers/w16358) · [Duarte-Longstaff-Yu](https://www.anderson.ucla.edu/documents/areas/fac/finance/769.pdf) · [Du-Tepper-Verdelhan](https://www.nber.org/system/files/working_papers/w23170/w23170.pdf) · [Segmented Arbitrage](https://www.hbs.edu/ris/Publication%20Files/24-030_1506d32b-3190-4144-8c75-a2326b87f81e.pdf)

---

## 4. Calendar / announcement anomalies — one survivor (~1), the rest decayed; none near 5

The famous "huge Sharpe in a tiny time window" anomalies, validated against post-publication decay
(McLean-Pontiff: ~26% OOS, ~58% post-publication):

- **Pre-FOMC announcement drift** (Lucca-Moench 2015): ~80% of the equity premium since 1994 earned in the
  24h before scheduled FOMC meetings — a spectacular *in-sample* Sharpe in those ~8 windows/year, but the
  effect **weakened/reversed after ~2011 and post-publication** (the drift even flipped negative in later
  samples). Not a durable > 5.
- **Overnight / "night effect"** (Cooper-Cliff-Gulen; Boyarchenko-Larsen-Whelan "Overnight Drift"): the entire
  equity premium accrues close-to-open (SPY overnight Sharpe ~0.8; the 2–3am ET European-open hour ≈ +3.6%
  annualized). **Real gross, but NOT capturable net of open+close costs in equities** — empirically proven by
  the **NightShares ETFs (NSPY −6.9% vs S&P +22%), which liquidated in 2023.** Only the cheap-futures version
  (Bondarenko-Muravyev) nets ~1.6, and it's waning.
- **Turn-of-month** (McConnell-Xu): 100% of the 1926–2005 market return accrued in a 4-day window; ~Sharpe 1
  gross and the **most durable / tradeable** (liquid index, ~24 round-trips/yr) — but the high Sharpe flatters
  because you sit in cash ~80% of the time, and on capital-deployed terms it's modest.
- **Halloween/Sell-in-May, Monday/weekend, pre-holiday, January/size:** all conform to the decay pattern —
  Halloween OOS-failed in liquid indices post-2002, Monday is dead/reversed, pre-holiday survives only in
  illiquid small-caps, January decayed since ~1988.

**Can stacking them reach 5?** No — most are the *same* macro/liquidity risk premium showing up in different
calendar guises (the FOMC-cycle, overnight, and turn-of-month effects share common drivers), so they hit the
same correlation ceiling as [`08`](08-accessible-high-sharpe-hunt.md)'s ρ-analysis. Combined, realistically
~1–1.5 net.
[Lucca-Moench](https://www.newyorkfed.org/research/staff_reports/sr512) · [Overnight Drift (NY Fed SR917)](https://www.newyorkfed.org/research/staff_reports/sr917) · [NightShares shutdown](https://www.etf.com/sections/news/2-nightshares-etfs-close-after-struggling-gain-traction) · [Turn-of-month](https://quantpedia.com/strategies/turn-of-the-month-in-equity-indexes)

---

## 5. Amateur & open-source work — every clean-looking Sharpe > 3 had a code-level bug

Per your instinct, we hunted GitHub / QuantConnect / Kaggle / Reddit for amateur high-Sharpe work and
**validated the promising ones by reading the actual source code**, not the READMEs. Result: **zero repos
survived as a clean, costed, out-of-sample Sharpe > 3.** The careful amateurs (Quantitativo, Robot Wealth)
who do proper walk-forward + costs top out at **~2.0–2.5**; the careless ones report 10–26 via identifiable
bugs.

| Repo | Claimed Sharpe | Code-level killer (verified) | Verdict |
|---|---|---|---|
| armelf/Financial-Algorithms | **4.83** | **Same-bar lookahead**: entry triggers on `HCrets=(High−Close)/Close` and the *same bar's* High is booked as the return — guaranteed ≥0 by construction; + survivorship (current S&P list on 2007–19) + cherry-picked 7-month window | FAKE |
| freqtrade hyperopt (#3994/#6209) | **26** (and 137,685% profit) | **Hyperopt overfit, no walk-forward** → negative OOS by the reporter's own admission; #6209 is future-candle lookahead + ROI-clipping | FAKE |
| fraserjohnstone/pairs-trading | none (disclaimed) | No transaction costs + **in-sample cointegration pair selection** on the same window | honest spike, no edge |
| ryanczm/Crypto-Stat-Arb | high (advertised) | **Best of the set** — includes 15bps costs, perp funding, point-in-time liquid universe, lagged features — but the alpha-weighting OLS is fit on a window that *includes period-T's own returns* (contemporaneous in-sample fit), + non-standard `Sharpe×16` annualization | OVERSTATED |
| stefan-jansen/ML4T | ~1 | none (Zipline point-in-time, event-driven) | LEGIT, no inflated claim |

**The three universal tells** behind every amateur Sharpe > 3: (1) same-bar / future-candle lookahead, (2)
hyperopt/grid-search overfit reported without walk-forward, (3) in-sample model fitting on the returns being
traded. The one repo that did costs+funding right still leaked via contemporaneous coefficient fitting; the
one repo that did *everything* right only claimed ~1. This is the Deflated-Sharpe / multiple-testing problem
([`06`](06-validation-methodology.md)) in the wild: amateur backtests are an enormous trial pool, so the
visible high Sharpes are overwhelmingly the noise-mined winners.
[armelf](https://github.com/armelf/Financial-Algorithms) · [freqtrade #3994](https://github.com/freqtrade/freqtrade/issues/3994) · [ryanczm/Crypto-Stat-Arb](https://github.com/ryanczm/Crypto-Stat-Arb) · [stefan-jansen/ML4T](https://github.com/stefan-jansen/machine-learning-for-trading)

**Canonical debunked cases the community cites** (the same bugs, named): Ernie Chan's USO/DNO daily
mean-reversion pair showed Sharpe ~1 on vendor end-of-day closes but **collapsed to ~0 when re-run on
executable best-bid/offer** — "beware of low-frequency data," because non-executable consolidated close prices
fake an edge that the spread eats. TradingView/Pine's "peeking into the future" (via `calc_on_order_fills`,
`lookahead_on`, or Renko/Heikin-Ashi synthetic bars) is the canonical generator of dead-straight Sharpe-5+
equity curves. The diagnostic heuristic experienced quants apply: *a too-smooth log-equity curve + annualized
return >12% + Sharpe >1.5 on daily data ⇒ suspect lookahead until forward/event-driven testing proves
otherwise.* [Chan – low-frequency data](http://epchan.blogspot.com/2016/09/really-beware-of-low-frequency-data.html) · [TradingView future-peeking](https://www.tradingview.com/support/solutions/43000614705-strategy-produces-unrealistically-good-results-by-peeking-into-the-future)

**Kaggle quant competitions** (Jane Street, Optiver, G-Research Crypto, Two Sigma, Ubiquant) are an
instructive control: they score on utility / R² / weighted-correlation, *not* Sharpe, on a held-out private
test set — and they are defined by **massive public→private leaderboard shakeups** (e.g., Two Sigma: only one
team stayed top-5 on both boards; G-Research Crypto "saw many big jumps and precipitous drops"; in Optiver,
contestant "stassl" fell from **#1 on the public leaderboard to #154 on the private/live data**). That shakeup
*is* the overfitting tax made visible: even elite competitors with rigorous CV see their edge shrink out of
sample — and these are *prediction-accuracy* metrics (R²/correlation/RMSPE), which translate to even smaller
*net* Sharpe once costs and execution are added. The most telling verified number: in the **Jane Street
Real-Time Market Data Forecasting** competition (Oct 2024–Jan 2025, 3,757 teams, $120k), the **winning** model
("ms capital") scored a sample-weighted out-of-sample **R² of just 0.0139** — i.e., the best of nearly 4,000
elite quant competitors explained ~1.4% of return variance on live data. That is the empirical ceiling of
honest predictive signal in liquid markets, and it is a *gross* statistical edge before any trading cost — a
stark, independent confirmation that durable net Sharpe in liquid markets is small. It's the same lesson as the repos, from the opposite direction — honest OOS predictive signal in
liquid markets is small, and anything that looked huge in-sample reverts.

---

## Phase 3 scorecard

| Niche | Is Sharpe > 5 real here? | Honest read | The catch |
|---|---|---|---|
| **Pro sports betting / racing syndicates** | **YES** (Benter ~double digits) | real, √N-founded, zero beta | tiny capacity, bans, survivorship, ops-brutal, not levereable |
| **Prediction-market arbitrage** | **YES gross** (highest gross Sharpe found) | near-zero per-trade variance × breadth | ~$40M/yr total capacity; decays in weeks |
| **Accessible (non-colo) market-making** | rarely; ~2–4 typical | real mid-tier by avoiding the speed war | adverse selection, rebate-dependence, backtest→live halving |
| **Funding-neutral perp + light MM** | borderline (3–5 plausible) | best accessible risk-adjusted bet | capacity-limited, crowding-compressed, counterparty tail |
| **Fixed-income pure arbitrages** | "∞" on paper, **~0.5–0.8 real** | conditional vol ≈ 0 is the illusion | dealer balance sheet / SLR; MTM/funding tail |
| **Pre-FOMC / overnight / calendar** | no (gross spikes, ~1–1.6 net) | only turn-of-month & futures-overnight net out | open/close costs; NightShares ETFs liquidated; decay |
| **Amateur GitHub/Kaggle backtests** | **no** | every clean-looking >3 had a code bug | lookahead / hyperopt overfit / in-sample fit |

---

## What this phase changes — and what it confirms

**It changes the framing in your favor on one point:** a genuine, honest Sharpe > 5 *does* exist — in
**professional betting and prediction-market arbitrage**, where the √N mechanism is real because the bets
carry zero systematic risk and are truly independent. This is not a measurement illusion; Benter's billion
dollars and the Polymarket arbitrage data are real. **You were right that it's out there.**

**But it confirms the structural thesis at a deeper level:** the property that *makes* the Sharpe real and
high — idiosyncratic, uncorrelated, microstructural edges — is the *same* property that caps capacity at
thousands-to-low-millions, invites bans, and decays in weeks-to-months. **High Sharpe and scalable, durable
capacity are mutually exclusive.** The fixed-income "infinite Sharpe" arbs make the same point from the other
side: their paper Sharpe is infinite precisely because the conditional volatility is zero, and the realizable
Sharpe collapses to ~0.6 the moment you add the leverage, funding, and balance-sheet reality needed to trade
them at size.

So the complete, three-phase answer: **honest Sharpe > 5 exists in exactly the corners that cannot be scaled
or levered** — HFT/market-making (infrastructure-gated), Medallion-scale breadth (closed), and tiny-capacity
uncorrelated-bet niches like professional betting (operationally brutal, ban-prone, capacity-capped). For a
scalable, allocatable, net-of-cost strategy, the ceiling remains **~1–2 single-strategy, ~2–3 diversified** —
and every amateur backtest we read claiming otherwise had a specific, identifiable bug.

*Research only; no existing repository files modified. A couple of source PDFs were Cloudflare/403-gated and
their figures triangulated across mirrors (flagged in the domain notes). The GitHub findings are from reading
the actual repo source, not READMEs.*
