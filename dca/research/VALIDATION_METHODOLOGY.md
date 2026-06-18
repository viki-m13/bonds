# Strategy Validation Playbook

*How to independently validate a DCA stock-selection strategy and decide
whether its headline numbers are real or an artifact. Written after
validating SUMMIT, ZENITH, the WAVE/ROTATOR replication, and the external
CRT / "Daily Stock Guide" strategy. Future agents: read this before you
trust any backtest in this repo or anyone's pitch.*

---

## TL;DR — the one-paragraph version

A strategy's headline backtest is **guilty until proven innocent**. The three
things that manufacture fake edges are, in order of impact: **(1) survivorship
bias**, **(2) recency / regime luck measured cumulatively-to-today**, and
**(3) benchmark choice (beating SPY ≠ skill)**. Re-run the strategy on a
survivorship-clean point-in-time panel, with a strict train/test split,
benchmarked against **QQQ-DCA** (not SPY), and look at the *rolling* window
beat-rate and the *cutoff-date trajectory* — not the cumulative-to-the-peak
number. Almost every "beats the market" DCA stock-picker we tested collapses
under one of these. The empirical bottom line of this whole research program:
**there is no durable cross-sectional stock-selection alpha over a
momentum-heavy benchmark (QQQ) that survives out-of-sample with the public
data we have.** What looks like alpha is regime/beta + a favorable recent era.

---

## The 10 tests (run them in this order)

### 1. Survivorship-clean panel (the biggest single distortion)
Free Yahoo data is **missing ~26% of historical S&P 500 constituents** (the
delisted/acquired losers), concentrated pre-2015 (≈50% coverage in 2003 → 96%
by 2025). Backtests on it are inflated.
- **Fix:** rebuild on a point-in-time panel that *includes delisted names* and
  masks selection to actual membership at each date. The broadest free
  survivorship-clean universe available is **PIT S&P 500 ∪ Nasdaq-100 with
  delisted/acquired names backfilled + volume** (≈690 names). Build it once and
  cache it.
- **Measured impact:** SUMMIT's published **20.0× → 9.81×** ITD; ZENITH's
  **25.7× → 11.2×**. Roughly **halves the MOIC, ~−5pp IRR**. Concentration
  *amplifies* the bias (k=1 ZENITH took a bigger haircut than k=2 SUMMIT).
- **Residual to watch:** even the clean panel is missing OTC bankruptcy tickers
  (LEHMQ, AAMRQ…). Treating acquisitions as 0% cash-out is mildly optimistic;
  treating bankruptcies as anything but −100% flatters crash-era windows.

### 2. Strict train/test (out-of-sample) split
Tune/select on **< 2018 only**; report **2018+ once**, never inside a selection
loop. Beware fake-OOS: "later start dates on the same panel" with a signal that
was itself developed on that panel is **not** out-of-sample.

### 3. Use the RIGHT benchmark — QQQ-DCA, not SPY-DCA
Beating **SPY** is easy: a growth/tech tilt does it automatically. The honest
bar is **QQQ-DCA** — a low-cost, cap-weighted *momentum machine* that already
holds the large-cap winners. "Beats the S&P 500 in 100% of windows" is usually
**growth beta**, not selection skill. Re-benchmark vs QQQ and the edge often
shrinks to a few pp — or **inverts in mega-cap-led regimes** (every strategy we
tested loses to QQQ when mega-caps lead, e.g. 2022–2025).

### 4. Cutoff-date trajectory (the recency test — the killer diagnostic)
"100% of windows beat X" almost always means windows **ending at today's peak**.
Compute the strategy ÷ benchmark money-multiple at **successive cutoff dates**
(2017, 2019, 2021, 2023, 2024, now). If the lead only materializes in the last
~24 months, it's regime luck.
- **Example (WAVE/ROTATOR vs QQQ, clean panel):** ratio was **0.70 (2017),
  0.72 (2019), 1.01 (2021), 1.04 (Jun-2024), then 2.10 (Jun-2026)**. The entire
  2× lead was built in the final ~2 years (the AI/semis run). Through 18 years
  it was *tied or behind*.

### 5. Rolling vs cumulative win-rate
A live page can show "win-rate by horizon" where **3y/5y/10y rolling ≈ 50%
(coin-flip, median ~0, worst −45% to −57%)** while **"to-end" = 100%**. The
to-end number is the recency artifact. Always read the *fixed-length rolling*
beat-rate, not the cumulative one.

### 6. Information Coefficient (IC) study — before believing any portfolio
Measure the signal's **cross-sectional rank-IC vs forward RELATIVE-to-benchmark
returns**, in train AND test, in the relevant universe (e.g. top-120 by dollar
volume). A real signal has stable IC with the same sign in both eras.
- **Findings:** price/volume signals (momentum, size, 52wk-high, accel,
  rel-strength, sharpe-mom) have **IC ≈ 0 (|t|<1)** vs beating QQQ at 1y, OOS.
  Fundamentals (gross profitability, earnings growth, ROE, issuance) look
  **strong in-sample (gross-profit t=6.5) but collapse / flip sign OOS**. PEAD
  / earnings-surprise is **sign-unstable and ~0 at 1y**. **Network / spillover
  momentum** (neighbors' momentum predicting a stock, the testable kernel of
  Pu-Zohren "network momentum") is also **IC ≈ 0, |t|<0.8, sign-flipping
  train→test** in the large-cap pond. Reason: **QQQ already embeds the
  price-momentum factor**, so momentum (own *or* networked) can't out-predict
  it; and the large-cap pond is too efficient for fundamental signals to
  survive.

### 7. Survivorship-robust controls (random-pick & equal-weight)
If the strategy can't beat **equal-weight of the same (equally-biased)
universe**, it has no selection skill — survivorship is doing the work. Same
logic as a random-pick control drawn from the same PIT universe.
- **Finding:** mid-cap momentum vs equal-weight = **38% (3y) / 53% (1y)
  in-sample** — i.e. *no durable selection skill even down-cap*; it only
  "works" 2018–2024 (recency again).

### 8. Leakage audit (necessary, not sufficient)
Truncation test: rebuild the signal with all data after date T deleted; the
signal row at T must be byte-identical to the full-sample build (`audit.py`,
`audit_builder`). Catches look-ahead/centering/global-fit bugs. **But clean,
causal code can still be survivorship- and recency-fooled** — passing the
leakage audit is not exoneration (SUMMIT and ZENITH both passed it).

### 9. Drawdown on the accumulating book
Report peak-to-trough on the actual DCA portfolio, not just the multiple.
Concentrated never-sell pickers run **−55% to −77%** drawdowns; crash-recovery
strategies require *enduring* the crash to capture the recovery.

### 10. Robustness & overfitting-via-selection
- Parameter choice should sit on a **plateau**, not an argmax spike. Sweep
  k / cadence / cost / schedule-offset.
- **Selection contamination:** if the final model/variant was chosen *by* its
  walk-forward/OOS performance (e.g. picking 1 of 150 variants, or the
  foundation-model filter that happens to rescue the single losing split), the
  OOS claim is partly in-sample. Count how many variants were tried.

---

## Case study: how SUMMIT was determined "no good"

SUMMIT (`dca/strategy_dca.py`, k=2 regime-switched mega-cap momentum + dollar-
volume tilt, never-sell) published **20.0× / 24.7% IRR since 2006, beats
QQQ-DCA in 93% of 244 windows**.

What the playbook found:
1. **Faithful re-build on the survivorship-clean union panel → 9.81× / 19.2%**
   (≈ ties QQQ-DCA's 9.17×). The headline 20× was **~half survivorship**.
2. **In-sample (pre-2018) rolling beat-rate vs QQQ collapsed to ~50%**, median
   *below* QQQ (medrel 0.95–0.98). The "93%" was survivorship + a window grid
   weighted to the favorable recent era.
3. **Code was causally clean** (passed the leakage audit) and the dollar-volume
   tilt is a genuine +~4pp lever — the problem was the *data and framing*, not
   leakage.
4. Net: SUMMIT is a fine *momentum tilt* but not the 20×/93% machine claimed.
   **Retired from the daily cron.** ZENITH (= SUMMIT at k=1) is strictly worse
   on honest data: 25.7×→11.2×, deeper drawdowns (−58%/−74%), not a Pareto
   improvement. The WAVE/ROTATOR replication beats both on the clean panel
   (19.2× ITD) **but** its lead is the 2024–26 recency artifact (test #4).

**Lesson:** the strategies that looked best had the most survivorship exposure
and the most recency-weighted metrics. The honesty checks that mattered were
#1 (clean panel), #4 (cutoff trajectory), and #6 (IC) — not the leakage audit
the authors leaned on.

---

## The standing empirical conclusion (as of 2026-06)

Tested across every signal family (price/volume, fundamentals via EDGAR, PEAD),
every universe (mega-cap, broad large-cap, mid-cap), with survivorship-robust
controls and strict OOS:

> **No durable cross-sectional stock-selection alpha over QQQ-DCA survives
> out-of-sample with public price + filing data.** Apparent edges are
> regime/beta (crash-recovery, dispersion, mega-cap momentum) plus a favorable
> 2018–2026 era plus (often) a soft SPY benchmark. The only durable lever is
> *risk/timing* (a Faber-style 200/210-dma cash gate cuts drawdowns), not
> selection.

Also tested and dead-OOS vs QQQ-DCA (so future agents don't re-run): **DCA
mechanics** (value-averaging — *sells into uptrends, lags to 0.71×*; a never-
sell asymmetric VA only *matches* DCA; buy-the-dip reserves wash out on cash
drag — "time in market beats timing"); **sector-ETF momentum rotation** (beats
QQQ in 0–30% of windows — QQQ already *is* the winning-sector bet);
**rolling-correlation regime gates** (absorption ratio: de-risks but lags QQQ
0–20%); and **ML on sector correlations** (GBM top-3 beats QQQ ~38% of months —
worse than a coin flip; a 9-asset × ~300-month panel has nothing generalizable).
Physics-inspired selection (**Hurst exponent**, **permutation entropy**) is also
~0 / sign-unstable OOS — a transform of price data can't add information that
isn't there. A **Critical-Slowing-Down gate** (go to cash when lag-1
autocorrelation *and* variance both rise — the tipping-point signature from
Scheffer/Dakos early-warning theory) looked promising *in-sample on QQQ*
(full-history it beat the 200-MA gate and ~matched plain), **but FAILED the
out-of-sample generalization test** (threshold fixed pre-2018, applied 2018+
across 27 ETFs/stocks): it reduced drawdown vs plain in only **11/27** assets,
beat the 200-MA on Calmar in **12/27** (coin flip), beat plain in **4/27**, cost
a median **−3.4 pp CAGR**, and on QQQ itself made the OOS drawdown slightly
*worse*. The in-sample "win" was **driven entirely by the 2008 GFC** (a slow
tipping-point crash); 2020 was too fast and 2022 too grinding for the signal,
and it doesn't transfer across assets. *Lesson: this one fooled me in-sample
on one asset/path until the generalization test killed it — hold your own ideas
to the SUMMIT/ZENITH bar.* The blunt 200-MA is the more *reliable* (if
return-costly) drawdown reducer; no gate improved risk-adjusted return OOS.
**"Just DCA into a tech ETF / rotate across ETFs" (common idea):** static
concentration in tech/semis (XLK 1.3×, SMH 2.8× vs QQQ) *did* out-return QQQ —
but only in the 2018+ AI/semis regime (beat QQQ in just 27–33% of *pre-2018* 5y
windows), with deeper drawdowns (SMH −34/−38% vs QQQ −24%), and it's pure
hindsight: the same idea in **ARKK lost to QQQ (0.58×) with a −74% drawdown**.
It's "more of the winning beta" (a concentration/risk dial, leverage's cousin),
not alpha. A **rule-based ETF momentum rotation does NOT beat QQQ** (8–50% of
windows, median 0.78–1.03) — same as sector rotation, it leaves tech at the
wrong times. **Oversold / mean-reversion ETF rotation also fails** (RSI,
drawdown, 1m-reversal, and "buy-the-dip-in-an-uptrend": 0–33% of windows beat
QQQ, median 0.79–0.94) — it rotates *into laggards, away from the winner*.
*Both directions of ETF rotation tested and dead:* momentum leaves the winner,
mean-reversion fights it; any rotation basket holds less of QQQ's winning beta
than 100%-QQQ. "A selector that knows which oversold ETF will bounce" = the
forward-return prediction the IC studies measured at ~0 — it can't be built.
**Active daily breakout/trend trading (Donchian/Darvas "buy the breakout, exit
when momentum dies") on ETFs** (params chosen pre-2018, OOS-tested): does NOT
beat buy&hold QQQ on return — it gives up half-to-two-thirds of CAGR (5–9% vs
11–21%) to whipsaws and cash-time — while cutting drawdown hard (−17/−29% vs
−35/−83%). Risk-adjusted it only edges buy&hold over a *full cycle containing a
catastrophe* (1999+ Calmar 0.20 vs 0.13, by dodging the −83% dot-com/GFC) and
*loses* even on Calmar in the bull-only test era (0.45–0.52 vs 0.59). This is
the Hurst-Ooi-Pedersen managed-futures result: trend-following is a
crash-protector/diversifier, NOT a return-beater vs a bull equity index — same
family as the Faber MA-gate and CSD: less drawdown, less return.
**Overnight/intraday decomposition (Knuteson; Lou-Polk-Skouras):** pre-2018 the
QQQ premium was *entirely overnight* (overnight-only 8.6%/−39% beat buy&hold
6.8%/−83%; intraday-only −11%/yr) — but it **decayed OOS** (post-2018
overnight-only 7.5% vs QQQ 20.9%); overnight-return momentum as a selector had
*negative* IC. Real, creative, arbitraged away.
**Cross-ASSET-CLASS dual-momentum switching (incl. BTC)** is the one rotation
that DID beat DCA-QQQ on return (15.7× vs 3.6×, 2015+) — because asset classes
have huge persistent divergent trends, unlike stocks/equity-ETFs. BUT it's
riding crypto (−71% DD ≈ BTC risk), a hindsight universe, no real OOS (sample =
the crypto bull), and it lost to buy&hold BTC. Verdict: trend-following works
*across asset classes* (Antonacci/Hurst-Pedersen) as risk-managed
diversification — but "beating QQQ" there = taking more risk via a higher-return
asset class, not alpha. *This is the legitimate home of "which to hold / when to
switch": asset-class allocation, not stock/ETF-equity selection.*
**Forward path (the real one): orthogonal, non-price, FREE data** — selection
edge is dead in price; the untested frontier is insider buying (SEC Form 4),
short interest (FINRA), institutional flow (13F), attention (Google
Trends/Wikipedia), and a FRED credit/curve macro-regime gate for timing. All
free, all orthogonal to price; expected modest and likely small/mid-cap-tilted,
but the first signals worth testing that aren't price in disguise.
**Free-data results (the forward path, partially executed):**
- *FRED/credit-spread regime gate* (HYG/IEF, since price-FRED is proxy-capped to
  3yr here): FAILED OOS — cut drawdown in only 12/27 assets, made QQQ's *worse*
  (−51% vs −35%), beat plain on Calmar 2/27, −8pp CAGR. Credit leads equities at
  big turns (GFC) but the daily signal whipsaws; the dumb 200-MA still beats it.
- *Asset-class DIVERSIFICATION + dual-momentum.* **The "beats QQQ on return"
  claim was UNIVERSE SELECTION BIAS — corrected.** It only beat QQQ when the
  hand-picked universe included Bitcoin (the era's best asset); handing a
  momentum rule the ex-post winner is circular, the same sin as cherry-picking
  winning stocks. Proof: on the broad NON-crypto universe (which included losers
  — EM, commodities, long bonds) dual-momentum did NOT beat QQQ on return (lower
  CAGR & Sharpe), only lower drawdown. Bias-free residual: diversification
  *reduces drawdown / modestly helps risk-adjusted* (a property of correlation
  <1, robust across universes); it does NOT raise return without hindsight-
  selecting the winning asset. *General rule: any backtest "beating QQQ" is
  selection bias unless the universe, rule, AND benchmark were all fixed without
  hindsight — beating it in-sample = overweighting what won = circular. Even
  "QQQ as the benchmark" is a hindsight choice. The only bias-free evidence is
  forward/out-of-sample, or robust mechanism-level claims (diversification lowers
  drawdown; cost & savings-rate dominate).*
- Short-interest free endpoint down (NASDAQ 503); insider Form-4 BUILT & tested
  (exp22): scraped 2,749 Form-4s for an objective 22-name liquidity-spread
  universe (mega->mid cap), parsed open-market purchases (code P) vs sales.
  Result = the most promising orthogonal lead found: at the **12-month horizon,
  insider-buy months beat no-buy months on forward return-vs-QQQ in BOTH OOS
  halves** (+0.8pp 2015-20, +10.5pp 2020-26) — directionally consistent, unlike
  every price signal (which sign-flipped). Matches Cohen-Malloy-Pomorski. BUT
  NOT established: tiny n (57-92 buy-months), 3m horizon flips sign, magnitude
  unstable (late +10.5pp likely a few beaten-down mid-caps), no significance
  test, large-cap-tilted (insiders sold ~$27B vs bought ~$1B here; the real edge
  is in small-caps we lack). Verdict: the one non-dead, theory-backed,
  orthogonal signal — worth a full small/mid-cap build + significance testing,
  the only lead that would justify more work.

**Entry-timing within a fixed DCA cadence (exp23 — "when to buy QQQ each cycle,
accepting QQQ"):** definitive — **buy immediately when the cash arrives; no
achievable trigger helps.** Across 2002-18 and 2018-26, every wait-for-a-dip
rule (down-day, below-cycle-start, 2% dip, sub-10d-MA, RSI<45) paid a HIGHER
average entry price and ended with EQUAL-or-LOWER terminal wealth than buying
on arrival. The "buy-the-dip beats immediate 73-82% of the time" is a
statistical illusion (many small wins, rare large losses when price runs away
and you're forced to deploy at cycle-end). Perfect-hindsight "buy the low each
cycle" is only +2-4% total over decades — the ceiling on ALL entry timing is
tiny. Shorter cadence + immediate deploy is best (monthly batching was worse
than biweekly: more uptrend drift missed while waiting). Optimized entry rule =
"the trigger is that the cash arrived" — buy biweekly on arrival.
**Qlib / multi-factor-ML (exp24 — Microsoft Qlib's flagship Alpha158-style
factors + LightGBM, walk-forward, NDX universe):** the ML approach from the
crypto multi-factor/order-flow papers, adapted to equities, FAILS. Cross-
sectional rank IC = +0.0018 (ICIR 0.013) — zero skill (Qlib gets ~0.03-0.05 on
retail-heavy CN markets; on US large-caps vs QQQ it's nothing). The apparent
"top-decile beats QQQ 23% vs 21%" was a BIAS MIRAGE: a RANDOM decile from the
same current-NDX names returned 25.7% and equal-weighting all of them 26.2% —
both beat QQQ by MORE than the ML model, which even underperformed random. The
"edge" is 100% survivorship (today's NDX = ex-post winners) + equal-weight tilt
vs cap-weighted QQQ, NOT the model. *Always run the random/equal-weight control
to expose this.* The library is fine; the alpha isn't there for US-large-cap-vs-
QQQ, and a long-short IC (if any) doesn't translate to a long-only index beat.
**Tactical ETF Sharpe ceiling (exp26):** a sustained Sharpe >3 OOS on ETFs is
arithmetically impossible — Fundamental Law IR≈IC×√breadth; ~10 ETFs × monthly =
√120≈11, and ETF-timing IC≈0, so even IC=0.05 caps IR≈0.55. Measured OOS ceiling
of legit tactical ETF books (diversification + inverse-vol/risk-parity + trend +
no-leverage vol-target) ≈ 0.8–0.85 Sharpe (best: inverse-vol 0.84, maxDD −18–23%
vs QQQ 0.79/−53%) — but they CUT CAGR to ~6–7% (from QQQ's 16%). Even perfect
monthly max-Sharpe MVO re-optimization gets only 1.06 OOS (1.35 in-sample — the
gap IS the overfit). Key insight for a NO-LEVERAGE investor: maximizing Sharpe
just buys a smoother, LOWER-return ride; high Sharpe only converts to high return
via leverage (lever the 8%-vol/0.84-Sharpe book up — Sharpe is leverage-
invariant). With leverage off the table, chasing Sharpe ≠ chasing return. Any
ETF backtest showing Sharpe ~3 is in-sample/lookahead — verify with the IS-vs-OOS
gap and the random/equal-weight control.
**Ernie Chan Kalman-filter PAIRS trading (exp27 — adaptive stat-arb):** faithful
implementation (dynamic hedge ratio via Kalman, trade forecast-error band), bug
caught (entry-day P&L mis-booking → lag positions), tested OOS net of cost.
Verdict: decayed. Most pairs that looked good IS flipped negative OOS (XLK/VGT
0.79→−0.81, IYR/VNQ 0.78→−0.31); only EWA/EWC survived (OOS Sharpe 0.66, still <
QQQ's 0.79, market-neutral ~1.7% CAGR, needs shorting); equal-weight book of all
pairs LOST OOS (−2.2% CAGR). Matches literature (pairs profits competed away
post-2003, Do-Faff). KEY LESSON: the Kalman filter is genuinely
dynamic/adaptive and works as designed — but adaptiveness CANNOT manufacture an
edge the market has arbitraged flat; it tracks a relationship faithfully, it
doesn't create alpha. Also requires shorting (margin) → off-limits anyway.
Near-theorem behind all of it: for a fixed savings stream into a positive-drift
asset, "invest immediately" (DCA) is near-optimal; every selection/timing/VA
scheme just *withholds or redirects exposure* to a rising asset, which costs
more than the cleverness gains. Honest return levers reduce to: more exposure
(leverage) or a different, less-efficient data universe.

Things that could change this verdict (untested here, would need new data):
analyst estimate-revision feeds, a paid PIT small/micro-cap dataset with
delistings, insider-transaction (Form 4) data, or genuine alt-data. Based on
the pattern above, bet conservatively even on those.

## Reusable tooling pattern
Build once, cache: a survivorship-clean union panel (`open/close/volume/member`
+ delisted backfill), a faithful signal port, a `run_dca`/rotation engine that
takes a precomputed score matrix + optional sell matrix, money-weighted
IRR/MOIC + flow-adjusted Sharpe/maxDD, and window scorecards at 1y/3y/5y/10y
for both train and test, vs QQQ **and** SPY. Then run tests #1–#10 above.
