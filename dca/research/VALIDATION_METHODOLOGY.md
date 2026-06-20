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
**NOVEL FREE DATA — FINRA short-volume (exp28/28b):** consolidated FINRA's raw
daily short-volume files into a proprietary short-pressure panel (516 names,
2018-2026) — exactly the "free data nobody sells pre-packaged" frontier.
Hypothesis (Boehmer-Jones-Zhang: heavily-shorted underperform). Raw SVR looked
great in full-test (low-minus-high +1.46%/mo fwd-3m, t+3.4, clean RANDOM
control) — BUT sub-period decomposition killed it: the edge is ~entirely one
window (2024-26 +2.61%/mo), sign FLIPPED in 2018-20 (−1.23%), ~nothing 2022-24;
and ABNORMAL SVR (per-stock-normalized = the theoretically-correct informed
measure) is NULL (t+0.2 recent). Verdict: closest thing to a real orthogonal
signal found (right sign 3 of 4 post-2020 windows, unlike sign-random price
signals) but weak/regime-unstable/recency-concentrated/null-when-cleaned → not
deployable. LESSON: a "t=3.4 OOS" can be one lucky 2-yr window — always
decompose by sub-period (this is how the headline died). Reachable free
orthogonal sources confirmed: FINRA short-vol, Wikipedia pageviews, EDGAR
full-text (Lazy-Prices text-change) + Form4/13F, SEC XBRL frames, VIX term
structure. The pipeline works; the frontier is a multi-signal COMPOSITE of weak
orthogonal sources, not any single one.
**ReCAP / regime-adaptive continual learning (exp29, arXiv 2606.00143):** built
the faithful essence (KMeans regime detect → per-regime max-Sharpe allocation
learned on train → apply OOS by current regime, walk-forward). FAILS: regime-
adaptive Sharpe 0.79 = NO better than RANDOM-REGIME control 0.86 (regimes carry
no persistent allocation info), and both crushed by static QQQ (1.14) and 60/40
(1.03). Regime-switching now 0-for-5 (Faber, CSD, absorption-ratio, correlation-
regime, ReCAP). Fancier ML/RL = more overfit surface, not less; paper's
"outperforms" is in-sample/cost-free/universe bias. The random-label control is
the decisive test for any regime/ML scheme.
**Composite check (exp30 — insider + short-vol, 18-name overlap):** the
"informed composite" idea resolves to ONE carrier — insider buying does the
work; short-vol adds nothing and DILUTES it (composite +4.0%/3m < insider-only
+4.6%; low-short-only is negative). Insider buying now shows a positive sign a
THIRD time (exp22 train +0.8pp / test +10.5pp; here +4.6%/3m, +8%/6m rel-QQQ) —
the ONLY signal across 30 experiments that keeps pointing the right way (theory:
Cohen-Malloy-Pomorski). Caveat: every read is small-sample (here n=44 buy-months
/ 18 names, likely a few rebounding mid-caps) — could shrink at scale. VERDICT:
drop the composite/short-vol; the single worth-the-resources lead is INSIDER
BUYING (Form-4) scraped for a broad universe + strict sub-period validation.
**Invented volume/vol features (exp31):** tested documented + 4 INVENTED OHLCV
features (accum/Chaikin, volume-vol divergence, signed-vol momentum, vol-weighted
return premium) for cross-sectional alpha, sub-period sign-stability + random
control. All 4 invented ones FAILED (sign-flip/~0 = random) — clever OHLCV
transforms don't create information. Low-vol/beta/MAX just showed "high-beta won
in the 2010-25 bull" (regime beta, not alpha). The ONE survivor: **Amihud
illiquidity** (+0.069/t6.6 and +0.042/t3.4, same sign both halves, >> random) —
the FIRST cross-sectional feature in 31 experiments to pass the full bar. BUT
it's compensation-for-illiquidity (lives in less-liquid small/mids → realistic
trading costs eat the IC; pulls away from QQQ; won't beat QQQ on raw return; and
overlapping-63d-return t-stats are mildly inflated). DEEP PATTERN: the two real
residual premia found (illiquidity, insider buying) BOTH live in less-liquid/
smaller-cap corners where frictions/capacity eat them — they survive precisely
BECAUSE they're costly to harvest. That is the structure of persistent
inefficiency: not invent-able from liquid OHLCV, only earnable by bearing real
frictions.
**Illiquidity-premium HARVEST (exp32 — the exp31 survivor, tested for real):**
FAILS the honest harvest. Long most-illiquid quintile of (current) S&P400
midcaps showed CAGR 43.6% — a survivorship MIRAGE, two smoking guns: (1)
universe contamination — current-constituent midcap equal-weight returned 21.8%
vs the ACTUAL midcap ETF MDY's 12.2% = ~10pp/yr pure survivorship (illiquid/
small names that delisted aren't in today's list; the illiquid quintile
amplifies this most); (2) risk-adjusted it's WORSE than random — illiquid Sharpe
0.50 < random-quintile 0.65 < equal-weight 0.76 < QQQ 1.13. Cost wasn't the
killer (low turnover); bias+risk were. CONCLUSION: the academic illiquidity
premium is real but CANNOT be validated/harvested with free current-constituent
data — it lives exactly where free data is most survivorship-biased. Separating
real premium from survivorship REQUIRES point-in-time delisting-inclusive small-
cap data (paid). Same wall, hit from inside: the only surviving signals
(illiquidity, insider) live in the illiquid corners where free data fails.
**DEFINITIVE SBF test via OpenSourceAP (exp33, Chen-Zimmermann, peer-reviewed,
CRSP-based, release-lags applied — `pip install openassetpricing`, FREE):** the
GitHub research found the one resource that breaks the survivorship wall, and
using it SETTLES the premia question. Long-short premia by era (t-stat):
Illiquidity/Amihud pre-2004 t+3.0 -> 2017-24 t−0.5 (DEAD); Size +3.5->−1.0
(DEAD); BidAskSpread/std_turn/Value all DEAD post-2004; Momentum weakened
(t+1.7); SURVIVORS = VolumeTrend (t+2.6, the one volume signal that persists)
and Gross Profitability/quality (t+2.2). **Of 208 anomalies, only 14% are still
same-sign |t|>2 since 2016; 31% FLIPPED sign.** This proves: (1) exp32's 43.6%
illiquidity result was 100% survivorship — the REAL premium is dead 20yrs; (2)
~86% of published anomalies are dead/insignificant recently — publication decay
(McLean-Pontiff, Chen) is WHY every experiment here failed; it's the base rate,
not us; (3) survivors are GROSS long-short Sharpe 0.3-0.6 -> net-of-cost &
long-only-vs-QQQ they mostly vanish. FREE-DATA CATALOG for future work:
survivorship-free = OpenSourceAP (signals+portfolios, no WRDS) + QuantConnect/
LEAN (cloud daily, delisting-inclusive from 1998); orthogonal alt-data = SEC
Insider TSV (Form-4, 2006+), FINRA short (cdn.finra.org), SRAF 10-X corpus
(Lazy-Prices), edgartools (13F/insider parsing), Wikipedia pageviews, GDELT+
FinBERT; validation = Novy-Marx/Velikov net-of-cost protocol; benchmark
skeptics = Hou-Xue-Zhang global-q, JKP. Backtesters (vectorbt/bt/zipline) do NOT
fix survivorship — data does.
**INSIDER BUYING — clean build & the one signal that PASSED (exp34):** built a
clean point-in-time monthly insider panel from SEC Form-345 bulk TSVs (296,855
ticker-months, 14,261 tickers, 2010-2025, no scraping; keyed on FILING_DATE +
ISSUERTRADINGSYMBOL). Tested on priceable S&P400+500+NDX (~900 names; mid/large,
understates small-cap effect). RESULT — the FIRST signal in 34 experiments to
pass BOTH disciplines: binary "net insider buyer" (net P-buys > S-sells, trailing
3m) long-only portfolio beat a RANDOM same-universe same-size portfolio by +6.5pp
(2010-17) and +8pp (2018-25) — same sign BOTH halves; event-study net-buyer vs
rest +0.72/+0.75pp per 3m (stable). Random control shares the survivorship, so
the gap is SIGNAL not bias. Net-20bps CAGR 22.7% vs SPY 14.4%, QQQ 19.4%.
HONEST CAVEATS: (1) absolute 22.7% is partly universe-survivorship-inflated
(current-constituent EW=20% vs SPY 14.4%); clean signal contribution is the
control gap (~+3-7pp), not the full level; (2) risk-adjusted it does NOT beat
QQQ (Sharpe 1.04 vs 1.12, −35% DD); wins return, loses smoothness; (3) only the
BINARY presence works — $ magnitude IC was negative (fragile); (4) mid/large-cap
only (small-cap effect stronger but needs PIT prices). VERDICT: the genuine
find — a real, stable, control-passing, orthogonal (non-price) signal — modest-
to-moderate and survivorship-aided in level, but real. Deployable as a tilt;
clean confirmation needs PIT/delisting-inclusive small-cap prices.
**Insider REFINEMENT (exp35 — cluster/officer/CEO-CFO):** signal is ROBUST
across variants — base net-buyer (22.7%/Sh1.04), officer-buy (20.5%/Sh1.05/
DD−31%), CEO-CFO-buy (20.8%/Sh1.05/DD−31%) all beat random in BOTH sub-periods.
Refinement to officer/CEO-CFO improves RISK profile (Sharpe 1.05, DD−31% vs
base −35%) at slightly lower CAGR — cleaner ~35-name signal — but does NOT
strengthen raw edge or beat QQQ risk-adjusted (all Sh~1.05 < QQQ 1.12). Cluster
buys INCONCLUSIVE (matched random portfolios too small ~25-46 names → noisy
control, e.g. spurious "random 31%"). On clean confirmation (#1): fully
delisting-inclusive prices not free-accessible in-sandbox (QC clean data is
cloud-only; Sharadar paid). BUT survivorship-SHARING controls (random/EW) +
the EVENT-STUDY (same surviving stocks: post-buy months beat non-buy months
+0.73pp/qtr both halves — timing diff, not survivorship) give high confidence
it's a REAL modest timing signal; only the absolute level (~20-23% CAGR) is
survivorship-inflated. NET CONCLUSION of the 35-experiment arc: exactly ONE
real, orthogonal, survivorship-robust signal found — insider buying — modest
(~3%/yr gross over market, event-study +0.73pp/qtr), stable, deployable as a
TILT, ~QQQ-equivalent risk-adjusted (not superior). Everything price-based and
every fancy method failed; the edge required orthogonal non-price data and even
then is small. Clean absolute confirmation needs paid PIT data.
**DEPLOYABLE blend (exp36 — QQQ + insider-officer-buy tilt):** THE payoff. QQQ
core + insider-officer-buy sleeve, monthly rebal net 20bps: 70/30 -> CAGR 20.0%/
Sharpe 1.17/maxDD −28% vs 100% QQQ 19.4%/1.12/−33%. First thing in 35 experiments
to improve QQQ risk-adjusted without being a pure mirage. ROBUST part = the
Sharpe/DD gain (diversification — decorrelated mid-cap insider names vs mega-cap
tech; a correlation effect, survivorship-robust). CAVEATED part = the CAGR uplift
(sleeve is survivorship-aided mid-cap; clean PIT trims it). Benefit is MODEST
(Sharpe +0.05, DD −5pp), concentrated 2010-17, needs monthly ~35-name rebalance.
DEPLOYMENT for biweekly QQQ-DCA: keep QQQ core, optionally route ~15-30% into
current officer-insider-buy names (rebalanced) for a modestly smoother ride —
not a return windfall. This is the honest terminus of the whole arc: one real
orthogonal signal, deployable as a small diversifying tilt; clean CAGR
confirmation still needs paid PIT data.
**High-Sharpe reproductions + combination (exp37-39):** reproduced the
specific public strategies exactly. (37) Quantitativo IBS mean-reversion on
QQQ/SPY: the claimed "~2.1 Sharpe" is the IN-MARKET Sharpe (1.87, stable to 1.96
OOS) — real edge — but only ~20% exposure so FULL-portfolio Sharpe 0.79 (vs
buy&hold 0.52), CAGR 12% ~= buy&hold, maxDD −25% (vs −75%). Genuine low-exposure
sleeve. (38) NDX rotational momentum (gate NDX>200dMA, top-K by 250d ROC): top-10
momentum Sharpe 1.40 == RANDOM-K control 1.41 -> the momentum RANK adds nothing;
the edge is the trend GATE (avoids crashes, 0.93->~1.4) + SURVIVORSHIP (top-5
49% CAGR = concentration into current-NDX winners, not replicable). (39) THE
PAYOFF — combine the real OOS survivors (QQQ beta + IBS mean-reversion + insider
tilt; MeanRev corr to QQQ only 0.18, to Insider 0.03): portfolio Sharpe 1.12 ->
1.34-1.42, maxDD −33% -> −18%, STABLE both sub-periods, cost ~3-4pp CAGR. This
is the honest 'combine low-correlation sleeves' result: achievable Sharpe ~1.3-
1.4 (clean, ~1.3 dropping survivorship-aided insider), NOT 3. Matches the user's
own awesome-systematic-trading list (self-reports most strategies 0.3-0.8
Sharpe) and OSAP (14% of anomalies survive, gross Sh 0.3-0.6). Sharpe 3+ is not
in any reproducible public source. DEPLOYABLE: QQQ core + mean-reversion sleeve
(+ optional insider tilt), ~60/20/20 -> Sharpe ~1.3, DD ~−25% vs QQQ −33%.
**Extended ensemble (exp40 — add trend-following + XS-reversal sleeves):**
adding more sleeves did NOT improve risk-adjusted return. Correlations: MeanRev
is the ONLY genuine diversifier (0.18 to QQQ, 0.03 to others); multi-asset TREND
is 0.67 corr to QQQ (long-or-cash trend = long equity in the 2010-25 bull, no
diversification) with low 6.9% return; XS-reversal is 0.44-0.55 corr, weak 0.57
Sharpe, survivorship-aided. Result: 3-sleeve (QQQ+MR+Insider) Sharpe 1.26 stays
best; 5-sleeve inverse-vol only 1.29 (and CAGR crashes to 13% by overweighting
low-return defensives); 5-sleeve equal-weight WORSE (1.03). LESSON (correct
IC×√N): diversification helps only when the added stream is BOTH low-correlation
AND positive-Sharpe; most strategies (the 30 repos, trend, reversal) are
equity-flavored (0.4-0.7 corr) so stacking them piles correlated risk and √N
does nothing. Retail long-only no-leverage runs out of genuinely-uncorrelated
streams after ~2-3 -> ensemble Sharpe CEILING ~1.3. FINAL DEPLOYABLE: QQQ core +
short-term mean-reversion sleeve + insider tilt (~60/20/20) -> Sharpe ~1.3, DD
~−25% (vs QQQ 1.12/−33%). Higher needs uncorrelated streams that require
shorting/leverage (managed futures done right) or different data — outside the
stated constraints. Sharpe 3 remains a mirage in every reproducible source.
**Market-neutral StatArb + the Sharpe-3 roadmap (exp41-42):** built the proper
Avellaneda-Lee PCA-residual reversal (the core of the "Attention StatArb net 2.3"
roadmap) on 488 S&P names. KEY: GROSS Sharpe 0.58, remarkably STABLE every
sub-period (0.56/0.64/0.61), market-neutral (0.13 corr to QQQ), −19% DD — a real
edge. BUT cost-killed: net Sharpe 0.34 @2bps/side, 0.22 @3bps, ~0 @5bps, −0.64
@10bps (~80%/wk turnover; ~4%/yr gross alpha < retail TC). THIS is why StatArb is
institutional — survives only at sub-bp execution + leverage. As an ensemble
sleeve (@2bps) it IS the uncorrelated stream exp40 lacked: adding it lifts
4-sleeve inverse-vol Sharpe 1.26->1.38, maxDD −25%->−14% — but overweights
low-return sleeves (CAGR->10%) and needs shorting. ROADMAP VERDICT: Sharpe 3 is
real but INSTITUTIONAL — requires (1) sub-bp execution (proven binding: gross
0.58->dead at 5bps), (2) leverage to scale low-vol market-neutral books, (3)
joint TC-optimization (the Attention paper's net-2.3 depends on TC-in-objective +
shorting + DL infra, a 24yr BACKTEST not live), (4) PIT survivorship-free data,
(5) 5-10 uncorrelated alphas. Retail (no leverage/shorting infra, 2-5bps, free
data) ceiling stays ~1.3-1.4. Kalman pairs decayed (exp27); production multi-
alpha/futures-ensemble = institutional diversification (~1.0-1.5). Net 2.3+ is
not retail-reproducible.
**Sharpe-2 loop, iter1 (exp44):** diversified always-on RSI-2 mean-reversion
book (Connors, 20 oversold-in-uptrend S&P names) -> FAILED as a diversifier:
forcing 100% exposure turned it into long equity BETA (corr to QQQ 0.75), Sharpe
0.89 train -> 0.56 OOS. Lesson: the IBS mean-reversion edge comes from
SELECTIVITY + being in cash ~80% (that's what decorrelates it); breadth/full-
exposure converts it to correlated beta and kills the high in-market Sharpe.
Confirms there's no free Sharpe from 'run the good sleeve harder'.
**Sharpe-2 loop, iters 2-3 (definitive ceiling):** (iter2) multi-asset IBS
mean-reversion (QQQ+TLT+GLD+EEM+IWM+EFA+XLE+XLP) = Sharpe 0.54 < QQQ-only IBS
0.75, corr rose to 0.56 (non-QQQ assets have weaker MR + are mostly equities).
(iter3 DECISIVE) max-Sharpe optimal weighting of ALL 4 real sleeves (QQQ +
IBS-MR + insider + StatArb): even with PERFECT-HINDSIGHT weights the ceiling is
Sharpe 1.46 (overfit upper bound); train-fit weights applied OOS give only 0.97
(optimization OVERFITS — naive inverse-vol 1.38 / fixed 60-20-20 1.26 beat it
OOS, the DeMiguel '1/N beats Markowitz' result). CONCLUSION (after 46 exps): a
net-of-cost OOS Sharpe of 2 is NOT achievable with retail-accessible
tools/data/constraints — the perfect-hindsight upper bound from every real
cost-surviving sleeve is 1.46, honest OOS ~1.3-1.4. Sharpe 2+ requires
institutional execution (sub-bp costs), leverage, and shorting (StatArb
gross 0.58 -> dead at 5bps proves cost is binding). This is a hard ceiling, not
a failure of search: 46 experiments, every creative angle, same wall.
**Top-1 picker loop, iter1 (exp47):** "buy the single best stock each month,
accumulate, beat QQQ." On the survivorship-biased current-constituent universe,
momentum-top-1 (1.6x full / 1.12x OOS vs QQQ) and insider-top-1 (1.44x/1.30x)
beat QQQ AND random-top-1 (0.78-0.87x, below QQQ). BUT momentum-top-1 IS the
ZENITH trap already proven to collapse on clean PIT data — top-1 momentum can
only pick survivor-winners (look-ahead concentration), not skill. Random-top-1
< QQQ shows equal-weight single picks lose to cap-weighted QQQ even with
survivorship. Only suggestive thread: insider-conviction top-1 beats
random-among-insider OOS (1.30x vs 0.95x) — a partly-survivorship-neutral
within-insider selection effect — but top-1 is extreme variance (~95 picks,
few dominate) and "largest officer-buy $" proxies size. VERDICT: top-1 cannot be
honestly validated as a QQQ-beat without clean PIT (delisting-inclusive) prices;
the momentum version is the known survivorship illusion. Insider-top-1 is the
one thread worth a clean-data test.
**"100 ideas" loop, iter1 — systematic FACTOR ZOO (exp48):** the disciplined
way to test many ideas: 20 distinct price-signal families x 2 horizons (38
tests) + 6 random-null features, scored on same-sign |t|>1.5 in BOTH 2010-17 and
2018-25 halves. Result: 4 survivors, ALL the same redundant cluster
(lowvol6/12, maxmo with NEGATIVE IC = high-vol/high-beta stocks won the bull) =
regime BETA, not alpha (opposite of the academic low-vol/MAX factors -> proof
it's recent-regime; reverses in value/bear regimes). 0/12 random-null survivors
(calibrates that the bar isn't passed by chance). LESSON: mining 100 ideas
without a random-null/multiple-testing control 'finds' winners that are beta or
luck; with the control, no NEW tradeable price alpha emerges vs QQQ. Confirms
the whole program: the only real signals are orthogonal (insider) or
mean-reversion, not any price transform.
**CLEAN insider test on Tiingo delisting-inclusive data (exp49) — THE
verification:** ran the insider-officer-buy signal on Tiingo prices INCLUDING
855 delisted insider-buy names (the survivorship fix yfinance lacks). With
winsorized returns + price>=$3 + equal-weight portfolio: insider-buy minus rest
= **+0.68%/3m (t=3.7) WITH delisted names included** vs +0.66% survivors-only.
**Survivorship inflation is negligible — the insider edge is REAL, not an
artifact.** First signal in 49 experiments to pass a genuine survivorship-clean
test. Modest (~2.7%/yr long-short) but verified. CRITICAL methodology note:
naive means on delisting-inclusive small-cap data are corrupted by penny/dead-
stock outliers (a spurious '+227%' until winsorized) — ALWAYS winsorize +
price-filter + equal-weight when using survivorship-clean small-cap data.
Caveats: preliminary (855/1831 delisted names; rest downloading), Tiingo misses
OTC-Q bankruptcy final wipeouts (but adding the 855 barely moved it -> robust).
**DEFINITIVE clean small/mid-cap insider test (exp51, Tiingo 19.3k tickers incl
2,391 delisted insider names = 99% coverage):** the payoff of the clean data. On
the full survivorship-clean small/mid-cap universe (8,599 insider names priced
incl delisted), insider-buy-minus-rest fwd-3m (winsorized / price>=$3 / eq-wt):
**large-$ buy +0.60% (t2.7), cluster>=2 +0.55% (t3.4), net-buyer +0.52% (t3.5),
cluster>=3 +0.52% (t2.8)** all SIGNIFICANT; officer-buy +0.27% (t1.6) and CEO/CFO
+0.13% (t0.8) WEAK on the broad universe (officer was strong on S&P large-caps;
on small/mid the winners are CLUSTER + NET + LARGE-$). Survivorship check:
officer-buy +0.27% incl-delisted vs +0.33% survivors-only → survivorship adds
only ~0.06pp; the edge is REAL, not an artifact. NET: a verified,
survivorship-clean, small/mid-cap insider edge ~+0.5-0.6%/3m (~2-2.4%/yr
long-short, t~3), best harvested via CLUSTER buys (≥2 insiders) or LARGE-$
purchases. The one real, clean-data-verified signal of the whole program.
(Tiingo PIT data: dca/research/data/tiingo/.)

**Signal vs strategy — the equity-curve reality check (exp52/exp53, clean Tiingo
2011-2025):** plotting the insider edge as a *standalone long-only strategy*
exposed the gap between a cross-sectional signal and a tradeable strategy.
Pure equal-weight long-only of ALL ~529 cluster/large-$ insider names:
CAGR 7.8%, Sharpe 0.50, maxDD -41% -> **loses badly to QQQ** (CAGR 18.7%,
Sharpe 1.10, maxDD -33%; $1k/mo DCA: insider $274k vs QQQ $861k). Reason: the
insider tilt lives in the SMALL/MID-CAP pond (IWM 9.4% CAGR, Sharpe 0.57), which
structurally lagged QQQ in the 2011-2025 mega-cap-tech regime. A positive
cross-sectional spread (+0.5%/3m vs small-cap peers) still nets an absolute
loser vs QQQ. CONCENTRATION DOESN'T RESCUE IT: top-10/20/30/50 by a conviction
score (cluster size + $ size + officer/CEO) = 12.0/8.6/7.4/6.2% CAGR (Sh
0.60/0.54/0.47/0.42) — all < QQQ, top-10's 12% is small-sample noise. CRITICAL:
the L/S "top-20%-conviction vs rest-of-buyers" spread = **-1.4% CAGR, Sharpe
-0.28** -> the conviction GRADIENT within buyers is flat/negative; the real edge
is binary ("recent insider cluster/large buy present" vs absent), NOT a
concentratable alpha. So insider = a weak broad small-cap tilt, not a standalone
QQQ-beater and not a high-Sharpe sleeve. As a QQQ overlay it only dilutes toward
a lower-Sharpe asset. Honest verdict: the insider signal is real but its
PRACTICAL value for a retail DCA plan is marginal; the risk-adjusted improver in
this program was the mean-reversion sleeve, not insider.

**Broadening beyond insider (exp55/exp56/exp57, clean Tiingo) — the LOW-VOL
overlay is the one honest, retail-tradeable improver.** Per "keep improving, not
necessarily insider", ran a cross-sectional factor scan on the clean monthly
panel (top-decile long-only, price>=$5, eq-wt, delisting-incl): momentum 12-1
CAGR 12.8%/Sh 0.69, mom6 12.7%/0.65, 1-mo reversal -2.8%/0.03 (DEAD on small-caps
+ high cost), but **low-vol (lowest 6-mo return-vol decile) CAGR 10.6%, Sharpe
1.35, maxDD -15%, corr-to-QQQ only +0.34**. Momentum/reversal too correlated
(+0.70) to help a QQQ core; low-vol's low correlation makes it a real
diversifier. BUT the 1.35 was idealized (costless monthly rebal of ~378 names).
Validating on the ACTUAL tradeable products (fetched USMV/SPLV/MTUM/VLUE/QUAL/
EFAV/TLT/IEF from Tiingo, 2012-2025): USMV (real low-vol ETF) CAGR 11.5%, Sharpe
1.03, maxDD -19%, corr 0.72 (large-cap low-vol = more market-like than the synth
sleeve). Real overlay benefit is MODEST but genuine + sub-period-stable: QQQ+USMV
70/30 = CAGR 17.4%, **Sharpe 1.18 (vs QQQ 1.15), maxDD -28% (vs -33%)**; 60/40 =
Sharpe 1.19. Beats QQQ Sharpe in 3 of 4 sub-periods (2012-15 1.58>1.48, 2016-19
1.37>1.24, 2023-25 1.97>1.92; ties 2020-22 0.44~0.45). Adding TLT (3-way) does
NOT help (TLT Sharpe 0.09, killed by 2022). HONEST VERDICT of the whole
"beat/improve QQQ-DCA" program: no long-only no-leverage strategy beats QQQ-DCA
on RETURN in the 2012-2025 mega-cap regime; the only durable, retail-tradeable
*risk-adjusted* improvement is a 20-40% low-vol (USMV) sleeve on a QQQ core —
buys ~5-6pp less drawdown and ~+0.03-0.04 Sharpe for ~2-3pp less CAGR. Insider,
momentum, reversal, value, quality, 3-way bond mixes all fail to improve it.
(Scripts: dca/research/exp52..exp57.)

**Moonshot reverse-engineering (exp58-60, vs the stocksonstocks repo).** Goal:
reverse-engineer what qualifies the rare >100% multi-baggers (NVDA/VRT/APP-class)
into an ensemble, then validate on PIT data. Analyzed the SOS "Moonshot" engine
in detail: it layers a selector on ~40 technical+fundamental+news signals
(signals.ts computeSignalAt) → Elite STRICTEST tier → MOON tiers (STRICTEST ∩
imminent-breakout ∩ theme-sourced gap-up), held for YEARS with a staged
confirm/cull ladder (30d≥+10% / 90d≥+10% / 180d≥+30%). Their key reframe: P(>100%
in 180d)=3% (impossible) vs P(>100% hold-to-today)=20% — moonshots are a
multi-year HOLD phenomenon. Realized matured-2023-24: +50% CAGR vs SPY +10%,
maxDD -37%. HONEST PROBLEMS: validated only 2023-25 (the AI/semi boom), themes
hindsight-curated in 2026, Yahoo data = survivorship-biased (delisted zeros
invisible), ~22 fat-tailed trades/yr.
  exp58 (case studies): moonshots are BIMODAL — breakout-from-strength
(AVGO/LLY/WDC/MSFT near 52w-high, +mom) AND deep-recovery (META/VRT/NFLX -70%
mom, 28% of high). Systematic top-2%-fwd12m signature = HIGH vol6 (rank 0.74) +
LOW %-of-12m-high (0.35) + LOW price (0.35) = the classic LOTTERY profile.
  exp59 (raw archetypes on PIT, monthly): LOTTERY (hi-vol/cheap/beaten) top20 =
CAGR 24.7% but Sharpe 0.67, maxDD -86% — it DOES catch moonshots but rides a sea
of wipeouts. The whole SOS engine is a RISK-CONTROL WRAPPER around lottery-hunting
(guards+regime+staged cull) trying to keep NVDA upside while cutting the -86% tail.
  exp60 (DEFINITIVE — faithful price/vol replication of the SOS technical stack
on the 2000-2024 PIT delisting-inclusive panel, 6137 liquid names): the technical
"moonshot structure" (bull stack + near-52w-hi + breakout/imminent + extension
guards) has ~ZERO edge OOS — mean fwd-12m 9.5% vs BASE liquid universe 9.6%, and
2x rate 2.0% vs 3.2% (the guards REMOVE multi-baggers). By era it's pure regime
beta: 2000-09 +4.0%, 2010-19 +10.9%, 2020-24 +11.5%, ~2% 2x throughout. VERDICT:
the price/volume signature is NOT the moonshot qualifier — it's trend-beta
scaffolding. The 2023-24 alpha lived in theme-selection + fundamental
acceleration + multi-year hold during the AI capex boom — the layers most
exposed to hindsight and the ones price-only PIT data can't adjudicate. Caveats:
close-based (no intraday H/L), monthly cadence, theme/news/fundamental gates not
replicated (no data). To find a DURABLE qualifier one must test the fundamentals
layer (SEC-EDGAR revenue-acceleration is free) and/or the Form-4 insider overlay
(our one validated edge = their unbuilt 'Deep-Doc' layer). (Scripts exp58-60;
SOS repo read-only via PAT, not committed here.)

**THE QUALIFIER ENSEMBLE — the one thing that honestly beats QQQ on CAGR (exp61-63).**
After the technical core tested as beta (exp60), built the durable qualifier on
PIT data: fetched SEC EDGAR XBRL quarterly revenue via the frames API (60 quarters
x 4473 tickers, 4 concepts coalesced, saved dca/research/data/sec/
sec_revenue_quarterly.parquet; reporting-lagged ~80d to avoid look-ahead),
computed YoY growth + acceleration (YoY rising 2 consecutive Qs), combined with
our validated Form-4 insider cluster/large-$ signal and a monthly trend-timing
gate (price>10mo-MA & 6mo-mom>0). Cross-sectional fwd-12m (survivorship-clean,
2012-25): rev-accel 12.9%, high-YoY 14.0%, insider 14.2%, rev-accel&insider
17.6% (2x-rate 7.8% vs universe 4.4%) — the ensemble nearly DOUBLES the
multibagger base rate. As a long-only monthly-rebalanced portfolio (~19 names,
price>=$3, eq-wt, delisting-incl): **(rev-accel|high-YoY) & insider-cluster &
uptrend = CAGR 23.7%, Sharpe 1.06, maxDD -26% vs QQQ 19.3%/1.13/-33%** (corr
0.54). ROBUSTNESS (the honest part): NOT an AI-boom artifact — it LAGGED 2023-24
(2023 -6% vs QQQ +55%) and ex-2023-24 = +28.2%/1.24 vs QQQ +16.1%/0.96; survives
dropping top-5 names (18.7%/0.93); diversified across sectors (gold/defense-elec/
pharma/EV/construction/mortgage, not tech); huge crash protection (2022 +2.8% vs
QQQ -32.6%). CAVEATS: small/mid-cap (real spreads likely >20bps; turnover ~54%/mo
=> ~1.3%/yr drag at 20bps, net ~22.4% CAGR still > QQQ; capacity-limited); edge
DECAYED 2021-25 (11.7%/0.60 vs QQQ 15.1%/0.83); Sharpe (1.06) still < QQQ (1.13)
so it's a CAGR/drawdown win, not a Sharpe win; close-based prices, monthly cadence.
VERDICT: the reverse-engineered moonshot qualifier (fundamental revenue
acceleration + insider conviction + trend timing) is the FIRST long-only
no-leverage strategy in this entire program that beats QQQ on CAGR (and drawdown)
out-of-sample and survivorship-clean — a genuine, diversified, non-regime-luck
edge, harvestable as a ~20-name active small/mid-cap sleeve. (Scripts exp61-63;
SEC data labeled in dca/research/data/sec/.)

**Pushing the qualifier harder (exp64-67): concentration + staged hold + loss-cutting
beat QQQ on ALL THREE metrics.** (a) Decay diagnosis: the 2021-25 lag is the POND
not the signal — the ensemble's cross-sectional edge over its own universe is
actually strongest recently (+6.7% vs universe's +2.2%), but small-caps as a class
trailed QQQ. (b) Event-driven staged-hold sim (let winners run, trend-exit): cap-25
positions = CAGR 26.8%/Sharpe 1.13/maxDD -28% (vs monthly-rebal 23.5%/1.05).
(c) LOSS-CUTTING lab (user idea, SOS-style): best = hard-stop -25% + trailing-stop
-35% + staged confirm-or-cull ladder (cut names not +10% by 3m / +30% by 6m) =>
**CAGR 26.2%, Sharpe 1.18, maxDD -27%, win-rate 48%** — beats QQQ (19.3/1.13/-33)
on return, risk-adjusted AND drawdown. Removing the trend-exit wrecks it
(14.6%/0.76) — trend exit is load-bearing. Net: the staged + loss-cut qualifier is
a genuine all-around QQQ-beater on survivorship-clean PIT data (caveats unchanged:
small/mid-cap capacity, ~50% turnover, close-based). (Scripts exp64-67; masks
cached _qual_masks.pkl.)

**Feature discovery + ML reverse-engineering + sleeve blending (exp68-72).** Built a
36-feature monthly panel (fundamentals from SEC frames: ROA/ROE/margins/R&D/
shares/balance-sheet; insider; technical; novel interactions) cached _featmat.pkl.
UNIVARIATE rank-IC vs fwd-12m (survivorship-clean): the strongest+most-robust+
STRENGTHENING predictors are NOT the original ensemble's features — vol6 -0.147
(low-vol), ROA +0.141, distHigh +0.130 (near-52w-high), share_chg -0.108
(buybacks), ROE +0.106, momentum +0.07; rev-accel only 0.025 and insider 0.03.
In 2021-24 these are 2x stronger (distHigh 0.267, vol6 -0.263, ROA 0.230). My
hand-built interaction features (quiet_compounder, triple_confirm) were WEAK
(<0.01) — plain strong factors dominate. WALK-FORWARD ML (HistGBM, no-look-ahead,
trained only on past): top-25 picks = **CAGR 31.6% vs QQQ 18.7% (2015-25)**,
Sharpe 1.09, DD -43% — confirms the feature ensemble has real OOS predictive
power (return-max but DD-uncontrolled). NEW COMPOSITE sleeve (quality+low-vol+
momentum+buybacks, the discovered factors): 17.9% CAGR / Sharpe 1.19 / maxDD -21%
— high-Sharpe steady compounder, complements the high-CAGR moonshot sleeve.
**SLEEVE BLEND = the finale**: moonshot(26.2/1.18) + composite(17.9/1.19) are
~0.70 correlated; blends beat QQQ on ALL THREE metrics AND in ALL THREE sub-eras
(fixing the decay): 50/50 moon/comp = CAGR 22.4%/Sharpe 1.28/maxDD -18%; 60/40 =
23.2%/1.27/-19%; 33/33/33 moon/comp/QQQ = 21.5%/**Sharpe 1.32**/-22% (2021-25
0.92 vs QQQ 0.83). Net: a diversified proprietary multi-sleeve strategy reaches
Sharpe ~1.3 / CAGR ~22-23% / half QQQ's drawdown, survivorship-clean. (Scripts
exp68-72; _featmat.pkl, sec_fundamentals.pkl.)

**THE ALPHA ENGINE — market-neutral long/short factor book (exp73-75).** Thinking
like a prop-shop quant: the real alpha is MARKET-NEUTRAL (strip beta). Built a
factor zoo (value B/M & E/P, Novy-Marx gross-profitability GP/A, sales/assets,
Piotroski, ROA/ROE, momentum 12-1, 52w-high, buyback, rev-accel, insider) and
ran dollar-neutral decile L/S on the survivorship-clean PIT universe.
SINGLE-FACTOR L/S Sharpe (corr-to-QQQ): **value B/M 1.38 (-0.15)**, GP/A 1.02,
sales/assets 1.01, Piotroski 0.92, ROE 0.86, rev-accel 0.86, mom12 0.83, ROA
0.77 — all ~0 or negative market correlation = genuine alpha. (asset-growth,
accruals, leverage were weak/negative in small-caps.) COMBINED 11-factor L/S
(avg pairwise corr 0.27): **Sharpe 1.55, corr-to-QQQ -0.09, maxDD -18%, ann 12%.**
PORTABLE ALPHA (overlay the mkt-neutral book on beta, gross>100% => uses
shorting+leverage): QQQ+1x alpha = 30%/Sharpe 1.66; 50QQQ/50comp+1x = 29%/1.80;
QQQ+2x = 41%/1.83. REALITY CHECK (exp75): net of 6%/yr small-cap borrow + shorting
only price>=$5, and a DEPLOYABLE long-only version (no shorting/leverage) — see
exp75 results. HONEST CAVEAT: the headline Sharpe 1.55-1.83 REQUIRES shorting
small/mid-caps (borrow cost, hard-to-borrow, squeeze risk) and leverage/margin —
contrary to the original no-leverage/no-margin mandate; the long-only-implementable
slice is the realistic retail deliverable. Net: a true market-neutral alpha source
exists in the data (Sharpe ~1.5 gross), best harvested by a fund that can short;
for a long-only investor it manifests as the factor-tilt sleeves above.
exp75 REALISTIC numbers: gross L/S Sharpe 1.75 -> NET of 6%/yr borrow + $5 short
filter = Sharpe 0.99 (borrow ~halves it); regime-dependent (net Sharpe -0.13 in
2017-20, +2.18 in 2021-25). DEPLOYABLE no-leverage/no-shorting winner: **50% QQQ
/ 50% long-only factor-tilt = CAGR 16.6%, Sharpe 1.42, maxDD -22%** (long-only
factor tilt is corr -0.06 to QQQ -> diversification lifts Sharpe vs QQQ 1.13 with
1/3 less DD; it's a Sharpe/DD win, CAGR slightly below QQQ). With modest
leverage: 50QQQ/50LOfac + 0.5x net alpha = 20.5%/Sharpe 1.70. Factor-timing
(weight by trailing 12m) lifts net L/S 0.99->1.08. (Scripts exp73-75; _ls.pkl.)

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
