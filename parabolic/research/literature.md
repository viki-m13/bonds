# Literature & practitioner review: catching stocks before they go parabolic

Synthesis of an extensive web sweep (academic / SSRN, FinTwit & trading
educators, Reddit / retail communities) commissioned for the IGNITION strategy.
Goal: enumerate every OHLCV-computable signal claimed to precede large upside
("parabolic") single-stock moves, with each claim's *direction*, whether it
predicts the **mean** or the **tail**, and its robustness — so the backtest tests
real hypotheses, not folklore. Sources are cited inline.

The single most important organizing fact, on which the academic and retail
evidence agree:

> **Volatility is predictable; direction is not.** Realized volatility is
> strongly autocorrelated (volatility clustering), so contraction/"squeeze"
> signals genuinely predict that *a big move is coming* — but they carry **no
> directional information**. And the signals that best *select the fat right
> tail* (MAX, idiosyncratic vol, skewness) are precisely the ones with the
> **worst average returns**. A pre-parabolic strategy is therefore only viable
> if it is explicitly **convex** (a few big winners dominate a basket) and/or it
> **conditions** the tail-selectors with mean-positive, breakout-aligned signals.

That sentence is the design brief for IGNITION.

---

## 1. Academic evidence (peer-reviewed / SSRN)

### Mean-positive, breakout-aligned (Tier 1)
- **52-week-high nearness** — `Close / max(High, 252d)`. George & Hwang (2004,
  *J. Finance* 59:2145) show nearness to the 52-week high **subsumes** raw
  Jegadeesh–Titman momentum as a forecaster and, unlike momentum, **does not
  reverse** long-term. Positive, predicts the **mean**. Mechanism = anchoring /
  under-reaction at the reference high.
- **High-volume return premium** — abnormally high daily/weekly volume vs a
  trailing window predicts ~1 month of out-performance via the
  visibility/recognition channel. Gervais, Kaniel & Mingelgrin (2001, *J.
  Finance* 56:877); replicated across developed + emerging markets. The cleanest
  academic licence for "volume confirms the move."
- **Momentum & residual momentum** — Jegadeesh–Titman 12–1 (skip the last
  month); Blitz, Huij & Martens (2011) show residual (factor-neutralized)
  momentum ~doubles risk-adjusted profits and halves crash risk. Positive, mean.
- **Frog-in-the-pan / information discreteness** — `ID = sign(PRET)·(%neg −
  %pos days)` over the formation window. Da, Gurun & Warachka (2014, *RFS*
  27:2171): **continuous** (smooth, low-ID) winners keep running (+5.9%) while
  **discrete** (jumpy) ones don't (−2.1%). A momentum-quality conditioner. Note
  the tension with "buy the explosive gap" — smooth advances sustain best.

### Tail selectors — NEGATIVE mean, fat right tail (use only conditioned/convex)
- **MAX / lottery effect** — `max(daily return, 21d)`. Bali, Cakici & Whitelaw
  (2011, *JFE* 99:427): high-MAX stocks have **low** subsequent mean returns
  (long-low/short-high > 1%/mo) **but** higher subsequent volatility and
  skewness — i.e. they *are* the jackpot pool. The negative premium weakens for
  stocks **near** their 52-week high → MAX × 52WH-nearness is worth testing.
- **Idiosyncratic-volatility puzzle** — Ang, Hodrick, Xing & Zhang (2006): high
  IVOL → abysmal next-month returns. Tightly linked to MAX and skewness (one
  "lottery" cluster). Parabolas come from here, but the *average* member loses.
- **Expected idiosyncratic skewness** — Boyer, Mitton & Vorkink (2010, *RFS*
  23:169): high expected skew → ~1%/mo lower alpha. Right-tail richness is priced.
- **Betting-against-beta** — Frazzini & Pedersen (2014): low beta wins
  risk-adjusted on average. For parabolic *upside* you want the **reverse**
  (high beta in risk-on regimes), accepting unconditional under-performance;
  Novy-Marx & Velikov (2022) show BAB itself is fragile to implementation.

### Explosivity / bubble detection (regime, tail)
- **PSY SADF / GSADF explosivity tests** — Phillips, Shi & Yu (2015, *Int. Econ.
  Rev.*): recursive right-tailed unit-root tests date-stamp explosive
  (super-martingale) regimes in real time. Directly operationalizes "parabolic,"
  fully OHLCV-computable — but flags a move *already underway* (late entry).
- **Log-Periodic Power Law (Sornette)** — **rejected as a tradable signal**:
  unstable 7-parameter fits, poor out-of-sample crash prediction, incoherent
  theory (Brée & Joof 2013). A research curiosity, not an edge.

### Weak / myth from the academic side
- **Coiled-spring / Bollinger squeeze as standalone alpha** — the only
  refereed result is that the probability of a large move scales inversely with
  the length of the preceding quiet period — **direction-agnostic**. VCP has no
  peer-reviewed validation. Use compression only as a *trigger*, never a signal.
- **Amihud illiquidity as a timing signal** — it's a cross-sectional risk
  premium living in microcaps, not a parabolic trigger.

---

## 2. FinTwit / practitioner methodologies (codeable specs)

Every elite price-action trader buys the **same archetype** — a liquid,
high-relative-strength name in a multi-MA uptrend near its 52-week high, that
made a large prior thrust and is now coiling in a tight, low-volume,
shrinking-range consolidation — and fires on a **volume-expansion breakout**
through a pivot, with a stop one contraction-low below.

- **Mark Minervini — Trend Template + VCP.** 8-point Stage-2 gate: `C>SMA150`,
  `C>SMA200`, `SMA150>SMA200`, `SMA200` rising ≥1mo, `SMA50>SMA150>SMA200`,
  `C>SMA50`, `C ≥ 1.25·min(low,252)`, `C ≥ 0.75·max(high,252)`, RS≥70. VCP = 2–4
  contractions each ~half the prior depth (e.g. 25→12→6%), volume drying into the
  pivot, breakout on `V ≥ 1.4·SMA50(V)`.
- **Kristjan Kullamägi (Qullamaggie).** Prior 1–3mo thrust of +30–100%+; ADR% =
  `100·(mean_20(High/Low) − 1)` > ~5%; tight flag riding the rising 10/20-EMA;
  buy the breakout, stop ≤ 1×ADR, trail the 10/20-EMA, sell into strength.
  Episodic Pivot (EP) = `Open/Close[-1] ≥ 1.10` on huge volume out of a flat base.
- **Pradeep Bonde (Stockbee).** 4% breakout scan `c/c1≥1.04 & v>v1`; 3–5 day
  momentum bursts; "anticipation" = NR7/squeeze near the 10/20-MA after a prior
  move; EP via a ≥9M-share volume footprint.
- **William O'Neil / IBD CANSLIM.** Cup-with-handle / flat base breakouts; **RS
  line at a new high before price** (`Close/SPY` ratio at a 252d max while price
  is not); breakout volume ≥ 1.4×SMA50(V); RS Rating = percentile of
  `0.4·r63+0.2·r126+0.2·r189+0.2·r252`.
- **Stan Weinstein — Stage Analysis.** Stage-2 breakout above a rising 30-week
  (150d) MA on ≥2× volume; **Mansfield RS** `(C/SPY)/SMA200(C/SPY) − 1` rising.
- **Darvas box / High Tight Flag.** New-high box breakouts on rising volume;
  HTF = ~+100% pole in 4–8 weeks then a ≤25% flag, breakout on volume.

**Common denominators (encoded as features):** MA stacking & rising long MAs;
nearness to 52w high + ≥25% above 52w low; prior multi-week thrust; relative
strength / RS-line new high; realized-vol & range contraction; volume dry-up
into the pivot then expansion; ADR/ATR liveliness; gap-up catalyst; breakout
trigger with volume confirmation.

---

## 3. Retail / community evidence and debunkings

- **TTM Squeeze (BB inside Keltner), BandWidth-percentile squeeze, NR7,
  ribbon compression** — all confirmed *volatility-expansion* predictors and all
  **explicitly direction-agnostic**. Bulkowski's 29,021-sample NR7 study: up- and
  down-break failure ~46/47% (symmetric), pattern rank 11/23. StockCharts on the
  BB squeeze: "narrowing bands do not provide any directional clues."
- **RVOL spikes** — the defensible version is the GKM monthly premium, not the
  uncited "RVOL>2 → 40% better follow-through" vendor stat.
- **OBV / accumulation** — confirmation-only; not reliably predictive standalone.
- **Signals requiring data we do NOT have** (excluded): options flow / gamma,
  dark-pool prints, social sentiment, fundamentals / earnings dates, float /
  short interest, intraday bars. The *price/volume footprint* of an earnings
  catalyst (gap + volume) is computable; the *cause* is not.

---

## 4. What this implied for IGNITION — and what the data then said

The review prescribed: **Direction layer (52WH nearness, MA-stacking, RS) ×
Timing/coil layer (squeeze, contraction) × Confirmation (volume shock, gap),
with the lottery/energy axis used only as a conditioned tail-sampler inside a
convex basket.**

We encoded all of it (`features.py`) and tested it honestly with an IS/OOS
event study (`eventstudy.py`) and a survivorship-controlled basket backtest
(`backtest.py`). The result **partly contradicts the practitioner folklore on
this universe (S&P 500 large caps, 6-month +50% objective)**:

1. The **"buy the breakout near 52-week highs"** archetype (Trend Template,
   nearness, RS-line-new-high) **does not precede parabolic moves** here — those
   names are already extended and sit *below* the parabolic base rate. The
   literal practitioner composite is *falsified* for this objective (it even
   loses to a random pick in-sample). It selects steady compounders (what SUMMIT
   already harvests), not rockets.
2. The strongest precursors are the **energy / lottery axis** (beta, realized
   vol, ADR, MAX) — exactly the negative-mean, fat-tail selectors the academics
   warned about: ~3–5× the parabolic base rate, but **no in-sample edge over a
   random pick from the same high-energy pool** (a regime bet that paid in
   2016–26, not robust alpha).
3. The one signal with a **positive, sign-stable IS→OOS rank-IC** and positive
   excess is **"already turned off the 52-week low"** (`dist_52w_low`), joined by
   the **episodic-pivot gap** (`ep_gap`, positive excess both splits) and
   **low correlation** to the market.
4. **Frog-in-the-pan smoothness** adds mean quality (positive IC both splits)
   without selecting the tail — a stabilizer, exactly as Da et al. predict.

IGNITION therefore blends the **mean-positive conditioners** (`dist_52w_low`,
`ep_gap`, low-`corr`, `fip`) **with** a controlled dose of the **energy tail**,
restricted to the high-ADR pond — the convex, conditioned construction the
academic review prescribed. That conditioning is what flips the in-sample
edge-over-random from negative (pure energy, 0.47) to positive (IGNITION, 0.52),
and holds it positive out of sample (0.57). See `../README.md` for results.

---

### Primary sources
- George & Hwang (2004), *The 52-Week High and Momentum Investing*, J. Finance 59:2145.
- Gervais, Kaniel & Mingelgrin (2001), *The High-Volume Return Premium*, J. Finance 56:877.
- Jegadeesh & Titman (1993); Blitz, Huij & Martens (2011), *Residual Momentum*, JEF 18:506.
- Da, Gurun & Warachka (2014), *Frog in the Pan*, RFS 27:2171.
- Bali, Cakici & Whitelaw (2011), *Maxing Out*, JFE 99:427.
- Ang, Hodrick, Xing & Zhang (2006), *The Cross-Section of Volatility and Expected Returns*, J. Finance.
- Boyer, Mitton & Vorkink (2010), *Expected Idiosyncratic Skewness*, RFS 23:169.
- Frazzini & Pedersen (2014), *Betting Against Beta*, JFE; Novy-Marx & Velikov (2022).
- Phillips, Shi & Yu (2015), *Testing for Multiple Bubbles* (SADF/GSADF), IER.
- Brée & Joof (2013), critique of the LPPL crash model, IRFA.
- Practitioner: Minervini *Trade Like a Stock Market Wizard*; Qullamaggie (qullamaggie.com);
  Bonde (stockbee.blogspot.com); O'Neil / IBD CANSLIM; Weinstein *Secrets for Profiting*;
  Darvas *How I Made $2,000,000*. Bulkowski (thepatternsite.com) for pattern failure rates.
