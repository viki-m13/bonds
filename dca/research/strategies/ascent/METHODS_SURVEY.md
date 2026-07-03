# Stock-Picking Methods — external research survey, graded against our bar

*Deep-research sweep (web search → source fetch → 3-vote adversarial verification),
synthesized against ASCENT's validation bar: **era-stable 2000–2026 incl. 2003–2014,
top-decile precision above random base rate, factor-orthogonal errors, realistic
costs, retail-accessible point-in-time data.** 13 claims verified 3-0; a further
12 well-sourced but unverified (the verify pass was cut short) are flagged
`[unverified]`. Sources listed at the bottom by [S#].*

---

## 0. The meta-finding that reframes everything (verified)

Before grading any single method, three large meta-studies set the prior — and it
is brutal for *published* signals:

* **McLean & Pontiff (2016)** [S1]: across 97 peer-reviewed predictors, long-short
  returns are **26% lower out-of-sample and 58% lower post-publication**. Average
  in-sample edge 58 bps/month → ~24 bps/month surviving after publication,
  **before costs**. Sample ends 2013.
* **Chen & Velikov, "Zeroing In" (2022)** [S2]: across 204 anomalies, after
  effective spreads + post-publication decay + restricting to the post-2005
  electronic era, the average anomaly nets **~4 bps/month** (vs 68 bps gross
  in-sample). Decay is 50% post-publication → **72% post-2005 → 93% after costs**;
  the *90th-percentile* anomaly nets ~6–10 bps/month. Only **7% of post-2005 net
  t-stats exceed 2.0** — what pure luck would produce.
* **Multi-signal combination is the worst offender** [S2]: Fama-MacBeth / rank-avg
  / IPCA / LASSO ensembles gross ~380 bps/month in 1985–2005 but net **~20
  bps/month** post-2005. *This is exactly the ML-ensemble trap our own exp24/76
  and WAVE rebuild hit — an independent, quantified confirmation.*
* **Jacobs & Müller (2020)** [S4], 241 anomalies × 39 countries: **the US is the
  only country with reliable post-publication decay.** For a QQQ/US strategy,
  assume published edges decay; the mechanism is arbitrage/crowding, not that the
  originals were fake.

**Implication for our bar:** any method whose evidence is a *published US
cross-sectional anomaly* should be assumed near-dead net-of-cost today. This is
not pessimism — it is the measured base rate, and it explains why 46+ of our own
experiments died. The survivors must be either (a) un-arbitrageable for a
structural reason, or (b) not yet crowded.

---

## 1. Method-by-method grade

### A. SEC filing-text change ("Lazy Prices", Cohen-Malloy-Nguyen) — **most promising untested**
* **Claim (verified [S3]):** shorting firms that materially change their 10-K/10-Q
  text vs buying "non-changers" earned **up to 188 bps/month alpha (>22%/yr)**,
  1995–2014, all US filers. Returns accrue **slowly** as the changed information is
  later revealed — an inattention-driven multi-month drift, exploitable *after* the
  public filing [S3, verified].
* **Grade vs our bar:**
  - Retail data: **YES** — EDGAR full-text 10-K/10-Q is free and point-in-time
    (filing date is the timestamp; no look-ahead).
  - Era stability: published 2018; **2019–2026 is genuine untested OOS.** Given the
    US-decay finding [S4] expect haircut, but the driver (investor inattention to
    boilerplate changes) is a *behavioral* edge less prone to fast arbitrage than a
    price factor.
  - Precision/orthogonality: the signal is textual, **structurally orthogonal to
    price/size/momentum factors** — the one property our tail-precision diagnosis
    (§6c) says is required. Unknown whether it clears the 15–20% top-decile bar;
    must be measured.
  - **This is the single best candidate we have not built.** It directly matches
    the §6c spec (idiosyncratic, non-price, free, PIT). Caveat: heavy compute
    (parse every 10-K/Q, compute cosine/Jaccard similarity vs prior filing).

### B. Insider trading, opportunistic vs routine (Cohen-Malloy-Pomorski) — **real but small, already in our program**
* **Claims:** "opportunistic" insiders (no calendar routine) earn **82 bps/month**;
  routine insiders ~zero [S9, unverified]. BUT **70–80% of the alpha dissipates
  between the insider's trade and the next day — before the Form 4 is public**;
  filing-date-entry Sharpe collapses from >9 (unimplementable transaction-date) to
  **~1.5** [S10, unverified]. Practitioner backtests overstate this precisely by
  entering at transaction date (look-ahead) [S10, unverified].
* **Grade:** retail data YES (Form 345 bulk TSVs, which we already built). This is
  **the one signal our own program independently validated** (exp34–51: clean
  Tiingo test, +0.5–0.6%/3m, survivorship-robust) — external literature agrees it's
  real, agrees it's *small at the filing date we can actually trade*, and adds the
  refinement we should apply: **filter to opportunistic (non-routine) insiders**
  and expect ~QQQ-level risk-adjusted, not a QQQ-beater. Consistent with our §8
  accuracy result. Low incremental value but a legitimate orthogonal tilt.

### C. Post-Earnings-Announcement Drift (PEAD) — **decayed, cost-gated**
* **Claims (verified [S5-class, S12/S13]):** classic SUE decile L/S ~2%/60 days
  (~18%/yr) in the Bernard-Thomas era; **measurably decayed** post-publication
  (Chordia 2014, Martineau 2019) though still nonzero and persistent
  internationally. Net profitability **disputed and concentrated in small,
  illiquid, high-cost stocks — near zero after frictions** [unverified S5].
* **Grade:** needs earnings dates + estimates (SUE). Standardized surprise needs a
  consensus-estimate feed (not free PIT). Our repo already found PEAD sign-unstable
  / ~0 at 1y (exp-notes). **Below the bar** for a QQQ-beater; a possible weak gate.

### D. Congressional copy-trading — **dead, textbook decay**
* **Claims [unverified but multi-source S7/S8/govgreed]:** the cited Ziobrowski
  edge (~12%/yr Senate, ~6%/yr House pre-2012) **reinterpreted as no informed
  trading**; 2004–08 the average member *underperformed* by ~2.8%/yr [S7]; post-
  STOCK-Act 2012–2020 bought stocks **underperformed** by 26 bps/6mo, and even the
  99th-percentile member's picks are statistically **indistinguishable from random**
  [S8].
* **Grade:** **fail.** Popular (Quiver/Unusual Whales marketing) but the peer-
  reviewed record says no durable edge, and it fails the random-base-rate test by
  construction. Do not build.

### E. 13F institutional cloning ("alpha cloning") — **weak, stale, capacity-limited**
* Quantpedia/academic [S11, S14]: super-investor 13F clones show some historical
  alpha but 45-day disclosure lag + quarterly staleness + our own exp81-82 result
  (breadth-change IC ~0.037, Sharpe 0.65 — modest, only 33 months dense) put it
  **below the bar** as a standalone. Orthogonality is real (it's holdings, not
  price) but the lag guts the timing.

### F. ML / LLM stock selection — **the ensemble-overfit trap, quantified externally**
* Gu-Kelly-Xiu replications [S17] and "expected returns of ML strategies" [S24]
  confirm what our exp24/69/76 found: impressive in-sample IC, heavy decay net of
  cost, and the gains concentrate in **micro/small caps and short-side** that a
  long-only QQQ-benchmarked retail book can't harvest. LLM-based selection [S19] is
  early, prone to look-ahead (training-data leakage of the future) and unproven OOS.
  **Below the bar** as return generators; useful only as nonlinear *combiners* of
  already-orthogonal signals.

### G. Event-driven (spinoffs, index adds/deletes, buybacks, lockups, merger-arb)
* Not returned with verified quantified OOS claims before the sweep was cut. Prior
  knowledge + the decay meta-finding: **index-effect and merger-arb have visibly
  decayed** (index-add pop arbitraged away post-2010); **buyback-announcement drift
  and spinoffs** retain modest documented edges but are **event-sparse** (too few
  QQQ-universe events for a concentrated monthly book) and hard to PIT-source free.
  Candidate for a *satellite event gate*, not a core picker. Flagged for a future
  targeted sweep.

---

## 2. The 3 methods worth building (in priority order)

1. **Lazy-Prices filing-text change (A).** The only surveyed method that is
   simultaneously: quantified-large in-sample, retail/free/PIT (EDGAR), genuinely
   OOS-untested since publication, and **structurally factor-orthogonal** — the
   exact §6c property every price/fundamental transform lacked. Highest expected
   information-per-unit-effort. Build: parse consecutive 10-K/10-Q pairs, compute
   text-similarity (cosine on TF-IDF + Jaccard), rank cross-sectionally, test on our
   survivorship-clean Tiingo panel with the §6c precision + factor-orthogonality
   gates *before* any portfolio backtest.
2. **Opportunistic-insider tilt (B).** Already 80% built in our repo; the external
   literature says the one refinement that matters is **routine-vs-opportunistic
   classification** and **honest filing-date (not transaction-date) entry**. Low
   effort, real-but-small; deploy as a diversifying tilt, not a core.
3. **8-K item + text content (extends our loop7).** We have the item-level event
   panel (1.02M filings). The Lazy-Prices logic applied to 8-K *content* (not item
   counts, which we showed are null) is the natural merge of A and our existing
   infrastructure.

**Explicitly not worth building:** congressional copy-trading (dead), pure
published-anomaly factors (93% cost-decayed), ML return-generators (ensemble
overfit trap), standalone 13F cloning (lag-gutted).

---

## 3. The honest caveat this survey reinforces

Every verified number says the same thing our 119 experiments said: **published,
price-derived, US-large-cap edges are arbitraged to ~4–10 bps/month net.** The only
places edge survives are (a) **non-price, inattention-driven textual/behavioral
signals not yet crowded** (Lazy Prices) and (b) **structurally un-arbitrageable
corners** (opportunistic insider, small-cap frictions) that are small and capacity-
limited. There is no verified method in the external literature that clears
"consistently, with high accuracy, beat QQQ" — consistent with our §8 finding that
the base rate itself (only 32–46% of stocks beat QQQ at 12m) makes *accuracy* the
wrong target. The realistic prize is a **small, orthogonal, expectancy-positive
tilt**, and Lazy-Prices text-change is the best unexplored shot at one.

---

## Sources
[S1] McLean & Pontiff (2016), *Does Academic Research Destroy Stock Return Predictability?* — researchgate 315421495
[S2] Chen & Velikov (2022), *Zeroing In on the Expected Returns of Anomalies*, JFQA — cambridge.org S0022109022000874a
[S3] Cohen, Malloy & Nguyen (2020), *Lazy Prices*, J. Finance — ssrn 1658471
[S4] Jacobs & Müller (2020), *Anomalies across the globe: Once public, no longer existent?*, JFE — sciencedirect S0304405X19301618
[S5] *PEAD survey* — sciencedirect S2214635020303750
[S7] Eggers & Hainmueller, congressional portfolios 2004-08 — ssrn 1762019
[S8] *Congressional trading post-STOCK-Act* — sciencedirect S0047272722000044
[S9] Cohen, Malloy & Pomorski (2012), *Decoding Inside Information*, J. Finance — wiley 10.1111/j.1540-6261.2012.01740.x
[S10] *Insider alpha timing / filing-date decay* — ssrn 5966834
[S11] Quantpedia, *Alpha cloning / 13F* ; [S14] Berkin/hedge-fund cloning — wiley jofi.12365
[S17] Tidy-Finance, Gu-Kelly-Xiu replication ; [S19] arXiv 2304.07619 LLM selection ; [S24] Quantpedia, expected returns of ML strategies
*(Full URL list in the workflow output; [S6],[S13],[S15-16],[S18],[S20-23] are supporting/duplicate.)*
