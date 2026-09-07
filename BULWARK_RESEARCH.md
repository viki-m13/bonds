# BULWARK — cheap junk that does not go bust (corporates)

**Question (owner, 2026-09-06):** can we build a corporate strategy that buys
*very cheap* (junk-spread) bonds while avoiding the ones that default — and
validate that honestly, including the realised default experience of the
book?

This document is written in the repo's standing discipline: design and kill
gates are committed BEFORE results, in-sample screening is separated from a
single out-of-sample look, and every failure is reported.

## 1. Can we even see a default? (step 0, DONE before designing)

The tape carries no default flag, so a label had to be built and validated.
`corps/research/bulwark_label.py` labels each bond from the audited engine2
cache:

- **DISTRESS DAY** — `mid < 70` AND `cs > 20%` AND `mat >= 1`. Both legs are
  required: price alone misclassifies deep-discount/zero-coupon bonds, spread
  alone misclassifies data glitches.
- **BUST** — the bond's last 10 prints have median `mid < 55` while more than
  1.25y of maturity remains: it left the tape in distress rather than
  maturing, being called, or going quiet near par.
- **GOOD EXIT** — terminal prints >= 85, or terminal maturity <= 1.25y
  (OSBAP truncates coverage at ~1.0y, so that is *coverage*, not credit).

**Validation.** 55,542 bonds labelled: 7.23% ever distressed, 1.58% bust
(434 of 7,906 CUSIP6 issuers). New-distress counts by year reproduce the
actual credit cycle with no tuning — 2008: 1,861; 2009: 345; 2015: 230;
2016: 120; 2019: 96; 2020: 345; 2023: 67 — and near-zero in 2010, 2013,
2017, 2021. The largest labelled busts were then resolved *independently*
through EDGAR full-text search on the CUSIP: Lehman Brothers Holdings, General
Motors, Enron, Chesapeake Energy, Frontier Communications, AMR Corp and Arch
Coal all resolve to the correct issuer (7 of 10 sampled resolve to the issuer;
the other 3 return fund holders that report the CUSIP, which confirms the
identifier but not the name). The labeller measures credit, not noise.

**Disclosed limits.** It is a tape-inferred proxy, not a rating-agency default
file: it cannot separate Chapter 11 from a distressed exchange or a covenant
default, and a bond that defaults but keeps printing above 55 is missed.
Right-censoring at the data end (2025-03-31) makes late "bust" labels
provisional.

## 2. Design (PRE-REGISTERED — committed before any strategy result)

**Universe U:** engine-eligible bond-days with `mat <= 5`, `cs >= 4%`
(400bp — crossover/junk), `mid >= 40`. 2.27M eligible bond-days, present in
every year (41k–210k/yr), so the strategy is not a crisis-only artefact.

**Mechanics (frozen, identical to the audited GRANITE/BEDROCK engine):**
signal on day *t* uses data through *t* only; entry at the first customer-ask
print strictly after *t* (<=7d) subject to the limit rule (ask <= prior mid +
0.25); exit at the first customer-bid print after 365d, hard stop 455d; REAL
recovered coupons (`coupon_inv`); depth weights are NOT used — BULWARK is
equal-weighted (there is no dislocation depth to weight by); 30d per-bond
cooldown; one open position per CUSIP6 issuer.

**Survival screens** (each tested alone on IS, then in combination):
- **S1 not-the-cheapest** — exclude `cs` above the same-day cross-sectional
  97th percentile of U, and exclude `mid < 70`. *(Campbell-Hilscher-Szilagyi
  distress anomaly: the deepest-distress tail carries the defaults.)*
- **S2 short maturity** — `mat <= 3`. *(Israel-Palhares-Richardson defensive
  style; less time to default; the credit-spread puzzle is largest short.)*
- **S3 no deterioration** — `cs_t <= 1.25 x cs_{t-60 rows}` AND `mid_t >=
  mid_{t-60 rows} - 5`. *(Jostova et al. momentum in HY: losers keep losing.)*
- **S4 issuer-curve coherence** — the bond's `cs` exceeds the median `cs` of
  >=2 sibling CUSIP6 bonds by >= 0.5pp: cheap versus *its own issuer curve*,
  i.e. idiosyncratic/liquidity cheapness rather than a whole-issuer credit
  story. Requires >=2 siblings (fails when absent — evidence is required here,
  unlike BEDROCK's G4 which passes on no information).
- **S5 activity** — trailing-90d active-day count above the same-day median of
  U (a name the market is actually watching).
- **S6 issuer not in distress** — no sibling bond of the same CUSIP6 is in the
  DISTRESS state on the signal day (point-in-time cross-section, no lookahead).

**Kill gates (frozen).** A screened book is admitted to the single OOS look
only if, in-sample (2003-2015), ALL of:
- **G-A** (the headline claim) — bust-linked trade rate <= 25% of the
  unscreened U cohort's rate, where a trade is bust-linked if the bond is
  labelled bust and its bust exit falls inside the holding window;
- **G-B** — mean per-trade return and Sharpe(m) both >= the unscreened cohort;
- **G-C** — per-trade excess versus the matched random-entry control (same
  universe, same conventions, issuer-clustered bootstrap) > 0 at p < 0.01.

**OOS:** one look, 2016-01-01..2024-01-01 (censor-safe: data end − 455d).
Adopt iff the bust-rate reduction retains >=60% of the IS reduction AND excess
stays positive at p<0.05. Otherwise report the kill.

**Reported regardless:** realised bust-linked count, share of trades with
return < −20%, worst-decile mean, and full return/Sharpe/drawdown stats.

**Phase 2 (only if Phase 1 survives):** SEC/XBRL fundamentals (leverage,
interest coverage, cash) as an additional quality screen on the 2010+
subsample, where XBRL coverage exists — noting it halves the sample and only
covers SEC registrants.

## 3. IS RESULTS (2003-2015)

Two corrections were made before reading any of this, both disclosed:

1. **Exit censoring (`bulwark_censor_audit.py`).** `e2.run_events`' stale-exit
   fallback dates an exit at entry+455 using the last bid at or before that
   date *without requiring it to post-date the entry*, and accrues coupon
   across the gap. On the BULWARK book 31.3% of fills exit stale with a
   median stale gap of **249 days**, and 22 fills had no bid print after
   entry at all (exit price predating entry, booked at +12.05%). BULWARK
   therefore uses an **honest exit**: first bid >=365d; else the last real
   bid strictly after entry, at ITS OWN date with coupon only to that date;
   else the position is booked as a credit loss at min(last mid, 40). Cost
   to the baseline: **+10.41% -> +8.05% per trade, CAGR +9.09% -> +7.42%,
   Sharpe 1.11 -> 0.77**.
2. **Control convention mismatch.** The first v2 run scored honest-exit
   strategies against a control still using the inflated convention, which
   made every excess column negative. The control was rebuilt on the same
   honest exit (`matched_control_honest`). All numbers below use it.

| book | n | mean/trade | win | bust-linked | ret<-20% | CAGR | Sharpe(m) | maxDD | excess (p) |
|---|---|---|---|---|---|---|---|---|---|
| **U unscreened** | 3448 | +8.05% | 83% | 38 (1.1%) | 6.4% | +7.42% | 0.77 | −28.3% | **+2.33% (0.000)** |
| +S1 not-cheapest | 3350 | +6.59% | 84% | 21 (0.6%) | 6.1% | +6.21% | 0.68 | −27.2% | +0.30 (0.209) |
| +S2 mat<=3 | 1807 | +7.30% | 83% | 19 (1.1%) | 6.9% | +6.05% | 0.64 | −29.8% | +0.73 (0.127) |
| +S3 no-deterioration | 2780 | +7.02% | 84% | 24 (0.9%) | 5.8% | +6.57% | 0.64 | −33.0% | +0.82 (0.032) |
| **+S4 issuer-curve** | 616 | **+11.36%** | 86% | 7 (1.1%) | 5.7% | +9.08% | 0.66 | −40.5% | **+4.51% (0.000)** |
| +S5 activity | 1481 | +7.34% | 82% | 24 (1.6%) | 8.4% | +7.67% | 0.63 | −38.0% | +2.19 (0.001) |
| +S6 issuer-not-distressed | 3379 | +7.11% | 84% | 29 (0.9%) | 6.3% | +6.76% | 0.73 | −27.4% | +1.28 (0.003) |
| S1+S4 | 576 | +9.47% | 87% | 3 (0.5%) | 4.5% | +7.01% | 0.57 | −35.6% | +1.20 (0.126) |
| S2+S4+S6 | 326 | +7.40% | 87% | 1 (0.3%) | 4.9% | +4.86% | 0.38 | −45.7% | −1.44 (0.886) |
| **S1+S2+S4+S6** | 325 | +8.27% | **88%** | **0 (0.0%)** | 4.3% | +5.65% | 0.48 | −37.8% | −0.88 (0.755) |

**No specification passes all three frozen gates.** Under the pre-registered
protocol that is a KILL for the screen program, and it is reported as one.

**What the evidence actually says — three findings, all against the premise
of the question:**

1. **The defaults were never the problem.** The unscreened cheap-junk
   universe already loses to credit on only **1.1% of trades** and wins 83%
   of the time. "Super cheap junk" that is also short-dated (<=5y), liquid
   and priced >=40 is *structurally* not a default-heavy population — the
   universe definition is doing the survival work before any screen runs.
2. **Zero defaults is achievable, and it is not worth it.** S1+S2+S4+S6
   books **0 bust-linked trades in 325** (expected 3.6 at the baseline rate;
   p~2.7% under independence — suggestive, in-sample, and clustered across
   251 issuers). But eliminating a 1.1% loss rate costs 90% of the book:
   3448 trades -> 325, and the portfolio gets *riskier*, not safer —
   **maxDD −28.3% -> −37.8%, Sharpe 0.77 -> 0.48**. Concentration risk
   dominates the credit risk it removes. Screening for zero defaults is a
   bad trade.
3. **The one screen with real edge is a value signal, not a safety signal.**
   S4 (cheap versus the bond's own issuer curve) earns **+11.36%/trade,
   excess +4.51% over matched random entries at p<0.001** — the strongest
   selection result in the program — while leaving the bust rate exactly
   where it was (1.1%). It finds mispricing, not solvency.

## 4. DISCLOSED EXTENSION — one OOS look (committed BEFORE the OOS run)

The frozen G-A gate ("bust rate <=25% of baseline") was calibrated before the
baseline bust rate was known to be 1.1%; it demands <=0.28% and is close to
unmeasurable at these sample sizes. The screen program is killed as
pre-registered. But two specifications are worth a single OOS look, and the
rule is fixed here before running:

- **BULWARK-U** — the unscreened cheap-junk carry book (IS excess +2.33%,
  p<0.001, bust 1.1%).
- **BULWARK-S4** — U + issuer-curve cheapness (IS excess +4.51%, p<0.001).
- **BULWARK-Z** — U + S1+S2+S4+S6, the zero-default spec, carried purely to
  test whether "0 busts" survives OOS.

**Adoption rule (frozen):** OOS excess > 0 at p<0.05 AND OOS mean/trade > 0
AND OOS bust-linked rate <= 2x its IS rate. Reported exactly as printed,
adopted or killed.

**Results:** (to be filled after the single OOS run.)
