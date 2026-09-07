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

**Results:** (to be filled after the run; this section was committed first.)
