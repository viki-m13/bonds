# External strategy notes — for learning, not endorsement

*Records how notable outside strategies work and what's worth borrowing or
avoiding. Validate any of these with `VALIDATION_METHODOLOGY.md` before
trusting headline numbers.*

---

## CRT / "Daily Stock Guide"  (repo: `viki-m13/crt`, site: dailystockguide.com)

**The best-validated DCA stock-picker reviewed so far** (genuinely good
methodology hygiene), audited 2026-06-15. Live = "v5 / E2".

### How it works
- **Universe:** point-in-time S&P 500 (985 unique tickers across 2003–2026),
  membership-masked. Broader 1,833-ticker panel for pre-2003 / generalization.
- **Features:** 79 **price-only** signals per stock (momentum, trend, recovery,
  RSI, vol, drawdown, + 12 custom "novel" factors), cross-sectionally ranked.
- **Primary model:** **gradient-boosted trees (GBM)**, walk-forward, retrained
  annually with a **7-month embargo**, predicting forward-return ranks at
  1/3/6-month horizons.
- **Confirmation filter:** **Chronos-bolt-tiny** (Amazon, 9M-param zero-shot
  time-series foundation model) — keep a pick only if its Chronos p70
  cross-sectional rank ≥ ~0.45. Adds a price-shape prior independent of the GBM.
- **Crash gate:** SPY −8% in 21d or −5% over 6m → cash.
- **Construction:** two regime-conditional "sleeves", each top-2–3 by score,
  inverse-vol weighted, 40%/name cap → ~6 names max. 6-month min hold, monthly
  rebalance only when a sleeve's picks fall out of the top-K, forced 24-month
  max hold. ~1.0–1.5× annual turnover.

### Claimed performance
~56.6% CAGR (2003–2025), Sharpe ~1.10, **beats S&P-500-DCA in 100% of rolling
10-year windows (159/159)**, ~55%/yr median 10y vs ~13% for S&P-DCA. Honestly
discloses: monthly win ~58% (coin-flip), worst 1y DCA 0.51×, drawdowns −56%
(DCA) / −77% (lump), and that the edge is "front-loaded in 2003–2009… will not
repeat at that scale."

### What the validation playbook found
- **Edge is concentrated in crash-recovery eras** (their own per-split data, vs
  SPY): **GFC 2008–10 +108.8pp/yr**, **COVID 2020–22 +57.8pp**, post-GFC
  +32.9pp. Normal eras are modest: 2014–16 the GBM *lost* (−1.5pp, "rescued"
  to +1.0pp only by the Chronos filter), 2017–19 +6.9pp. → It is mechanically a
  **crash-recovery / high-beta amplifier**, which is why drawdown is −77% and
  why the headline CAGR is non-repeatable.
- **"100% beats S&P" ≈ growth/tech beta, not skill** (playbook test #3). Their
  own **QQQ test** (the right benchmark, 2015+) is the honest one: +7.15pp
  full-window but dominated by a tiny 2015–16 window (+52pp) and the AI rally
  (+22.5pp), and it **LOSES to QQQ in the recent mega-cap windows** (STRICT
  2022–25 **−9.25pp**, Post-COVID 2020–26 **−9.73pp**). Their report candidly
  says NDX deployment is "a curiosity… NOT recommended as a production switch."
- **Overfitting flags:** the Chronos filter was selected *on* the walk-forward
  splits (it's precisely the variant that flips the one losing split to a win)
  → the "10/10 beats SPY" is partly selection-contaminated (playbook #10). The
  core GBM is price-only ML, which has ~0 durable cross-sectional IC in our own
  tests (playbook #6) — so its edge is most plausibly crash-recovery beta, not
  selection alpha.
- **Survivorship:** best handled of any strategy reviewed — they *measured* the
  gap (51%→96% coverage), backfilled 161 acquired/renamed names (→99.7% by
  2025), Monte-Carlo delisting stress. Residual: 213 OTC bankruptcy tickers
  still missing and acquisitions booked at 0% (not −100%) — both **flatter the
  GFC window**, which is the one carrying the headline.

### Verdict
Validation rigor **A−** (real walk-forward + embargo + PIT + survivorship
correction + QQQ test + honest disclosures). Deployable edge **far below the
56% headline**: strip the non-repeatable crash-recovery eras and benchmark vs
QQQ and it's a *modest, regime-dependent* edge — beats QQQ in dispersion
regimes, loses in mega-cap regimes, with −56% drawdowns. Consistent with the
standing conclusion in `VALIDATION_METHODOLOGY.md`: the wins are regime/beta +
recent era + a soft benchmark, not durable selection skill.

### What's worth borrowing
1. **Methodology hygiene to emulate:** walk-forward with embargo, measured
   survivorship correction (backfill delisted names, quantify coverage), an
   explicit QQQ test, and frank front-loading/drawdown disclosures.
2. **The crash-recovery effect is real but episodic** — a coherent product
   ("amplify systemic sell-off recoveries") if and only if you can stomach
   −56% and commit through the crash. Not a steady-state QQQ-beater.
3. **Zero-shot time-series foundation models (Chronos) as a *filter*** is a
   neat idea, but treat any model selected on the OOS splits as in-sample until
   re-validated on a truly untouched cohort (their zero-shot NDX application is
   the clean test — and it's where it loses the mega-cap windows).
4. **Always benchmark vs QQQ.** A "100% beats SPY" headline is the #1 tell that
   growth beta is being mistaken for skill.
