# The Final Recommendation (verdict.html §2.11) — summary for the story edition (#c7)

**The signed default: 50% QQQ + 50% SPY (or VTI), no ballast, automated every
payday, forever.** Profile-scoped: 100% VTI/VT for zero-belief investors;
100% QQQ permitted only under three explicit signatures; gold 0–10% only if
the −33% stomach test fails; BTC 0–5% only against a personally-signed premium.

## The four measured results behind it (all reproducible: scripts/verdict_final_rec.py)

1. **Equal-belief Kelly = 0% QQQ.** σ(QQQ)=23%, σ(SPY)=15%, ρ=0.91. At equal
   expected returns the growth-optimal QQQ weight is exactly zero; geometric
   drag (½σ²) gives QQQ a standing **+1.5 pp/yr hurdle** vs SPY. Every dollar
   of QQQ is a priced belief, not a conclusion.
2. **The exclusion audit (2016–26, top-500 stocks):** QQQ's rules (no NYSE
   listings, no financials) excluded **14 of the top 25 large winners** and
   **59% of all large-winner gains** (TSM +2,100%, LLY +1,539%, CAT +1,531%,
   MS +1,038%…). QQQ beat SPY anyway via double weight on one ticket
   (NVDA +18,180%). 100% QQQ = betting the next era's dominant ticket is again
   Nasdaq-listed & non-financial — paid twice (MSFT-90s, NVDA-20s), failed
   once (2000s). Fallback asymmetry: SPY holds every QQQ winner; QQQ holds
   none of SPY's exclusives.
3. **Market evolution:** rolling 36m QQQ–SPY correlation flat ~0.9 for 25
   years (no convergence in co-movement; the *rules* never converge). Top-10
   concentration of large-cap dollar volume: 38% (2005) → 11% (2015) → 32%
   (2025) — tripled since 2015, and the one prior peak fully unwound in a
   decade. Feeling obvious at the concentration peak is the 1999 feeling.
4. **Contributor asymmetry:** worst-case QQQ entry (2000 top): fell to 0.58×
   of the SPY contributor, 8.4 years behind, ended 2.08× ahead. The bet's
   downside is bounded relative regret, not ruin — which is what licenses it.

## Why 50/50 specifically
Owns every Nasdaq winner through BOTH halves (SPY holds them too — the split
halves only the *extra* weighting); owns the excluded-59% pool at full weight;
expresses the concentration belief at exactly half size against the 1.5pp
hurdle; never the worst machine in any measured era; regret bounded in both
directions. SPY-vs-VTI within it: measurably indifferent (0.99 corr, ~87%
overlap).

## The three signatures required for 100% QQQ
(1) "QQQ beats broad by ≥1.5 pp/yr indefinitely." (2) "The next dominant
winner is again Nasdaq-listed & non-financial." (3) "If wrong, I accept
permanently less, with no fallback." Sign all three → coherent owned bet.

Full derivation chain: verdict.html §2.1–2.11 (impossibility proof → regret
asymmetry → world menu → qualification rubric → forever-combination →
zero-information derivation → candidate table → this recommendation).

## Addendum — dip arithmetic (verdict.html §7b, provenance: scripts/verdict_dipboost.py)

**The dip-boost (WORKS — extra money only):** per-dollar terminal multiples on
QQQ 1999–2026: ordinary DCA dollar 12.1×; extra dollar added at ≥−5% dip
16.1×; at ≥−20% **18.9× (+56% per dollar)**; at ≥−40% 21.3×. Fixed-horizon
sweet spot: the −25…−40% band (+19.8%/yr forward-3y, 96% positive). Rule:
tiered boost — add spare cash at −15…−20%, add more below −30%. Caveats:
boost = ADDITIONAL money (the base plan never pauses); dip months cluster
(half of QQQ's months were ≥20% under a peak).

**The reserve control (FAILS):** withholding 25–50% of regular contributions
to deploy at −10/−20/−30% dips loses to plain DCA in every configuration
(−1.3% to −5.1%). Extra money at dips: always. Withheld money for dips: never.

**The "sell high / too expensive" model (CATASTROPHIC):** sell on +60/80/100%
trailing-2y run-ups, re-enter at −15/−20/−30% dips: final wealth **−58% to
−87% versus never selling** — the largest wealth destruction of any behavior
modeled in the study; 110–217 months spent in cash. "Expensive" is not a
timing signal: QQQ spent most of its winning years looking expensive.

**The ATH fear check:** buying at all-time highs → forward-1y mean **+17.5%,
81% positive**, vs +13.4%/80% from all months. Highs were better-than-average
buy points.

---

## Addendum 2 — "Why not just buy Apple?" (the single-stock case, argued WITHOUT volatility)

Audience note: the paper's readers can already hold through pullbacks —
drawdown/volatility arguments are real but not compelling to them. The case
against single stocks must stand with iron hands granted. It does, five ways
(verdict.html §4.4; provenance `scripts/verdict_singlestock.py`, all committed
data — `data/stocks/AAPL.csv` is Tiingo daily 1981–2026):

1. **Holding power fixes drawdowns, not endings.** Full delisting-inclusive
   panel, 18,100 stocks (≥1yr history, $5+ peak): **36% no longer trade**;
   **25% ended or stand ≥70% below peak** (8% died there — permanent).
   JPM Agony & Ecstasy: 40% catastrophic never-recovered declines; median
   stock −54 pts lifetime vs index. You can hold through everything except
   an ending, and endings are common.
2. **"It'll always go up" carries no information.** The 10 most valuable US
   companies of March 2000, held 26 years with dividends reinvested:
   **only 3 beat QQQ (7.38×)** — Microsoft 12.3×, Walmart 11.5×, XOM 8.2×.
   The era's most Apple-like names all lost: GE 3.1×, Cisco 2.7×, Intel 3.9×,
   Nokia 0.52×, Citigroup 0.75× (two lost money outright). Same argument-form
   that selected Japan-1989 / financials-2007 (§2.6); top-dog cohorts 11/40.
3. **Even the RIGHT company loses for longer than conviction survives.**
   Apple (2,825× since 1981 — the best case in modern history): bought
   September 2012 at peak obviousness (largest company in history), the buyer
   spent **7.3 years losing to QQQ (Sep 2012 – Dec 2019, −44% at the trough)**
   before winning (15.8× vs 11.4×) in one late burst. Earlier: 8.4 years
   underwater 1991–99 (−81%), ~90 days from insolvency in 1997.
4. **Variance drag is a fee courage can't waive.** g ≈ μ − ½σ². Apple's
   calmest decade: σ 27.4% → 3.7 pp/yr drag (QQQ 1.8, SPY 1.2). Apple full
   history: σ 43.3% → 9.4 pp/yr. A typical single stock must out-earn the
   index by 5–10 pp/yr forever merely to tie in compound growth.
5. **QQQ is not "instead of" Apple.** The index holds Apple at full market
   weight (cap-weighting promoted it to the top as it won) PLUS an automatic,
   tax-free claim on its successor — as it rode MSFT→AAPL→NVDA without anyone
   choosing. Apple-only is a bet that succession never comes; every cohort
   table says succession always comes. The stock holder must be right
   forever; the index holder needs no opinion.

The FAQ "I'll just buy NVIDIA / Apple / the obvious winner" now routes to
§4.4 with this five-point summary.
