# BEDROCK — literature-driven design for a post-GRANITE fixed-income strategy

*2026-08. An extensive sweep of academic literature, industry research, and live
practitioner records across fixed-income sectors — six parallel research tracks
plus targeted deep-dives (~60 primary sources) — filtered against the hard
constraint of what our data can actually implement, and synthesized into a
pre-registered strategy design intended to beat GRANITE-XL. Nothing here has
been backtested yet; this document freezes the specs BEFORE any results are
produced, in the repo's established discipline.*

---

## 0. The mandate and the honest starting bar

GRANITE-XL, at audited real-coupon numbers, delivers **OOS ≈ +12.6–14.2% CAGR,
Sharpe(m) 0.84–0.86, annual-frequency Sharpe ~0.6, maxDD −35%**, on one idea:
buy idiosyncratic forced-seller dislocations at the ask, harvest reversion at
the bid. To beat it we need either (a) better entries for the same idea, (b)
additional *independent* return streams that diversify its drawdown, or (c)
both. The literature sweep was aimed at exactly this question.

**The capacity insight that frames everything:** Robeco's flagship multi-factor
credit fund — the cleanest live factor-credit record in existence — earned
**+45bp/yr net** (IR ~0.7) over its benchmark; their live HY factor fund earned
**−31bp/yr**. Institutional factor investing in credit works at tens of basis
points. GRANITE-type books earn 10–20× that because they harvest a
**capacity-limited liquidity premium at retail/odd-lot size** that
institutions structurally cannot reach (and post-Volcker dealers abandoned —
Bao-O'Hara-Zhou 2018; Bessembinder et al. 2018; Anand-Jotikasthira-Venkataraman
2024 show the liquidity premium migrated to patient customers). The lesson:
**do not pivot to factor tilts; deepen the liquidity-provision + structural
franchises where small size is an advantage.**

## 1. The binding meta-result: the bond factor zoo is mostly fake — and what survives

The single most important input comes from the team that builds our own data
(OSBAP): Dickerson, Robotti & Rossetti, **"The Corporate Bond Factor
Replication Crisis"** (2026), plus Dickerson-Mueller-Robotti "Priced Risk in
Corporate Bonds" (JFE 2023, which got Bai-Bali-Wen **retracted**) and
Dick-Nielsen-Feldhütter-Pedersen-Stolborg "Replication Failures" (2023).
Testing **108 signals × 4 constructions**:

- Only **~6% of specifications survive** false-discovery correction.
- **Short-term reversal is >90% microstructure artifact** (the same noisy price
  sits in the signal and the return denominator): −0.99%/mo collapses to
  −0.09%/mo, t: −4.46 → −0.51.
- **Momentum is a look-ahead artifact**: 6-month momentum's +0.30%/mo goes to
  zero without asymmetric ex-post winsorization; 12-month momentum is
  *negative* (−0.13%/mo) before ex-post filtering flips it.
- Volatility/downside-risk factors: 57–78% of alpha is ex-post filtering.
- **The survivors: credit-spread-based VALUE and SPREAD-CHANGE signals** —
  retaining ~half their premium, still significant — plus Dickerson-Nozawa's
  spread × duration "credit primitives" factor, which survives market-risk
  controls *and* transaction costs.

Two implications. First, methodological: their prescribed hygiene — signal
computed on prices ≥1 day before the execution price, fills at the touch,
ex-ante-only filters — is **exactly what our engine already does** (signal day
t, entry at the next ask print, exits at bids). Our platform is
replication-crisis-proof by construction, which is rare and valuable. Second,
directional: **our panel carries a `credit_spread` field that GRANITE never
uses cross-sectionally. The one factor family blessed by the harshest referees
in the field is sitting unused in our own dataset.**

## 2. Literature map — what's real, at what magnitude, and what our data can implement

Implementability (1–5) is scored against our actual panels: corporates = daily
per-CUSIP clean price, bid, ask, yield, credit spread, **exact maturity date**,
volume, 2002–2025 (no ratings, no fundamentals, no equity link, issuer =
CUSIP6); munis = full per-CUSIP EMMA trade tape with price/yield/size/side,
coupon, maturity, state, 2005–2026.

### 2a. Forced selling & liquidity provision (GRANITE's home turf — deepen it)

| Finding | Source | Magnitude / horizon | Impl. |
|---|---|---|---|
| Liquidity premium migrated from dealers to patient customers post-Volcker; liquidity-supplying funds earn alpha | Anand-Jotikasthira-Venkataraman (Mgmt Sci 2024); Bessembinder et al. (JF 2018) | thesis-level | **5** (is GRANITE) |
| Insurer regulatory fire sales revert from ~week +16, i.e. 4–8 months | Ellul-Jotikasthira-Lundblad (JFE 2011) | several pts | 3 |
| Sell-herding distortions revert over ~2 quarters | Cai-Han-Li-Li (JFE 2019) | "substantial" | 2 |
| Forced selling identifiable from price dispersion ALONE (no flows/ratings needed) | Feldhütter (RFS 2012) | flags GM/Ford '05, 2008 | **4–5** |
| Price drop alone ≠ fire sale — demand volume/spread stress confirmation | Ambrose-Cai-Helwege (2008/12) | discipline | **5** |
| In systemic crashes the LIQUID/SAFE end cheapens most (pecking-order inversion) — a regime flag | Ma-Xiao-Zeng (RFS 2022); Haddad-Moreira-Muir (RFS 2021) | IG costs 3×, LQD −5% NAV | **4** |
| Muni retail panics (2010, 2013, 2020, 2022): 100–365% M/T ratio spikes, reversion weeks→quarters | NY Fed; Brookings | 5–15 pts in tails | 4 |

### 2b. Index mechanics & calendar flows (new territory, high fit)

| Finding | Source | Magnitude / horizon | Impl. |
|---|---|---|---|
| **Sub-1y maturity index exclusion**: trackers sell mechanically at month-end; concession ~34bp+ (doubled post-2008 for short IG, tripled for HY); reversion days–weeks; **event date deterministic from maturity + calendar** | Dick-Nielsen & Rossi (RFS 2019) | 30–100bp per event; self-liquidating exit (hold to maturity → zero exit cost) | **5** |
| Fallen-angel overshoot: −13% avg around downgrade, ~120–150bp excess spread at HY-index entry, recovery 6–24mo; selling starts BEFORE downgrade; live-validated (FALN/ANGL beat HY in 15/22 years) | Ben Dor & Xu (JFI 2011); Insight; VanEck | multi-pt; months | 3.5 (ratings-free proxy: spread-jump + volume + month-end signature) |
| Treasury month-end index-extension effect: last-days-of-month returns, Sharpe ≈1 standalone | Hartley & Schwarz | ~13bp/day EOM | 5 (context only — not our book) |
| Muni reinvestment season: Jun/Jul/Aug redemption waves ($98B Jul–Aug) vs light supply; May/Jun/Jul index +0.83/+0.43/+0.82%/mo (10y avg); April tax-selling trough | LPL; Bond Buyer; BlackRock | 50–150bp/yr as timing | **4** (overlay) |
| Quarter-end dealer balance-sheet contraction (repo +10bp Q-end, +97bp Y-end) → provide, don't demand, liquidity at quarter ends | Fed Notes; NY Fed | execution overlay | 3 |
| Issuer-curve issuance pressure: existing bonds cheapen ~9bp on a sibling mega-deal, revert ~6 weeks | Helwege-Wang (JFI 2021) | 9bp / 6wk | 4 (as entry-timing) |

### 2c. Structural tax mispricing (munis — the largest untouched edge)

| Finding | Source | Magnitude | Impl. |
|---|---|---|---|
| **De-minimis cliff**: market-discount munis trade ~25bp cheap (37bp interdealer below threshold); implied tax rates ~2–3× statutory — **near 100% in retail prints vs ~40% true** — i.e. most of the concession is EXCESS return for tax-indifferent capital; fair-value drop of 0.5–1+ pts crossing the cliff | Ang-Bhansali-Xing (JF 2010); Fidelity; PIMCO; Kalotay | 25–80bp yield / 1–3 pts price | **5** |
| 94% of munis are issued at premium precisely to avoid the cliff → the 2022 rate shock pushed the low-coupon stock through it: **our own tape shows below-threshold bond-days exploding from ~hundreds/yr pre-2022 to 17k–56k/yr in 2022–26 (13.2% of all bond-days; 1,131 of 3,085 bonds)** | Landoni (JFE 2018) + our data | the trade's era is NOW | **5** |
| Muni retail markups 2.30% (≤$100k) vs 0.16% (>$500k); post-2018 disclosure + SMA growth compressed but didn't kill it; institutional dealers now do 44% of odd-lots — **regime-split all muni backtests at 2018/2021** | Green-Hollifield-Schürhoff (RFS 2007); Griffin et al. (JF 2023); MSRB | cost model + caveat | — |
| Odd-lot/micro discounts: corp micro-lots ~104bp vs 13bp blocks (2015); munis retail ~3× institutional and WIDENING (2023) | O'Hara et al.; MSRB 2023 | structural, pro-small-size | **5** (already implicit in our engines) |

### 2d. Factors & ML (use sparingly, on the blessed family only)

| Finding | Source | Verdict for us |
|---|---|---|
| Spread-based value (vs maturity/vol/issuer-fitted fair spread) survives replication-crisis hygiene at ~½ premium; ~20–40bp/mo gross, meaningful net at low turnover | Dickerson et al. 2026; Robeco "True Value" 2024; Ben Dor ESP | **Adopt as entry gate** (impl. 4–5) |
| Spread × duration interaction ("credit primitives") survives costs | Dickerson-Nozawa | Adopt in sizing (impl. 5) |
| IPCA: spread, duration, trailing vol carry most cross-sectional information | Kelly-Palhares-Pruitt (JF 2023) | Confirms feature set (impl. 5) |
| ML adds mostly macro-state × spread/duration timing, not selection; OOS cross-sectional R² with price-only features ~0.3–1%/mo | Bell-Kakhbod-Lettau-Nazemi (NBER 2025); Feng et al. (JBF 2024) | Keep existing walk-forward ranker for capacity allocation only |
| Momentum, short/long-term reversal, vol-sorted factors: artifacts; equity-linked signals (equity momentum, ΔIV): real but need data we don't have | replication trio; Cao et al. | **Reject** |
| High-turnover strategies hit capacity fast; slow spread/carry strategies survive at scale | Ivashchenko-Kosowski (FAJ 2024) | Keep holds long |

### 2e. What delivered LIVE (the practitioner scoreboard)

Robeco GMFC +45bp/yr net; Conservative Credits +48bp/yr (Sharpe 0.76 vs 0.50);
Robeco MF High Yield **−31bp/yr**; iShares HYDB (defensive+value HY)
**+0.9%/yr vs HYG with smaller drawdowns** — the best live factor-ETF result;
fallen-angel indices (FALN/ANGL) multi-%/yr over broad HY across cycles — the
strongest live anomaly in credit; PIMCO's "structural alpha" list (new-issue
concession, roll-down, off-the-run, vol selling) is real but allocation- and
balance-sheet-gated — closed to us except roll-down. Pattern: **defensive
tilts, forced-seller capture, and execution-cost engineering survived contact
with reality; generic factor zoos didn't.**

## 3. Design: BEDROCK — a four-sleeve liquidity-and-structure book

Everything the sweep validated points to one architecture: keep GRANITE-XL's
engine and honesty rules, add the two blessed cross-sectional signals to its
entries, and bolt on three *independent* capacity-limited franchises the data
already supports. Working name: **BEDROCK** (the layer under granite).

### Sleeve V — GRANITE-XL v2 (corporate dislocation core, upgraded)

Frozen upgrades to the existing entry stack, each literature-motivated:

1. **Spread-value gate** *(Dickerson survivor #1)*: at signal time, fit
   cross-sectionally (weekly, trailing data only) log(cs) ~ f(maturity bucket)
   + trailing-90d return vol bucket; require the bond's residual ≥ 0 (bond
   cheap vs peers). Kills dislocations in already-rich bonds.
2. **Spread-change confirmation** *(survivor #2)*: require the bond's cs to
   have WIDENED over the trailing 20 trading days (the dislocation is
   credit-priced, not a stale-quote artifact).
3. **Stress confirmation** *(Ambrose-Cai-Helwege)*: require entry-day volume
   ≥ 2× the bond's trailing-90d median volume OR a two-sided print day with
   widened spread — a fire-sale signature, not just a low print.
4. **Issuer-curve depth** *(our own flowml finding + curve-RV literature)*:
   depth measured vs the CUSIP6 issuer curve (bond's dislocation minus the
   median contemporaneous dislocation of sibling CUSIPs) — isolating
   bond-specific forced selling (+7.32%/trade vs +4.45% issuer-wide in our
   prior study).
5. Mechanics from the audit: real coupons, lagged-mid recovery exit,
   position-based issuer lockout (the honest protocol), drift weights.

### Sleeve X — Index-exclusion immediacy (corporate, new)

Dick-Nielsen-Rossi's natural experiment as a strategy. Universe: IG-proxy
bonds (cs below the historical IG/HY boundary bucket) with **exact maturity
date** crossing 1.0y. Window: the 15 trading days before the month-end on
which remaining maturity < 1y (the index drop date). Entry: limit-style —
only if the ask prints ≥ X points below the bond's trailing 20-day median mid
(X pre-registered at 0.30, the documented concession scale). Exit: first bid
≥ recovery target within 90d, else **hold to maturity** (sub-1y paper
self-liquidates at par — the exit leg is free, the single largest cost
advantage available in credit). Feasibility measured on our panel: hundreds of
bonds per year trade across the 1.0y boundary with executable asks in the
window (numbers in §5).

### Sleeve D — De-minimis cliff harvest (muni, new)

The Ang-Bhansali-Xing mispricing, in the biggest de-minimis regime in the
data's history. Universe: munis with price in the cliff zone [threshold − 3,
threshold + 1], threshold = 100 − 0.25 × ceil(years-to-maturity) (par-issue
approximation; OID revised-price refinement later). Entry at customer-buy
prints ≥ K points below trailing 60-day median mid (KEYSTONE's signal)
**and** below the de-minimis threshold — buying the double concession
(dislocation + tax cliff) from tax-motivated sellers. Exit: KEYSTONE recovery
logic, or price re-crossing above threshold + 0.5 (the convex snap-back when
rates rally), or 455d stop. The tape shows 149k below-threshold bond-days,
63.7k in the cliff zone, 92% of them 2022+ — a live, current edge KEYSTONE
never distinguishes. Tax note: the excess is real alpha only for
tax-indifferent capital (IRA/institutional); disclosed, not hidden.

### Sleeve A — Fallen-angel proxy (corporate, exploratory)

Ratings-free event detector: cumulative 20d return ≤ −8%, volume ≥ 3× trailing
median, cs migrating from the IG bucket into the HY bucket, month-end
proximity. Entry at the first post-event ask; hold 12–18 months (the
documented overshoot-bleed horizon), recovery exit allowed after 6 months.
Admitted to the book only if it passes IS gates — the event inference is the
risk (Ambrose-Cai-Helwege's null result says information and pressure are
entangled at downgrades); we expect fewer, larger positions.

### Overlays (all sleeves)

- **Calendar throttle** (muni sleeves): scale entries up in the March–April
  supply/tax trough, ease exits into the Jun–Aug reinvestment wave; corp
  quarter-end entry preference. Zero incremental turnover — timing of trades
  we'd do anyway (the only cost-surviving use of seasonality).
- **Systemic-stress regime flag**: cross-sectional bid-ask widening +
  dispersion + the Ma-Xiao-Zeng quality-inversion signature → in flagged
  regimes, defer non-cliff entries N days (dislocations cluster and deepen;
  GRANITE's GFC/2022 lesson) and tighten the depth threshold.
- **Combination**: MOSAIC inverse-vol across sleeves (frozen rules, 24m
  lookback, no leverage), exactly as already built in `corps/research/combine.py`.

## 4. Why this should beat GRANITE-XL (and how it can fail)

**Per-trade quality** (Sleeve V): the spread-value + stress gates are the two
signal families that survived the replication crisis, applied as filters on an
entry family we know earns +4–6%/trade excess. If they raise mean/trade by
even 50–100bp at similar breadth, book CAGR rises 1–3pp.
**Diversification**: Sleeves X and D monetize *mechanical* counterparties
(index trackers, tax-motivated retail) rather than distressed credit sellers —
their P&L should be nearly uncorrelated with credit drawdowns (X is
self-liquidating short-duration IG; D is convex to rate *rallies*, the exact
regime that hurt KEYSTONE in 2022). GRANITE-XL's maxDD is −35–42%; a book
where 30–50% of capital sits in X/D should cut that materially while keeping
double-digit CAGR.
**Failure modes, honestly**: V's gates could shrink breadth below viability
(fills are the scarce resource); X's concession may already be arbitraged
post-2019 (electronic odd-lot algos) — Bloomberg even suspended the sub-1y
rule in March 2020, a regime risk; D's excess is partly a genuine tax
liability priced by the marginal holder and the SMA institutionalization of
retail flow is compressing muni edges post-2021 (regime-split everything); A
may be uninferable without ratings. Each sleeve gets a kill gate.

## 5. Data feasibility (verified on our own panels, this session)

- **Sleeve X data gap found during feasibility probing (disclosed):** the
  committed corp panel stores maturity as **integer years** (1–97; a
  dtype-mangled column), so the day-precise 1.0y boundary date cannot be
  computed from it, and OSBAP's index-eligible universe may drop bonds once
  they fall below 1y (the post-exclusion reversion leg would then be
  unobservable in-panel — though a hold-to-maturity exit at par + coupon is
  computable analytically, with default-risk caveats). **Prerequisite for
  Sleeve X:** re-extract exact `bond_maturity` (float years / maturity date)
  per CUSIP from the OSBAP source parquet — a one-field, one-time job — and
  verify sub-1y coverage before any backtest. The 2.0y anniversary (the int
  field flipping 2→1) plus 365 days gives an approximate boundary date in the
  meantime, adequate for the event study but not for publication.
- Muni tape: **149,070 below-de-minimis bond-days (13.2% of all), 63,675 in
  the cliff zone, 1,131 distinct bonds; 92% of the exposure is 2022–2026** —
  the sleeve trades in the current regime, not a backtest ghost.
- Credit spread (`cs`) present on the corp panel for the value/spread-change
  gates; volume present on both panels for stress confirmation.
- Everything else (issuer curves via CUSIP6, real coupons via `coupon_inv`,
  honest MTM, controls, bootstrap) already exists in the engine.

## 6. Pre-registered validation protocol

The repo's discipline, unchanged: IS = corp 2003–2015 / muni 2012–2022; one
locked config per sleeve; **one** OOS look each (corp 2016–2024, muni
2023–2026 censor-safe); matched controls face every fill cap; two-leg cluster
bootstrap; live-protocol replay (position-based lockouts, lagged triggers,
real coupons) BEFORE publication, not after; regime splits at 2018/2021 for
muni sleeves. Kill gates (pre-registered): a sleeve is dropped unless IS
excess vs its matched control is positive with p<0.01 AND the live-protocol
replay retains ≥60% of pipeline CAGR AND (for X, D) the effect exists in the
post-2019 subsample. Combination is rules-only (inverse-vol); fitted weights
never travel to OOS.

**Honest expectation** if the literature magnitudes hold at our sizes: book
CAGR in the mid-to-high teens with maxDD nearer −20–25% than GRANITE-XL's
−35–42% — the drawdown improvement, not the headline, is the point. If V's
gates add nothing and X/D die in their kill gates, we keep GRANITE-XL and will
have bought certainty for the cost of three backtests.

## 7. Execution plan

1. `corps/research/bedrock_v.py` — Sleeve V gates on the existing engine
   (spread-value residual, spread-change, stress confirmation, issuer-curve
   depth), IS only.
2. `corps/research/bedrock_x.py` — FIRST re-extract exact maturities from the
   OSBAP source (see §5 data gap) and check sub-1y panel coverage; then the
   exclusion-window event study (does the Dick-Nielsen-Rossi concession exist
   on our tape at all?); only then the sleeve, IS only.
3. `munis/research/bedrock_d.py` — de-minimis event study (regression
   discontinuity at the threshold on the tape, retail vs interdealer), then
   the sleeve, IS only.
4. `corps/research/bedrock_a.py` — fallen-angel proxy detector precision test
   against known 2005/2008/2020 episodes, then sleeve, IS only.
5. Lock configs → single OOS batch → live-protocol replay → MOSAIC
   combination → publish with the same audit standards as XL_AUDIT.md.

## 8. VALIDATION RESULTS (2026-08, this repo, pre-registered protocol §6)

Everything below was produced after the specs in §3 were frozen and pushed.
Two implementation bugs were found and fixed during validation (disclosed):
a `np.bool_ is True` filter bug that initially zeroed Sleeve D's IS run, and
Sleeve A's spread thresholds written in percent against a cache that stores
credit spreads in decimal.

### Sleeve X — index-exclusion immediacy: **KILLED by its event study**

The truncated-at-1y cohort exists exactly as hypothesized (OSBAP global min
maturity 1.002; **21,142 bonds** truncate at the boundary — the exclusion
population). But the pre-registered diff-in-diff (bond cs vs own [-180,-120)d
baseline vs market, buckets to [-10,+1)d before the 1.0y crossing) is
**economically zero** (≈0–0.1bp, max t=1.85), and the final-10-day ask-side
YTM pickup has median ~0 (mean ~+10bp, p75 ~+40bp — a thin right tail only).
Interpretation: the Dick-Nielsen–Rossi concession is a *trade-level execution*
phenomenon (sellers crossing to thin bids around the drop date); it does not
survive into daily VWAP levels, so our panel cannot harvest it. No backtest
was run; the sleeve dies at the identification stage — which is exactly what
the event-study-first protocol is for.

### Sleeve D — muni de-minimis cliff: **event study CONFIRMED; admitted to one-shot OOS**

- **T1 within-bond discontinuity**: 30,960 below/above pairs (±120d, ±2pts)
  across 789 bonds: just-below prints yield **+15.1bp** more than the same
  bond's just-above prints (bond-clustered mean +17.2bp, **t = 13.9**) —
  the Ang-Bhansali-Xing cliff, live in our tape.
- **T2 monotone gradient + retail overpunishment**: own-bond excess YTW rises
  monotonically through the threshold (+5.8bp far above → +21.9 just above →
  +27.9 just below → +54.7 deep below), and the customer-buy-over-interdealer
  gap **peaks in the cliff zone** (1.19–1.22pts vs 0.78 far above) — retail
  pays the widest markups exactly where the tax penalty bites.
- **T3 forward returns (engine conventions)**: cliff-zone entries beat
  above-zone entries by **+1.36pp per ~1y hold in BOTH regimes**
  (2013–21: +5.78% vs +4.42%, n=1,223; 2022–26: +3.55% vs +2.19%,
  n=25,248). Disclosure: the 2022–26 split overlaps the strategy OOS window
  at the *event-study* level; the strategy-level OOS remains unrun.
- **Strategy IS screen** (KEYSTONE stack ∩ cliff zone, 2012–2022): **n=14**
  (thin by construction — 92% of the cliff universe is post-2022), mean
  +4.97%, win 79%, excess +2.97% (p=0.13, underpowered) vs complement
  +4.53%/71%. Sign-consistent; per the pre-registered contingency the
  event study carries IS identification and the sleeve proceeds to its
  **single** strategy-level OOS test (2023–2026, censor-safe, cap-matched
  control), reported below when run.

### Sleeve V — gated GRANITE-XL core: **4-gate stack REJECTED; disclosed V2 iteration**

IS 2003–2015, real coupons + lagged recovery exits (the audited honest
conventions), baseline = GRANITE-XL entries ungated:

| book | n | mean/tr | CAGR | Sharpe(m) |
|---|--:|--:|--:|--:|
| baseline | 2,012 | +5.41% | +17.68% | 1.08 |
| +G1 spread-value | 1,547 | +5.56% | +19.84% | 1.14 |
| +G1+G2 spread-change | 1,395 | +5.03% | +15.34% | 0.97 |
| +G1..G3 stress | 447 | +4.86% | +11.68% | 0.74 |
| **+G1..G4 (pre-registered stack)** | 338 | +5.34% | **+11.65% / 0.68 → REJECT** | |

Ablations (pre-registered diagnostics): **G1 spread-value alone (+19.84%/1.14)
and G4 issuer-curve alone (+20.08%/1.16, mean +6.11%) each beat baseline**;
G2 spread-change and G3 stress destroy breadth/performance — the fire-sale
"stress confirmation" idea, sensible in theory, kills 70% of fills and with
them the book. A **V2 = G1+G4 combination** is being screened as a *disclosed
second iteration* (selection risk acknowledged; its own kill gate incl. a
real-coupon-matched control, one OOS look only if passed). The v1 stack
remains rejected regardless.

### Sleeve A — fallen-angel proxy: **IS PASSED strongly; admitted to one-shot OOS**

Detector (after the disclosed decimal-units fix): **11,905 events across
8,473 bonds**, clustering precisely where fallen-angel history says they must
— 2005 autos (767), 2007–09 GFC (527/3,608/1,634), 2020 COVID (2,395) — face
validity without any ratings feed. IS returns (2003–2015 entries, ~1y holds,
real coupons): **n=5,161, mean +15.88%, win 81%, excess vs matched control
+4.52% (p<0.001)** — and that excess is conservative, since the control's
carry used the higher median-YTW proxy while the strategy leg used real
coupons.

### V2 kill-gate repair (disclosed in full)

The v1/v2 scripts' third condition — per-trade excess vs a 1-year-hold
control — was structurally unpassable for recovery-exit books (~250d holds vs
~380d control holds, the exact mismatch XL_AUDIT §6b documented): even the
untouched baseline printed "negative excess" under it. The condition was
re-evaluated the correct way — **hold-matched**: same entries, the 1y-hold
exits they were generated with, real coupons on both legs. Result: baseline
entry-excess **+5.27% (p<0.0001)** — sanity-consistent with GRANITE-CL's
known IS excess — and **G1+G4 entry-excess +6.88% (p<0.0001)**, +1.6pp of
genuine entry-quality improvement. With the repaired metric, V2 passes all
three IS conditions (Sharpe 1.20 > 1.08; CAGR +21.7% > +17.7%; excess
p<0.01) and is **admitted to the one-shot OOS**. Nothing about the v1 4-gate
rejection changes.

### One-shot OOS batch (specs frozen at this commit)

Admitted: **V2** (corp, 2016–2024, paired vs baseline + hold-matched excess),
**A** (corp, 2016–2024, real-coupon matched control), **D** (muni, 2023-01-01
→ 2025-04-08 censor-safe, cliff vs complement + cap-matched control).
Scripts: `corps/research/bedrock_oos.py`, `munis/research/bedrock_d_oos.py`.

### One-shot OOS results (reported as printed)

**V2 — PASSED, decisively (the new recommended operating point):**

| OOS 2016–2024, real coupons + lagged recovery exits | n | mean/tr | win | CAGR | Sharpe(m) | maxDD |
|---|--:|--:|--:|--:|--:|--:|
| baseline (audited GRANITE-XL conventions) | 2,729 | +5.98% | 79% | +14.77% | 0.89 | −35.1% |
| **BEDROCK-V (G1 spread-value + G4 issuer-curve)** | **1,499** | **+8.13%** | **86%** | **+16.65%** | **0.96** | −37.0% |

Hold-matched 1y entry-excess vs real-coupon cap-matched control: baseline
**+5.17%** (p<1e-4), BEDROCK-V **+7.32%** (p<1e-4) — the gates add +2.2pp of
OOS entry quality with **no IS→OOS decay** (IS +6.88% → OOS +7.32%), on 55%
of the trades (fewer, better fills — capacity-friendlier). Both gates are the
signal family the replication-crisis literature blesses, doing exactly what
the literature said they would.

**A — per-trade alpha CONFIRMED; standalone book weak; no diversification:**
OOS n=3,227, mean **+19.41%**, win 95%, excess **+7.01%** (p<1e-4) — the
ratings-free fallen-angel proxy's timing alpha is real out-of-sample. But the
standalone MTM book earns only +6.65% CAGR / Sharpe 0.45 (events are
episodic; capital idles between crisis clusters), and its monthly correlation
with BEDROCK-V is **0.725** — blending dilutes (80/20 → +14.48%/0.92 vs V
alone +16.65%/0.96), replicating this repo's earlier finding that all
surviving credit sleeves share one factor. **Role: capacity-extension
satellite** (a second entry trigger with +7pp/trade alpha when the V book
cannot absorb capital), not a default allocation.

**D — REJECTED at the strategy level:** OOS cliff-zone entries n=47, excess
+1.39% (p=0.20), *underperforming* the complement (+5.82%, p<1e-4). The
de-minimis discontinuity is real (event study), but conditioned on a
dislocation entry the cliff tilt adds duration/rate exposure rather than
extra reversion — KEYSTONE cannot monetize it as an overlay. (A standalone
tax-indifferent discount-muni buyer is a different mandate; the event-study
finding stands as knowledge.)

**X — killed at identification** (§ above). **v1 4-gate stack — rejected.**

## 8b. Final verdict

**BEDROCK-V beats GRANITE-XL out-of-sample on every axis that survived the
audit**: +16.65% vs +14.77% CAGR, 0.96 vs 0.89 Sharpe(m), +8.13% vs +5.98%
per trade, 86% vs 79% win, +7.32% vs +5.17% entry-excess — same engine, same
honesty rules, two literature-blessed gates. Definition of the upgrade, in
full: *GRANITE-XL entries (≥3pt dislocation, ≤5y, issuer cap, limit entry)
PLUS (G1) signal-day credit spread at-or-above its same-day maturity-bucket
median, PLUS (G4) the bond's own dislocation ≥2pts deeper than the median of
≥2 sibling CUSIP6 bonds (pass-through when <2 siblings), with real coupons,
lagged-mid recovery exits, depth weights.* Everything else tested was killed
by its own pre-registered gate and is documented above.

## 8c. BEDROCK-V finishing pass (XL_AUDIT standard) — ALL GATES GREEN

Run after the OOS admission, `corps/research/bedrock_v_final.py`:

- **Live-protocol replay** (chronological admission, issuer capacity against
  ACTUAL open positions, limit filter before capacity, gates at signal time,
  real coupons): position-lock full **+13.33% / Sharpe(m) 0.81** (OOS
  **+13.32% / 0.73**, n=3,056 — retention 85% of pipeline OOS CAGR, passing
  the pre-registered ≥60% gate); tight-lock full +14.79%/0.86 (OOS
  +10.14%/0.58). Compare GRANITE-XL's own replays (+10.7–11.5% / 0.59–0.74):
  **BEDROCK-V beats the incumbent by ~+2pp CAGR at equal-or-better Sharpe
  under identical live accounting**, and the replay book is 2.4× the pipeline
  book (5,983 fills) — more capacity, not less.
- **Slippage grid** (full-window pipeline): h=0 → +18.07%/1.04; h=0.125 →
  +17.38%/1.00; h=0.25 → +16.70%/0.96; h=0.5 → +15.37%/0.88. Budget the
  0.125–0.25 row.
- **Era decomposition**: positive in all six eras; best vintages are the
  crises (2008–09 +13.87%, 2020 +12.86%); weakest 2004–07 (+0.46%, 73% win);
  2021–23 rate shock +4.63%/79% — no new failure mode vs GRANITE.
- **Gate perturbations (robustness, not tuning)**: CAGR 17.8–18.6% and
  Sharpe 1.04–1.05 across G1 at the ~40th/50th/60th cross-sectional
  percentile and G4 gap ∈ {1,2,3} pts — **no knife-edge anywhere**.

**Live planning numbers for BEDROCK-V** (the honest row): OOS CAGR ≈ **+13%**,
Sharpe(m) ≈ 0.73 (annual-frequency lower, per the stale-mark caveat), maxDD
≈ −40%, with slippage already graceful and capacity ~2× GRANITE-XL's.

## 8d. Two further event studies (both honest nulls — not adopted)

- **Muni seasonality** (`munis/research/bedrock_s_event.py`): MUB summer
  premium May–Jul vs Sep–Oct **+47bp/mo (t=1.71)** and vs Mar–Apr +56bp/mo
  (t=1.83) — right sign, marginal significance on 15 years; KEYSTONE
  entry-vintage spring-vs-summer +0.49pp (t=1.22, ns); and the tape shows
  **no** Dec/April seller-imbalance wave (sell share flat 46.5–47.9% across
  months — the mechanism signature is absent at our resolution). Calendar
  overlay **not adopted**; a zero-cost timing *preference* at most.
- **Issuance pressure** (`corps/research/bedrock_i_event.py`, Helwege-Wang
  replication; 34,953 new-issue events, 54,786 sibling observations, 2,560
  issuers): event-window sibling cheapening **+1.1bp all / +2.3bp large
  deals (t≤1.6)** vs the literature's ~9bp, and no reversion signature.
  Below our data's noise floor and any cost floor. **Killed.**

The program's box score across two research rounds: **8 ideas taken to
evidence, 2 adopted** (BEDROCK-V core upgrade; A as capacity satellite),
**6 killed or shelved by their own pre-registered tests** (X, D-overlay,
V-4-gate, spread-momentum gate, stress gate, calendar overlay, issuance
pressure). The kills are the reason the survivors can be believed.

## 8e. Drawdown-minimization round (PRE-REGISTERED 2026-08-12, before results)

**Goal:** cut BEDROCK-V pipeline maxDD (−44% full / −37% OOS; replay −48%/−41%)
while keeping the upside (OOS pipeline ~+16.6%/0.96; replay planning ~+12.5%).

**Step 0 — anatomy first** (`bedrock_dd_diag.py`, measurement only): episode
table, open-at-peak vs entered-in-fall realized-return decomposition, upside
attribution to crisis-window entries, monthly beta/downside-beta vs LQD/HYG,
and full-window overlay diagnostics used ONLY to rank levers (disclosed as
in-sample measurement, not evidence).

**Candidate levers** (chosen from prior evidence before any new results):
- **H (beta hedge):** short LQD (or duration/credit proxy) against the book.
  Rationale: per-trade excess vs matched controls is the alpha; the controls
  carry the beta, and beta is plausibly what draws down. Hedge ratio must be
  estimated IS only and frozen.
- **VT (vol targeting):** scale exposure to trailing realized vol, cap 1x
  (no leverage), cash earns T-bill. Known risk (prior repo result [V]):
  Sharpe up, CAGR down — the kill gate below guards this.
- **TR (market trend de-risk):** reduce exposure when LQD sits >x% below its
  trailing high. NOTE: a market-regime *entry gate* was already REJECTED once
  (overfit GFC, failed OOS). This variant differs (scales the whole book, not
  entries) but carries the same overfit risk; treated with extra suspicion.
- **ST (staggered deployment):** cap per-month new-entry count/weight to slow
  crisis-wave concentration. Risk: crisis entries are where the upside lives.

**What is NOT on the table:** stop-losses at bid during stress (realizes the
dislocation we're paid to hold), and any lever tuned on OOS data.

**Kill gates (frozen):** a lever is ADMITTED to the one-shot OOS only if, on
IS (2003–2015) with parameters frozen from IS data alone:
1. maxDD improves by ≥ 8pp (e.g. −42% → −34% or better);
2. CAGR gives up ≤ 2pp vs the unmodified BEDROCK-V IS book;
3. Sharpe(m) does not fall.
Survivors get ONE OOS look (2016–2023.12 censor-safe) reported as-is; an OOS
result that keeps ≥ 60% of the IS drawdown improvement and passes the same
CAGR/Sharpe conditions OOS is adopted; otherwise killed and reported.

**Addendum (2026-08-12, after the anatomy diagnostic, BEFORE any lever
backtest):** the diagnostic (`bedrock_dd_diag.json`) found (i) the worst
episode is 2014-08→2016-02 (−44.4%, energy bust), not the GFC; (ii) the
drawdown is a mark-to-market trough (open-at-peak cohorts small, realized
−2% to −13%; in-fall entries realize positive); (iii) 68% of trades enter
inside episode windows and carry ~76% of total return — ST is near-certainly
upside-destroying; (iv) monthly corr to LQD is 0.32 (R²=0.10) and every
hedge ratio WORSENS maxDD in the full-window diagnostic — H is near-certainly
dead; (v) vol-targeting moves nothing (vol is coincident, not leading);
(vi) in the 2014-16 bust, −3pt crisis entries realized only +1.5%/trade vs
+9.7% in 2020 — shallow entries in grinding busts are the weak cohort.
One lever is therefore ADDED before any lever backtest:
- **AD (adaptive depth):** when the tape-wide dislocation share (20d-smoothed
  share of customer-ask prints ≥3pts below med60) exceeds its IS q90 (frozen
  constant), require own dislocation ≤ −4 (variant: ≤ −5) at the signal row
  instead of −3. Normal tape unchanged. Same kill gates as the others.

**Second addendum (2026-08-12, after the IS screen, BEFORE testing):** all
five levers (H, VT, TR, ST, AD ×2 depths) FAILED the IS kill gates — none
moved maxDD by even 3pp in the right direction; ST worsened it to −56% and
cut Sharpe (`bedrock_dd_screen.json`). One final lever family is added,
qualitatively different because it changes portfolio construction across two
ALREADY-validated books rather than either book's signals:
- **BL (cross-market blend):** monthly-rebalanced KEYSTONE-XL (muni) +
  BEDROCK-V (corp) at fixed weights {50/50, 60/40, 70/30 corp share},
  evaluated on the published monthly NAV series, common window 2012-06+.
  Rationale: the books' crises differ (muni 2013 taper vs corp 2014-16
  energy / 2020), correlation is the only new estimate, and no signal or
  parameter is fitted. Judged against 100% BEDROCK-V on the same window with
  the same three gates (monthly-resolution maxDD, disclosed). A reference
  row of BEDROCK-V diluted with T-bills at equal CAGR give-up is reported so
  the blend must beat trivial de-risking, not just the undiluted book.

**RESULTS (2026-08-12; scripts `bedrock_dd_diag.py`, `bedrock_dd_screen.py`,
`bedrock_dd_blend.py`; JSONs alongside):**

*Anatomy.* The drawdown is a mark-to-market trough, not a permanent loss:
open-at-peak cohorts are small (17–87 positions) and realize −2% to −13%,
in-fall entries realize positive, and 68% of all trades — carrying ~76% of
total return at a HIGHER per-trade mean (+8.6% vs +6.2%) — enter inside
episode windows. The worst episode is the 2014-08→2016-02 energy bust
(−44.4%), not the GFC (−41.0%); COVID −37.0%; 2022 only −13.9%. Monthly
correlation to LQD is 0.32 (R²=0.10), alpha ≈ +1.2%/mo.

*Within-book levers: ALL KILLED on the IS screen.* H (LQD hedge at IS beta
0.57): DD unchanged-to-worse, CAGR −2 to −3pp. VT (10–12% vol target): DD
unchanged (vol is coincident with the fall, not leading — stale marks). TR
(LQD −5% trend de-risk): −2.8pp DD, short of the 8pp gate; and this family
already failed OOS once before. ST (4/mo entry cap): DD WORSENS to −56.1%
with Sharpe 1.07 < 1.20 — throttling entries removes the diversifying crisis
vintages, confirming the anatomy. AD (require −4/−5pt depth when tape stress
> IS-q90 13.6%): DD −45.2/−45.7% vs −44.8% base — deeper entries in stress
still mark down with the tape. The conclusion the four families point at:
**the drawdown IS the risk premium** — the book is paid ~+7pp/trade of
entry excess precisely for holding dislocated bonds through crisis marks
(Duffie 2010 slow-moving-capital, exactly as the literature says).

*BL cross-market blend: ADMITTED — the one lever that works.* On the common
published-NAV window (2012-07..2025-02, 152 mo), KEYSTONE-XL × BEDROCK-V
monthly correlation is **+0.02** (−0.24 in corp-down months), and it is
window-stable (+0.09 in 2016+, +0.14 in 2020+ and 2023+). Monthly
rebalanced, vs 100% BEDROCK-V (CAGR +10.53%, Sharpe 0.69, maxDD −40.7% on
this window):

| book | CAGR | Sharpe(m) | maxDD | vs gates |
|---|---|---|---|---|
| 50/50 corp/muni | +9.62% | 1.13 | **−16.8%** | ADMIT (+23.9pp, −0.9pp CAGR) |
| 60/40 corp/muni | +9.85% | 0.99 | **−22.1%** | ADMIT (+18.6pp, −0.7pp CAGR) |
| 70/30 corp/muni | +10.06% | 0.88 | **−27.2%** | ADMIT (+13.5pp, −0.5pp CAGR) |
| 50% BV + T-bills (ref) | +6.21% | 0.69 | −22.6% | dominated by every blend |

The blend beats T-bill dilution at every risk level (50/50 has both a
shallower DD than 50% dilution AND +3.4pp more CAGR), so the effect is real
diversification, not de-risking. In the corp-OOS decade the 50/50 DD is
−14.3% vs −33.6% for corp alone.

*Disclosures.* (1) maxDD here is monthly-resolution on the published NAV
series; daily DD runs ~4–6pp deeper (corp full-window: −44.4% daily vs
−40.7% monthly-common-window). (2) The common window overlaps both books'
IS periods; the correlation — the only estimated quantity — was checked on
the OOS sub-windows above and holds. (3) The muni series' smoothness partly
reflects stale odd-lot marks; its economic DD is somewhat understated, so
the blend DD is a floor estimate — but even doubling the muni book's DD
leaves the blend far ahead. (4) A 50/50 at scale is bounded by KEYSTONE's
odd-lot capacity (~$2–4M/yr deployable per the audit); at larger AUM the
feasible muni weight shrinks toward 70/30 or the muni sleeve saturates.

**Verdict:** no within-book lever can cut BEDROCK-V's drawdown without
paying for it — six families tried across two rounds, all killed by their
own pre-registered gates. The adopted answer is portfolio-level: run
KEYSTONE-XL and BEDROCK-V as ONE blended book at 50/50–70/30 (corp weight
by capacity), monthly rebalanced. Expected: ~keep the corp book's CAGR
within ~1pp while cutting maxDD by roughly half to two-thirds, Sharpe up.
This also matches the deployment plan already on the table (KEYSTONE live
via EMMA now, BEDROCK-V pending the TRACE feed).

*Owner decision 2026-08-12: KEYSTONE will run separately — the blend is NOT
the accepted mitigation. Round 3 below searches within BEDROCK-V only.*

## 8f. Drawdown round 3 — phase-selective entries (PRE-REGISTERED 2026-08-12)

**Hypothesis from the round-2 anatomy:** the drawdown is built by entries
made during the FALLING phase of a stress episode (tape stress share rising)
— they keep marking down before recovering, and in grinding busts they also
realize the weakest returns (2014-16 fall cohort +1.5%/trade vs book mean
+6-8%). Trough/recovery-phase entries carry strong returns. So: keep trading
crises, but only once the tape has stopped deteriorating.

**Stress state (frozen):** tape stress = 20d-smoothed share of customer-ask
prints ≥3pts under med60 (round-2 series); CRISIS = stress > IS-q90
(13.58%, frozen); RISING = stress today > stress 20 calendar days ago.

**Levers (all evaluated on the IS pipeline book, params frozen from IS):**
- **SS (stress-slope entry filter):** skip entries when CRISIS AND RISING
  (variant SS75: same with q75 threshold). Entries resume the moment the
  20d stress slope turns non-positive — no forecast, pure state.
- **FK (falling-knife, bond level):** in CRISIS, require the bond's own mid
  to have stabilized: mid(sig row) ≥ mid 5 prints earlier (variant: 3).
- **DE (deferred entry):** signals fired in CRISIS enter at the first ask
  print 30-44d later instead, only if still ≥3pts under the CURRENT med60
  and the limit rule (ask ≤ prior mid + 0.25) still passes. Normal-tape
  signals unchanged.
- **CC (distress cap):** in CRISIS, skip bonds whose value-gap
  log(cs) − bucket-median exceeds the IS-q90 of pipeline entries (drop the
  deepest-distress tail whose marks fall furthest).
- **MS (maturity shorten):** in CRISIS, require mat ≤ 3 (less spread
  duration = less mark-to-market per unit of tape widening).
- **SO (stress overlay):** portfolio overlay 0.5x exposure while CRISIS AND
  RISING, 1.0x otherwise (the TR idea but with the tape-native state).

**Diagnostic (IS only, printed before levers):** per-trade returns of IS
in-episode entries split by RISING vs not at entry, and by value-gap
quartile — confirms or kills the mechanism before the levers run.

**Kill gates:** identical to §8e (IS maxDD ≥8pp better, CAGR give-up ≤2pp,
Sharpe(m) not lower; survivors get ONE OOS look, adopt only if ≥60% of the
IS DD improvement holds with the OOS CAGR/Sharpe conditions met). The
known failure mode — regime gates overfitting the GFC and dying OOS — is
exactly what the one-shot OOS is for.

**Addendum (2026-08-12, after the §8f IS screen, BEFORE testing):** the
mechanism CONFIRMED ([F]: crisis&rising entries +3.2%/trade vs +8.0%
not-rising; deep value-gap quartiles weakest) but ALL entry-side levers
failed the DD gate — selection raises CAGR (SS +23.5%, DE +25.7%) yet
leaves the trough at −43% to −47%, because in a systemic episode every
held bond marks down together: entry selection changes what the book earns,
not what its marks do at the trough. The remaining implementable family is
therefore hedge-side, state-contingent (the constant hedge of §8e failed
by bleeding through recoveries; the state gate removes the bleed, and an
index short is executable without selling bonds at crisis bids):
- **SH (state-contingent index hedge):** short h × notional of HYG (or
  LQD) while CRISIS & RISING (state lagged 1 day; variants: crisis-only,
  q75 threshold), flat otherwise. h ∈ {0.5, 1.0}. Borrow cost ~1-3%/yr
  while on (state on ≪10% of days — negligible, disclosed); switch count
  reported. Diagnostic printed first: the share of each episode's fall
  that occurs state-ON (if the fall precedes the state, SH cannot work and
  is killed by construction). Same §8e kill gates; survivors get the same
  ONE OOS look.

**RESULTS (2026-08-12; `bedrock_dd_screen2.py`, `bedrock_dd_screen3.py`):**

*Mechanism: CONFIRMED.* IS in-episode entries made while tape stress is
RISING earn +3.22%/trade (win 76%); once the slope turns, +8.02% (win 73%);
normal tape +7.73% (win 91%). By value-gap quartile the deepest-distress
entries are weakest (Q3 +1.34%, Q4 +3.02% vs Q1 +9.54%).

*Entry-side levers: ALL FAIL THE DD GATE — and the failure is structural.*
SS (skip crisis&rising): CAGR +23.53% but DD −46.3%. SS75: +24.25%/−43.3%.
FK (bond stabilized): −46/−47%. DE (defer crisis entries 30-44d): CAGR
+25.66% — the best return book of the whole program — but DD −46.3%. CC
(drop deep distress): DD unchanged. MS (mat≤3 in crisis): −45.9%. The
lesson: **selection changes what the book earns, not what its marks do.**
In a systemic episode every held bond marks down together; skimming the
weak cohort shrinks and concentrates the book without lifting the trough.
On a fully-invested mark-to-market NAV, no entry filter can cut the
crisis drawdown.

*SH state-contingent hedge: DEAD BY CONSTRUCTION.* The coverage diagnostic
shows the states cover only 7.2% (q90) to 31.7% (q75&rising) of the
2014-16 fall and 30-55% of the GFC fall — grinding busts never hold tape
stress above the trigger for long. All 12 variants moved DD by at most
+0.9pp; LQD h=1.0 at crisis-level made it WORSE (−49.3%: LQD rallies with
rates inside credit crises while stale book marks keep falling). Slow stale
marks vs fast liquid hedges is a structural mismatch no state gate fixes.

**Round-3 verdict.** With KEYSTONE excluded by owner decision, the program
has now tested nine lever families (~30 variants) across three
pre-registered rounds: H, VT, TR, ST, AD, SS/SS75, FK, DE, CC, MS, SO, SH
— every one killed by its own gates. The consistent physics: the drawdown
is the mark-to-market trough of a fully-invested book of dislocated bonds;
its depth is a property of what is HELD in a systemic episode, invariant
to entry selection, and unhedgeable with liquid instruments against stale
marks. The −40%+ maxDD is the risk being paid for (entry excess ~+7pp/trade
vs matched controls; Duffie 2010). What remains implementable:
1. **Allocation sizing** (linear, always works): x% of capital in
   BEDROCK-V, rest T-bills, scales both CAGR-minus-rf and DD by ~x.
2. **The DD is not realized loss**: open-at-peak cohorts realize −2% to
   −13%; capital that cannot be forced to liquidate at the trough
   (no-redemption structure) experiences it on paper only.
3. **Tail options** (HYG puts / CDX payer options, small premium budget)
   are the one untested family — we have NO options data, so any backtest
   claim would be fabricated; flagged for live consideration only, with
   the caveat that they pay in fast crashes (2008/2020), not grinding
   busts (2014-16), which is our worst episode type.
4. *(Future, separate mandate)* SS/DE raised IS CAGR by +2-4pp with fewer,
   better trades. NOT adopted here — that would be repurposing a failed DD
   experiment post-hoc — but registered as a candidate RETURN-enhancement
   round with its own IS/OOS discipline if the owner wants it.

## 8g. Drawdown round 4 — no-trade trigger on a CAPITALIZED book
(PRE-REGISTERED 2026-08-12, before results)

**Why this can work where §8f could not.** All prior rounds scored levers on
the fully-invested NAV (daily-renormalized across open positions). Under
that convention a no-trade trigger cannot lift the trough: whatever remains
held still marks down together. A REAL portfolio is different: positions
are sized against fixed capital, cash earns T-bills, and if entries halt
during the deteriorating phase the book RUNS OFF into cash (holds are
~250–380d), so the trough is hit holding cash — which then redeploys into
the post-trough entries that §8f showed earn +8.0%/trade vs +3.2% in the
fall phase. The renormalized convention structurally hides this effect;
this round evaluates it with an explicit capital simulation.

**Capital simulation (convention, frozen):** capital starts at 1.0; a new
position is sized at NAV/K on its entry day and taken only if cash covers
it (chronological admission; skipped-for-cash entries reported); position
values follow the same honest mark path as `mtm_nav` (entry ask → mids,
stale flat, daily coupon accrual → exit bid); cash earns the 3M T-bill.
K ∈ {50, 100} — two capitalizations reported side by side; each trigger is
judged against the SAME-K no-trigger base. IS concurrency stats printed.

**Triggers (state lagged 1 day; thresholds frozen from IS):**
- **T1:** no new entries while tape stress > q90 AND rising (§8f state).
- **T2:** same at q75 (earlier, longer halts).
- **T3 (own-equity):** no new entries while portfolio NAV is >10% below
  its trailing 1-year high (needs no external data; the desk-native rule).
Runoff is passive — no forced sales, exits unchanged; halted-period
signals are NOT queued (missed, not deferred).

**Kill gates (same as §8e/§8f):** on IS at the same K: maxDD ≥8pp better,
CAGR give-up ≤2pp, Sharpe(m) not lower. Survivors get ONE OOS look
(2016+, frozen params): adopt iff ≥60% of the IS DD improvement holds and
the OOS CAGR/Sharpe conditions pass.

**Round-4 IS results (`bedrock_dd_screen4.json`):** the capitalized
convention UNLOCKS the trigger mechanism — first material DD movement of
the program — but no variant passed all three gates:

| K | book | CAGR | Sharpe(m) | maxDD | verdict |
|---|---|---|---|---|---|
| 50 | base | +9.50% | 0.76 | −41.8% | — |
| 50 | T1 q90&rising | +7.96% | 0.63 | −41.6% | reject (+0.2pp) |
| 50 | T2 q75&rising | +8.61% | 0.73 | −36.4% | reject (+5.4pp) |
| 50 | T3 equity −10% | +7.49% | 0.68 | **−31.4%** | reject (+10.4pp; CAGR −2.01pp, Sharpe) |
| 100 | base | +5.60% | 0.51 | −32.1% | — |
| 100 | T2 q75&rising | +5.65% | **0.70** | −27.6% | reject (+4.5pp < 8pp; CAGR/Sharpe PASS) |
| 100 | T3 equity −10% | +3.91% | 0.38 | −29.8% | reject |

Reading: T3 finds the DD (+10.4pp) but stays halted through the whole
recovery (NAV is still >10% under its high long after the trough), missing
the post-trough entries that §8f showed earn +8%/trade — hence the CAGR and
Sharpe misses. T2 at K=100 is a free +4.5pp (CAGR up, Sharpe 0.51→0.70).

**Addendum 4b (2026-08-12, BEFORE testing):** one mechanism-derived repair
of T3 plus its union with T2 — both defined from the already-confirmed
fall-phase mechanism, not from a parameter sweep:
- **T4 (equity slope):** halt new entries while NAV < 90% of trailing 1y
  high AND NAV < NAV 20 days ago (own equity underwater AND still
  deteriorating); resume as soon as the slope flattens, even if still
  underwater. All state lagged 1 day.
- **T2∪T4:** halt when either condition holds.
Same gates, same K ∈ {50, 100}; survivors get the ONE OOS look.

**Results 4b:** (to be filled after the run; committed before testing.)

## 9. Bibliography (primary sources)

**Replication/methodology:** Dickerson-Robotti-Rossetti 2026 (arXiv 2604.07880);
Dickerson-Mueller-Robotti JFE 2023; Dick-Nielsen-Feldhütter-Pedersen-Stolborg
2023 (SSRN 4586652); Dickerson-Nozawa "Credit Risk Primitives";
Ivashchenko-Kosowski FAJ 2024; Bai-Bali-Wen JFE 2019 (retracted 2023).
**Liquidity provision:** Anand-Jotikasthira-Venkataraman Mgmt Sci 2024;
Bao-O'Hara-Zhou JFE 2018; Bessembinder-Jacobsen-Maxwell-Venkataraman JF 2018;
Duffie JF 2010; Feldhütter RFS 2012; Ellul-Jotikasthira-Lundblad JFE 2011;
Cai-Han-Li-Li JFE 2019; Choi-Hoseinzade-Shin-Tehranian JFE 2020;
Ma-Xiao-Zeng RFS 2022; Haddad-Moreira-Muir RFS 2021; O'Hara-Zhou JFE 2021;
Kargar et al. RFS 2021; Coval-Stafford JFE 2007; Ambrose-Cai-Helwege 2008/12.
**Index mechanics:** Dick-Nielsen-Rossi RFS 2019; Ben Dor-Xu JFI 2011;
Chen-Lookman-Schürhoff-Seppi RAPS 2014; Hartley-Schwarz; Lou-Yan-Zhang RFS
2013; Todorov BIS 952; Pan-Zeng; Koont-Ma-Pastor-Zeng RFS 2025; Shim-Todorov.
**Munis:** Ang-Bhansali-Xing JF 2010 (NBER w14496); Landoni JFE 2018;
Green-Hollifield-Schürhoff RFS 2007; Harris-Piwowar JF 2006; Griffin-Hirschey-
Kruger JF 2023; Schwert JF 2017; Fleckenstein-Longstaff NBER w31389; Kalotay
(tax option); Wang SSRN 4072015; MSRB 2019/2023/2025 studies; LPL/BlackRock
seasonality; NY Fed/Brookings COVID-muni studies.
**Factors/ML:** Kelly-Palhares-Pruitt JF 2023; Israel-Palhares-Richardson JOIM
2018; Houweling-van Zundert FAJ 2017; Koijen-Moskowitz-Pedersen-Vrugt JFE 2018;
Jostova et al. RFS 2013; Gebhardt-Hvidkjaer-Swaminathan JFE 2005;
Bell-Kakhbod-Lettau-Nazemi NBER w33320; Feng-He-Wang-Wu JBF 2024;
Bali-Goyal-Huang-Jiang-Wen; Cao-Goyal-Xiao-Zhan Mgmt Sci 2023; Robeco "True
Value" (SSRN 4718484); Ben Dor et al. "Systematic Investing in Credit" (Wiley
2021). **Primary market:** Cai-Helwege-Warga RFS 2007; Helwege-Wang JFI 2021;
Bessembinder et al. JFE 2022; Rischen-Theissen JFI 2020; NN IP HY NIP 2022.
**Practitioner live:** Robeco "Ten Years" 2022 + 2024 peer study; BlackRock
HYDB/IGEB records; VanEck/Insight fallen-angel index records; AQR (Style
Investing in FI; Credit Risk Premium; Illiquidity Myth; Under the Hood);
PIMCO "Bonds Are Different"; Man Group systematic credit; MSRB odd-lot/SMA
research; MarketAxess/ICE market-structure notes.

*(Full URLs preserved in the research agents' reports; key claims
spot-verified against primary abstracts during the sweep.)*
