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

## 8. Bibliography (primary sources)

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
