# PHOENIX — Independent Strategy Review

**Date:** 2026-07-02
**Scope:** `alt/phoenix_production.py` and everything it consumes (five sleeve generators, price/macro data layer, refresh/live/paper pipeline, factsheet/webapp claims).
**Method:** line-by-line code review of every file in the production chain, plus empirical verification against the shipped CSVs (all numbers below were recomputed from `data/results/*.csv` and `data/etfs/*.csv`; reproduction snippets in the Appendix).
**Rule of engagement:** review only — **no strategy code was modified**. Findings and an improvement plan are recorded here.

---

## 1. Executive summary

PHOENIX is a competently engineered system with genuinely good hygiene in many places (per-sleeve signal lagging is clean, blend weights verifiably match the stated IS inverse-vol fit, history freezing protects the record, block-bootstrap and survivorship falsification tests exist). **But the headline numbers — Sharpe 2.34, CAGR 36.8%, "5 uncorrelated sleeves, max pairwise |ρ| = 0.19" — are materially overstated by three structural defects:**

1. **[CRITICAL] Cross-sleeve return-dating misalignment.** The five sleeves book "day-t" returns over four different physical windows (VANGUARD/CRYPTO: open[t−1]→open[t]; QUANTUM: close[t−1]→close[t]; ORION: open[t]→open[t+1]; HELIOS: open[t+1]→open[t+2]). Summing them by date label manufactures the near-zero correlation matrix, artificially smooths blend volatility (~17% labeled vs ~22.4% realigned, i.e. **32% understated**), and injects genuine lookahead into the overlay path. Realigned to a common window, VAN/ORI/HEL pairwise correlations are **0.43–0.65**, not ≈0, and PHOENIX's Sharpe drops **2.34 → 1.81**.
2. **[CRITICAL] QUANTUM's 2010–2018 series is in-sample model output.** The XGBoost model is trained on all of 2010–2018 and then "backtested" over those same years (IS Sharpe 2.73 vs honest OOS 0.87). PHOENIX's IS-fit blend weights and its IS/FULL headline metrics inherit this contamination.
3. **[CRITICAL] The crypto sleeve's economics are Grayscale premium dynamics, not BTC.** Its biggest booked days are GBTC/ETHE NAV-premium explosions (+68%, +57%, +39% single days) at 10 bps assumed cost on OTC trusts that traded at 50–300 bps spreads — and the live book trades IBIT, which structurally cannot earn that stream. The backtest CSV still holds GBTC after Jan-2024 while live trades IBIT.

On top of these, the **live data layer has active bugs**: since ~2026-04-06 the incremental price fetcher appends **unadjusted** rows onto a dividend-adjusted history (`auto_adjust=False` in `live_signal.py:91`), so every dividend after the seam is booked as a price loss — the bond-heavy sleeves (TMF/UBT/TYD, ~4%+/yr distributors) and the BIL cash leg (~4.5%/yr) are silently bleeding fictitious losses into live returns, and the history-freeze mechanism locks the corruption in permanently. Worse, a **second workflow** (`daily-update.yml`) rewrites SPY/TLT/IEF with fully-adjusted history every day while PHOENIX appends raw tails to the same files, and the fetcher has **no split handling** — a split on any of the ~30 LETFs (NUGT, UCO, YINN, TQQQ have all split historically) would freeze a fake ±N00% one-day return into the record with nothing to detect it.

**Corrected performance estimate** (recomputed, see §6): realigned and with QUANTUM's contaminated IS segment excluded, PHOENIX is roughly a **Sharpe 1.45–1.85, CAGR 22–30%, ~15% vol** strategy. The most defensible single number — realigned frame, 2019–2026 only, where every sleeve is genuinely out-of-sample — is **Sharpe ≈ 1.73, CAGR ≈ 29%, MDD ≈ −14%**. That is still a strong result, but it is before haircuts for (a) strategy-selection bias — PHOENIX is the winner of a ≥25-family in-sample tournament conducted in May–June 2026 over a fully realized 2010–2026 history (repo's first commit: 2026-05-22; only ~8 trading days are live), and (b) optimistic transaction costs on thin leveraged ETFs. A realistic live expectation is materially below the corrected backtest, and far below the advertised numbers.

**None of the defects look intentional** — the misalignment is documented in `refresh_all.py` as a cron-freshness concern (its correlation/overlay consequences were simply not connected), and the author's own robustness work (bootstrap, survivorship test, honest design docs that admit failing their Sharpe targets) is above average. The system is worth fixing rather than discarding; a prioritized plan is in §7.

---

## 2. What PHOENIX actually is

`alt/phoenix_production.py` — fixed-weight blend of five daily sleeve return series, then three risk overlays and a small TC charge:

- **Weights** (verified = inverse-vol on 2010-2018 IS to 3 decimals): VANGUARD 0.236, ORION 0.327, HELIOS 0.185, QUANTUM 0.152, CRYPTO 0.101.
- **Overlays**, all computed from trailing data and `shift(1)`-ed before application: 15% vol target (60d realized vol, mult clipped [0.25, 1.0]), drawdown throttle (linear de-risk vs 252d HWM of the *vol-targeted hypothetical NAV*, floor at −10% DD), vol-regime gate (halve exposure when 60d vol > its trailing-252d 99th percentile).
- **TC:** 10 bps × |Δ total multiplier| daily (overlay turnover only; sleeve-internal costs live inside the sleeve CSVs).

The sleeves:

| Sleeve | File | Strategy | Booking convention (day t) |
|---|---|---|---|
| VANGUARD | `vanguard_strategy.py` | Monthly rotation over {QLD, UGL, TMF, TYD}: 189d mom > 0 AND close > 200d SMA, inverse-60d-vol weights, daily 4-trigger macro-gate participation (VIX/HY-OAS/T10Y2Y/SPY-trend), constant 1.5× gross. 5 bps. SR 0.96 | open[t−1]→open[t] |
| ORION | `orion_strategy.py` | Weekly (Wed) 50/50 two-book: top-4 of 12 risk LETFs by 252d momentum above 200d MA, gated daily by VIX<30 & HY OAS<7; top-2 of 4 safe LETFs (TMF/UBT/TYD/UGL) by momentum/low-vol z-blend. 5 bps. SR 0.85 | open[t]→open[t+1] |
| HELIOS | `helios_strategy.py` | Weekly (Fri) cross-sectional momentum (close[t−42]/close[t−189]−1) on 13 **unlevered** underlyings, expressed 1-for-1 via matched 2x/3x LETFs, top-2 at 50/50, macro gate with defensive bypass, residual BIL. 5 bps. SR ~0.76 | open[t+1]→open[t+2] |
| QUANTUM | `quantum_strategy.py` | XGBoost (400 trees, depth 4) predicting fwd-21d returns of 17 LETFs from ~28 lagged price/macro features; top-3 equal-weight, rebalanced every 21 trading days; model fit once on 2010-2018, frozen. 10 bps. IS SR 2.73 / OOS 0.87 | close[t−1]→close[t] |
| CRYPTO | `phoenix_v2_crypto.py` | Weekly (Fri) 63d TSMOM on GBTC + ETHE (Grayscale trusts), macro-gated, cash in BIL; live proxy switches GBTC→IBIT from 2024-01-11. 10 bps. SR 0.84, MDD −71% | open[t−1]→open[t] |

Daily ops: GitHub Actions cron (23:30 UTC weekdays) → `refresh_all.py`: fetch prices/FRED → extend sleeve CSVs append-only with a "freeze history" merge (trims 1/2 trailing rows for ORION/HELIOS whose bookings need future opens) → re-run production → factsheet/audit/live-signal/paper-trader → inject JSON into `docs/phoenix.html` → validate vs frozen IS expectations.

---

## 3. Critical findings

### C-1. Cross-sleeve dating misalignment fabricates the "orthogonality" and leaks into the overlays

**Evidence (code):**
- `vanguard_strategy.py:196-199` — `o2o = opens/opens.shift(1) − 1`; day-t booking = open[t−1]→open[t].
- `orion_strategy.py:241` — `o2o = opens.pct_change().shift(-1)`; day-t booking = open[t]→open[t+1].
- `helios_strategy.py:219` — `r_fwd = opens.shift(-2)/opens.shift(-1) − 1`; day-t booking = open[t+1]→open[t+2].
- `quantum_strategy.py:385,422` — close-to-close; `phoenix_v2_crypto.py:110,116` — open[t−1]→open[t] (via `shift(-1)` then final `shift(1)`).
- `phoenix_production.py:53` concatenates all five **by date label** and computes correlations, blend vol, overlays and NAV on the result.

**Evidence (empirical, recomputed from the shipped CSVs):**
- Lagged correlations expose the offsets: corr(HEL[t+1], ORI[t]) = **0.648**; corr(ORI[t], VAN[t+1]) = **0.563**; corr(HEL[t+2], VAN[t]) = **0.426**; the same pairs at lag 0 are −0.02…−0.06 — which is what the factsheet reports.
- Realigned to the common window W_t = open[t]→open[t+1] (VAN shifted −1, HEL shifted +1):

| | VAN | ORI | HEL | QUA | CRY |
|---|---|---|---|---|---|
| VAN | 1.00 | **0.56** | **0.43** | 0.12 | 0.02 |
| ORI | | 1.00 | **0.65** | 0.19 | 0.00 |
| HEL | | | 1.00 | 0.11 | 0.02 |
| QUA | | | | 1.00 | 0.01 |

- Raw blend vol: 17.0% as-labeled vs **22.4%** realigned (32% understated). Running the identical production overlay code on the realigned frame: **Sharpe 2.34 → 1.81, CAGR 36.8% → 28.9%** (IS 1.88 / OOS 1.73, gap 0.16).

**Consequences:**
1. The factsheet/webapp claims — "5 uncorrelated LETF strategies", "max pairwise |ρ| = 0.19", "approaches the √N diversification bound" — are artifacts of date labels. Three of the five sleeves are substantially the same long-LETF trend bet (they share universes, 200d-MA filters and macro gates; that they *should* correlate is obvious ex ante).
2. **Genuine lookahead in the overlay path.** The overlay multiplier for day t is built from `raw[…t−1]` (`phoenix_production.py:100-114`), but raw[t−1] contains HELIOS returns realized through open[t+1] and ORION returns realized through open[t]. That multiplier is applied to VANGUARD/CRYPTO day-t returns whose exposure window (open[t−1]→open[t]) *ended before the information existed*. The DD-throttle and 99th-pct vol gate therefore react up to ~2 days ahead of real time in crashes — exactly the episodes the overlays are credited for (COVID Q1-2020, Q3-2022 are cited in the webapp).
3. The blended NAV is not the NAV of any executable portfolio (it compounds sums of returns earned over different windows), and `live_signal.py`'s overlay computation inherits placeholder trailing rows (see C-4/pipeline), so live ≠ backtest on precisely the volatile days.

**Fix direction:** adopt one booking convention (realization-dated open→open is the natural one), re-date ORION (+1) and HELIOS (+2), re-fit blend weights and re-run overlays on the realigned frame, and republish all metrics. This is the single highest-impact correction and touches no strategy logic.

### C-2. QUANTUM's 2010–2018 history is in-sample model output

`quantum_strategy.py:517,550,566-573`: the final XGBoost model is trained on **all of IS (2010-03-11…2018-12-31)**, then predictions are generated over the full window **including the same IS rows**, and the resulting backtest is written to `quantum_returns.csv`. IS Sharpe **2.73** vs OOS Sharpe **0.87** — a 1.86 gap that is memorization, not edge (the design cache `quantum_model.pkl` confirms fit-once-frozen).

Downstream: PHOENIX's blend weights are inverse-vol **fit on this contaminated IS**; QUANTUM's diluted in-sample vol earns it a 0.152 weight; the advertised IS Sharpe 2.56 and FULL 2.34 are propped up by 8.8 years of in-sample ML output. The only honest QUANTUM numbers are 2019+ (which are genuinely out-of-sample — the feature lagging, CV embargo, and freeze discipline are clean; the *published series* is the problem).

Secondary bug: the "(N, K) selected via CV" claim is half-false — K is never used inside the CV loop (`quantum_strategy.py:241-268`; `quantum_metrics.json` `all_scores` shows identical ICs for K=3/4/5 at fixed N), so K=3 wins by iteration order and is an unvalidated free parameter.

**Fix direction:** walk-forward (e.g., annual expanding-window refits with an embargo) for the historical series, or exclude/flag pre-2019 QUANTUM everywhere (blend fitting, headline metrics, factsheet).

### C-3. CRYPTO sleeve: premium-dynamics backtest, different instrument live

- The sleeve's largest booked days are Grayscale **premium** events, not BTC moves: GBTC open→open **+68.2%** (2017-12-26), **+56.9%** (2017-05-25), **+39.0%** (2017-12-22); ETHE's first weeks include a +400% print on zero-volume frozen marks that then feed `mom63` for 63 sessions. The 2023–24 leg includes the one-time discount-closing rally into ETF conversion.
- `main()`'s universe is GBTC/ETHE **only** — the production CSV holds GBTC even after Jan-2024, while `live_signal.py:238` trades **IBIT** (`use_live_proxy=True`). The live book and the booked backtest are different instruments with different expense ratios (1.5% vs 0.25%) and no premium dynamics.
- 10 bps assumed cost vs 50–300 bps historical OTCQX spreads; survivorship: {GBTC, ETHE} are precisely the two surviving winners (ETCG −95%, LTCN, BCHG, GDLC absent); the 0.101 weight rests on IS vol diluted by 5.5 years of sitting in BIL.

**Fix direction:** rebuild the historical series from spot BTC/ETH (or NAV-based) total returns with realistic trust spreads if trusts must be used; switch the booked series to IBIT/ETHA from availability; re-fit the weight; disclose the premium-era caveat on all pre-2019 crypto contribution.

---

## 4. High-severity findings

### H-1. Live data corruption: adjusted/unadjusted price seam since 2026-04-06 (active bug)

`live_signal.py:91` appends `yf.download(..., auto_adjust=False)` rows (raw prices, ≤14-day refetch window) onto CSVs whose history is `auto_adjust=True` total-return-adjusted. Verified: `Adj Close` is empty through 2026-04-03 and equals `Close` from 2026-04-06 in every file checked (TMF, TYD, QLD, BIL…); BIL shows zero ex-div drop days per year 2023–2025 but three <−0.15% days in 2026, and BIL 2026-YTD computes to 0.88% vs ~2%+ expected. Consequences: every dividend after the seam books as a fictitious price loss (TMF/UBT/TYD ~4%+/yr — ORION's safe book is 50% of the sleeve; VANGUARD holds TMF/TYD/UGL; every sleeve's BIL cash leg bleeds ~25 bps/month held), momentum/trend/vol signals on bond LETFs are computed on a distorted series, any split older than the refetch window would splice levels, and the freeze-history mechanism makes the corruption **permanent**. This is the most actionable pure bug in the system.

### H-2. The entire record is retrospective; selection multiplicity is unaccounted

Repo's first commit is 2026-05-22; the sleeve CSVs were first committed 2026-06-20; only ~8 trading days are live. The "OOS 2019–2026" was fully realized history when every design decision was made. The repo contains **≥25 sibling strategy families** (aurora, bastion, citadel, kraken, meridian, nebula, neutrino, nova, polaris, pulsar, revenant, solar, trident, vortex, zephyr, shannon, proprietary v1–v8, plus 7 PHOENIX variants) — PHOENIX is the tournament winner, and the five sleeves are the 5 survivors of ~60 strategy scripts. The IS/OOS split measures parameter stability, not selection bias; the bootstrap CI (§5 "clean" list) conditions on the observed series and cannot see this either. Expected live Sharpe is well below any backtest figure. (VANGUARD's design doc even admits the universe was narrowed because wider baskets "produced lower Sharpe" — full-sample selection; its `LEV_UNIVERSE` at `vanguard_strategy.py:52-57` is dead code that makes screening look broad, and `gross=1.5` was calibrated to hit a 20% CAGR target on the full sample.)

### H-3. Overlay parameters are the argmax of a 225-point grid with OOS visible

`data/results/phoenix_v2_grid.csv`: the production config (dd_win=252, dd_floor=−0.10, vol_win=60, vol_pct=0.99) ranks **1/225 by full-sample Sharpe** (and 1/225 by IS Sharpe), and the grid CSV contains an `oos_sr` column — OOS was not blind during selection. Strong mitigant: the grid's full-sample Sharpe range is only 1.99–2.10, so overlay tuning contributes little; but the "fit on IS only" framing is not supportable from the artifacts.

### H-4. Stale, inflated documentation in the canonical file

`phoenix_production.py:10-16` advertises Sharpe 2.37, **CAGR 57.4%**, vol target **20%**, cap **2.0×**, IS/OOS gap **0.28 "tight"**. The code and current output say target 15% (line 36), cap 1.0 (line 37), Sharpe 2.34, CAGR 36.8%, gap **0.43**. The docstring reflects an earlier, more aggressive parameterization — direct residue of iterating parameters against full-sample results, and misleading to any reader of "the single reference implementation".

---

## 5. Medium and low findings (by layer)

### Portfolio layer (`phoenix_production.py`, `refresh_all.py`, factsheet)

| Sev | Finding |
|---|---|
| M | **Sleeve NaN → 0.0 masking** (`phoenix_production.py:54` `fillna(0.0)`): any sleeve gap silently becomes a flat day. ORION/HELIOS trailing rows are genuine 0.0 placeholders published daily in NAV/factsheet (repaired next cron via the lookahead trim, `refresh_all.py:406-479`). The trim itself is gap-robust across cron outages (incomplete rows are always the trailing N rows, since the file only extends on successful runs) — but a mid-tail row computed from a stale-ffilled single ticker is *not* among the trailing N and freezes wrong forever (see the yfinance single-point-of-failure row in the pipeline table). No validation checks for frozen zeros or stale-fill returns. |
| M | **Calendar contamination**: `vanguard_returns.csv` contains 153 non-NYSE dates (incl. 11 Jan-1 rows) from `pd.bdate_range` + `ffill(limit=2)` (`vanguard_strategy.py:90-92`); the union index pads the other four sleeves with 0.0 on those dates. Direction is mildly conservative (NYSE-only calendar: Sharpe 2.39 vs 2.34) but VANGUARD's January rebalance regularly "executes" at a stale Dec-31 open on a holiday. |
| M | **Sharpe excludes the risk-free rate** (`metrics()`, `phoenix_production.py:61`): with BIL as rf, excess-return Sharpe is 2.25 vs reported 2.34; sleeves that sit in BIL book cash carry (~5% in 2023–24) as alpha. SPY benchmark treated identically (consistent), but headline overstates vs standard convention. |
| M | **DD throttle semantics**: dd is computed on the *vol-targeted hypothetical* NAV (`phoenix_production.py:105-109`), not the strategy's own NAV — actual MDD is −17.6% despite the "−10% floor" language; webapp phrasing invites misreading. |
| L | TC model covers only overlay turnover (|Δmult| × 10 bps); 10 bps/unit-notional is thin for LETF baskets in stress, exactly when the multiplier moves. |
| L | `metrics()` annualizes with n/252 on the padded 4255-row calendar (see calendar item). |
| L | Factsheet duplicates blend weights/params as literals in 4+ places (`refresh_all.py:182,253,347`, `live_signal.py:58`) — drift risk. |

### VANGUARD / ORION

| Sev | Finding |
|---|---|
| M | ORION: 256 dead warm-up zero days inside the scored window (prices sliced to START_DATE before the 252d momentum warm-up; first nonzero return 2011-03-16) — dilutes IS Sharpe and pads the IS/OOS-gap statistic; sleeves' IS windows aren't comparable (VANGUARD warms up on pre-2010 history). |
| M | HY-OAS publication lag: both sleeves trade at open[t] using `BAMLH0A0HYM2[t−1]`, which often posts to FRED around/after the 9:30 open — hours-scale availability risk (VIX leg fine). FRED series are latest-vintage, not point-in-time; OAS revisions can flip gates near thresholds. |
| M | Flat 5 bps one-way is optimistic for TYD (ADV often <50k shares), UGL, DRN, EDC, YINN, UCO (10–40 bps spreads); ORION's daily macro gate on top of the weekly freeze can liquidate/re-enter half the book on consecutive days. Realistic costs ≈ 1–3%/yr off VANGUARD. |
| M | VANGUARD runs 1.5× gross with zero financing cost (~1–2.5%/yr missing 2022–26 standalone; at the blend's 0.236 weight aggregate gross stays ≤1, so the *blend* may be self-funding). |
| L | VANGUARD trigger NaN asymmetry: `~(spy > 200dma)` fires risk-off on NaN while the other three triggers default risk-on (`vanguard_strategy.py:131`) — unintentional, conservative direction. |
| L | Silent NaN swallowing in P&L sums both sleeves; ORION last row fabricated 0.0 (`orion_strategy.py:245`); minor tie-break/dedup/cost-dating nits. |

### HELIOS / QUANTUM

| Sev | Finding |
|---|---|
| M | HELIOS implicitly re-trues to 50/50 **daily at zero cost** (constant ffilled target weights × daily returns, `helios_strategy.py:194-200,232`) while charging TC only on weekly target changes; on 3× LETFs drift trades are material and uncounted. |
| M | QUANTUM drops the overnight close[d−1]→open[d] return of outgoing holdings on every rebalance (~12 nights/yr vanish) and implicitly exits at the same close whose data feeds the day-d signal; turnover/TC computed on 21-day-stale target weights, not drifted weights — costs understated. |
| M | HELIOS params IS-tuned with residue: docstring says VIX z < 0.75, code has 1.5 ("softer gate chosen via IS", `helios_strategy.py:84`); "6-month momentum" vs 189d comment. ~8 free params. Mitigant: OOS (0.88) > IS (0.68). |
| M | HELIOS ships 0.0-minus-cost placeholders in its last two rows every day (skipna sum, `helios_strategy.py:232`); production publishes them (see portfolio layer). |
| L | HELIOS skips the whole week when Friday is a holiday (no Thursday fallback); opens ffilled up to 3 days = stale fills on illiquid names; QUANTUM loader lacks date dedup; QUANTUM early-window dead zone (sleeve at exactly 0% until enough names have 252d history); `live_extend` fabricates next-bday row by copying the last row (live Friday picks can use Thursday closes, diverging from the backtest convention). |

### CRYPTO / synthetic data

| Sev | Finding |
|---|---|
| M | Blend weight 0.101 rests on IS vol diluted by 5.5 years of BIL (measured 63% ann vs 90–100% active-period), on premium-era GBTC vol ≈ 2× IBIT vol. |
| M | Early ETHE zero-volume frozen prints (+400% open 2019-06-20) feed `mom63` for 63 sessions — first ETHE trades driven by non-tradable marks. |
| M | Synthetic LETF builder (`synthetic_letf_build.py`) — **quarantined from production** (only `phoenix_extended.py` stress test consumes `data/etfs_extended/`) but flawed: no swap-financing spread (3× funds ~0.5–1.5%/yr too cheap), FEDFUNDS monthly average applied from month-start (mild lookahead), verified +7.04% phantom close-to-close jump at UPRO's 2009-06-25 splice, ERX built at 2× though it was 3× until Mar-2020; correlation "validation" is scale-invariant and can't catch these. |
| L | Docstring omits two of the four gate terms; costless daily re-truing between Fridays; regime logic duplicated between `main()` and `build_weights()` (backtest/live gate can silently desync); `live_extend` +1 BDay can land on a holiday; synthetic BIL over-accrues ~14 bps/yr (offsets its −15 bps drag). |

### Live/paper/validation pipeline

| Sev | Finding |
|---|---|
| **C** | **Two racing workflows write the same price CSVs on incompatible bases.** `daily-update.yml` (23:00 UTC) rewrites SPY and ~28 bond ETFs (incl. TLT/IEF — HELIOS signal underlyings) with **fully-adjusted** history every day (`scripts/download_bond_etfs.py:82,109-112`, yfinance default `auto_adjust=True`); `phoenix-signal.yml` (23:30 UTC) then appends **raw** tails via `live_signal.py:91-107`. Confirmed: TLT.csv has adjusted 2005 levels with a raw 2026 tail; commit `225c45b` shows the entire IEF.csv churning (10k+ line diffs) inside a PHOENIX refresh. Every ex-div date shifts the adjusted history under the raw tail, injecting spurious returns into HELIOS's TLT/IEF trend signals and the SPY 200-dma gates of VANGUARD/CRYPTO — and the affected sleeve returns are then frozen forever. The push step retries **without fetch/rebase** (`phoenix-signal.yml:46-52`), so when the two bots race, an entire day's PHOENIX state can be dropped and later reconstructed by backfill (with the lookahead defects below). |
| **C** | **No split handling in the append-only fetch** (`live_signal.py:82-110`: refetch window ≤14 days, `drop_duplicates(keep="last")`). Yahoo rescales full history on a split; the local file keeps the old scale and appends new-scale rows → fake ±N00% one-day return, frozen permanently by the merge. LETFs in this universe split routinely (NUGT, UCO, YINN, DRN, TQQQ historically). `validate_state.py`'s wide metric ranges (Sharpe 2.0–2.7 etc.) would not reliably catch a single corrupted day. Highest-probability catastrophic-corruption path in the system. |
| H | **HELIOS live timing drifts from backtest; backfilled rows contain genuine lookahead.** HELIOS is the only sleeve whose signals are *not* internally lagged (`build_target_weights` uses `mom.loc[dt]` same-day, `helios_strategy.py:183-192`), so `live_signal`'s uniform "row at as_of+1 BDay" convention (`live_signal.py:179-191`) makes live rebalance HELIOS at Friday's open on Thursday's close (backtest: Friday close signal, Monday open fill), then re-trade Monday — unmodeled turnover. In backfill mode (`backfill_trades.py` → `--as-of Thursday`), the Friday row is computed from the **real Friday close** — information after the as-of date; the trade log is not point-in-time. The other four sleeves are clean here (internal `shift(1)`). |
| H | **Live overlay computed on structurally incomplete trailing rows.** `compute_overlay_mult` (`live_signal.py:274-299`) reads `raw_ret` whose last ORION row and last two HELIOS rows are placeholder zeros → 60d vol biased down, exposure biased up; when the rows are repaired 1–2 crons later, the recorded backtest multiplier differs from what was traded (verified: 2026-07-01 backtest `total_mult=0.611` vs live `overlay_mult=0.623` the same evening). A permanent, unmeasured live-vs-backtest wedge concentrated on volatile days. |
| H | **Backfill replay is not point-in-time**: replays against today's CSVs (data revisions and daily adjusted rewrites leak in), overlay uses repaired rows a real run never saw, FRED values dated d were published d+1, and `_macro_snapshot()` ignores `as_of` entirely (`live_signal.py:370-402`) — backfilled rows carry today's VIX/HY/200-dma stamped onto past dates. |
| M | **FRED T+1 lag / HELIOS gate has no lag at all.** VIXCLS/BAMLH0A0HYM2 end one day behind prices at cron time; sleeve gates ffill silently to t−1 live, while HELIOS's `build_macro_gate` (`helios_strategy.py:149-161`) uses **same-day** VIX/HY closes in the backtest — values a 23:30 UTC run can never have. ORION/HELIOS frozen rows get recomputed next cron with true same-day values while the executed trade used stale ones → recorded history diverges from executed positions on every gate-flip day. |
| M | **"No margin" can be violated latently**: raw risk gross tops out at ~1.118 (VANGUARD is 1.5× gross at 0.236 weight); `bil_weight = max(0, 1−gross)` floors at 0 and targets can sum to ~112% if the vol multiplier hits its 1.0 cap in a calm regime (`live_signal.py:326-339`). Currently benign (mult 0.62, gross 95.6%). |
| M | **yfinance is a silent single point of failure**: per-ticker failures swallowed (`live_signal.py:92-97`); a lagging ticker means HELIOS freezes a wrong non-zero return computed from a 3-day-stale ffilled open (mid-tail — the lookahead trim cannot reclaim it), while ORION silently drops the day (`opens.dropna(how="any")`). Freshness is only checked via SPY/QQQ/IBIT. |
| M | **`validate_state.py` would not catch its target failures and is not a gate**: no frozen-IS hash check, no trailing-zero detection, no seam/split detection, ranges wide enough that a ≤33%-weight sleeve going flat for weeks passes; `refresh_all.run()` ignores its exit code and the workflow commits regardless — "drift detected" ships to the public dashboard anyway. |
| M | **`apex/sleeves_phoenix_exact.py` is a divergent "EXACT clone"** (ORION top-3 vs production K=4; different macro gate — HYG/LQD ratio vs HY-OAS triggers; VANGUARD clone omits 1.5× gross; CRYPTO clone trades BITO/TQQQ) and `apex/test_phoenix_exact.py` contains **no assertions** — a research script named like a test. Nothing in the live path imports apex (contained), but it cannot pin behavior and has already drifted. |
| M | **Paper-trading fidelity**: sub-0.5% fills skipped in the fills log yet NAV assumes full targets; QUANTUM implicitly re-trued daily live vs drifting in backtest; paper NAV/fills rebuilt from scratch each run from a **mutable** positions log — the paper track record can silently rewrite itself (not an audit log). Reported `ann_tracking_error_bps = 2829.8` is dominated by the C-1 date misalignment, defeating its purpose as a drift alarm. |
| L | Cron timing itself is safe (23:30 UTC is after the close year-round; no date off-by-one). `T10Y3M` read but never refreshed (latent staleness trap); new-ticker bootstrap gets only 14 days of history (signals NaN for a year); live overlay scales only the non-BIL side while the backtest scales whole sleeve returns (small definitional wedge); two ITD numbers in `compute_performance.py` disagree slightly (first-fill TC). |

---

## 6. Quantified corrections (recomputed)

All rows run the **identical production overlay code** (`run_strategy`) — only the input frame changes:

| Variant | Sharpe | CAGR | Vol | MDD | IS SR | OOS SR |
|---|---|---|---|---|---|---|
| As shipped (labels as-is) | 2.34 | 36.8% | 13.8% | −17.6% | 2.56 | 2.12 |
| Realigned to common window | 1.84 | 29.9% | 14.8% | −13.8% | 1.94 | 1.73 |
| Realigned + QUANTUM only from 2019 | 1.45 | 22.1% | 14.5% | −15.1% | 1.19 | 1.73 |
| **Most defensible: realigned, 2019–2026 only** | **1.73** | **28.9%** | 15.4% | −13.7% | — | — |

Additional sensitivities: NYSE-calendar-only +0.05 SR; excess-return Sharpe −0.09; both minor. Not quantified here (all negative): realistic spreads on thin LETFs/OTC trusts, financing on VANGUARD's 1.5× gross, HELIOS drift-trade costs, crypto premium non-replicability, selection-bias haircut.

---

## 7. Improvement plan (prioritized; no code changed in this review)

**P0 — data integrity (do before trusting any live number):**
1. Standardize on **one price basis and one writer**: give PHOENIX its own price directory (or stop `daily-update.yml` rewriting shared tickers), fetch with `auto_adjust=True` everywhere, and re-download full adjusted history once to repair the post-2026-04 rows *before* more history freezes. Fix the push-without-rebase race in `phoenix-signal.yml`.
2. Add **split/seam detection**: a validator that recomputes the last ~30 frozen sleeve returns from prices, alarms on any single-day |return| beyond a sanity bound, on adjusted/raw seams, and on rows whose implied dividend days don't match the distribution calendar. The append-only fetcher must detect splits (compare overlapping rows on refetch).
3. Make `validate_state.py` **gate the commit** (`refresh_all.run()` currently ignores its exit code) and add: frozen-IS hash check, per-ticker freshness check (not just SPY/QQQ/IBIT), trailing-zero detection, and live-vs-recorded weight reconciliation. Make the paper positions/fills logs append-only (immutable audit trail).

**P1 — measurement honesty (changes numbers, not strategy):**
4. Unify booking to realization-dated open→open across all five sleeves (re-date ORION +1, HELIOS +2 at save time); re-fit inverse-vol weights on the realigned IS; re-run overlays; republish factsheet/webapp with corrected correlations and Sharpe (≈1.7–1.8, not 2.34). Remove "max |ρ| = 0.19" and √N claims. This also makes the paper tracking-error metric interpretable (currently 2,830 bps/yr of mostly-artifact).
5. Replace QUANTUM's published history with walk-forward output (annual expanding refits, N-day embargo), or exclude pre-2019 QUANTUM from all fitting and headline windows. Fix the K-never-used CV bug.
6. Rebuild crypto history on spot/NAV total returns with realistic trust spreads, or clearly split "premium-era proxy (not replicable)" from "IBIT-era (replicable)"; align the booked instrument with the live instrument from 2024-01-11.
7. Lag HELIOS's signals by one day (its signals and macro gate are the only unlagged ones) — removes the Friday-early live execution and the backfill lookahead in one change.
8. Subtract BIL from Sharpe numerators; fix the stale docstring; report the strategy's own MDD next to the throttle floor.

**P2 — robustness upgrades:**
9. Realistic cost model: per-ticker spread table (LETF tier ~10–20 bps, thin names 30–50 bps, OTC trusts 100–300 bps historical), turnover measured on drifted weights; charge HELIOS's daily re-truing or switch it to buy-and-hold-between-rebalances accounting.
10. Lag FRED HY-OAS by one extra day everywhere (publication lag); consider a point-in-time macro store (`data/pit/` exists but only holds equity membership). Make `_macro_snapshot()` respect `as_of`.
11. Deflate for multiplicity: report a selection-adjusted expectation (e.g., White's Reality Check / deflated Sharpe across the ~60 strategy scripts and 225-point overlay grid), and treat an untouched, append-only paper ledger as the only "live" evidence.
12. De-duplicate hardcoded weights/params (single source of truth consumed by production, refresh, live, factsheet); either make `apex/sleeves_phoenix_exact.py` a real pinned test with assertions or rename it so "EXACT" doesn't mislead.
13. VANGUARD: drop the fake `bdate_range` calendar (use the SPY calendar); fix the NaN-asymmetric SPY trigger; account financing on gross > 1; cap portfolio target weights at 100% (they can currently reach ~112%).
14. Reduce three near-duplicate trend sleeves (VAN/ORI/HEL realigned ρ 0.43–0.65) to their genuinely diversifying core, or acknowledge the concentration in sizing — after realignment the ensemble is effectively ~3 independent bets, not 5.

---

## 8. What is affirmatively clean (verified, no issue)

- **Per-sleeve signal timing**: every sleeve's own signal→execution chain is leak-free (all signals lagged ≥1 bar before fills; no `shift(-1)` in any signal path; QUANTUM's CV embargo correct; CRYPTO's `lookahead_days=0` claim verified correct).
- **Blend weights = IS inverse-vol** exactly as claimed (recomputed to 3 decimals).
- **Overlay layer timing** (given aligned inputs): all three multipliers are `shift(1)`-ed before application; vol-gate quantile uses trailing windows only.
- **No synthetic pre-inception data in production**: all five sleeves read `data/etfs/` only; every LETF CSV starts at real inception; `data/etfs_extended/` feeds only the labeled stress test.
- **Pre-2026-04 price history** is consistently total-return adjusted across Open/Close (adjusted-open fills are the correct convention).
- **History freezing works as designed** so far: pre-2025 sleeve rows are byte-identical across all commits that touch the CSVs.
- **Honest self-testing exists**: 21-day circular block bootstrap (`robust_bootstrap.py`), survivorship falsification with 15 collapsed LETFs (`robust_survivorship.py`), and design docs that openly report failing their own Sharpe targets.
- No shorting, no margin at the portfolio wrapper (cap 1.0 verified in code, with the ~112% latent edge case noted above), no bare `except` swallowing in the sleeve files, duplicate-date guards in most loaders.
- **Cron timing is safe**: 23:30 UTC is after the 4pm ET close year-round, and the UTC date equals the ET trading date at that hour — no off-by-one.
- **Live overlay math is an exact one-step-ahead replica of the backtest formula** (`compute_overlay_mult` matches `run_strategy` term-for-term; parameters re-verified by `validate_state.py`) — the wedge comes from incomplete input rows, not the math.
- **`live_extend` semantics are correct for VANGUARD/ORION/QUANTUM/CRYPTO** (internal `shift(1)` means the appended row yields the weight for tomorrow's open from tonight's real close); HELIOS is the sole exception.
- **The freeze-and-append mechanism is gap-robust** across cron outages and holidays (incomplete rows are always the trailing N of the file, since the file only extends on successful runs; `lookahead_days` = 0/1/2/0/0 verified correct per sleeve; the merge has no date overlap).
- **Paper trader has no fill peeking**: signals dated close-t are filled at the first open strictly after t; open-to-open returns on actually-held weights; two-sided TC accounted.
- **GBTC→IBIT live substitution** is a clean weight-level mask from IBIT's listing date and honestly disclosed in the code.

---

## Appendix — key reproduction numbers

- Lag correlations (shipped CSVs, 2010-03-11→2026-07-01): corr(HEL.shift(1), ORI)=0.648; corr(ORI, VAN.shift(-1))=0.563; corr(HEL.shift(2), VAN)=0.426; all lag-0 pairs |ρ|≤0.185.
- Raw blend vol 16.98% (labels) vs 22.44% (realigned, W_t=open[t]→open[t+1]: VAN−1, HEL+1).
- `run_strategy` on realigned frame: SR 1.84 / 1.81 (union / NYSE calendar), CAGR 29.9%, MDD −13.8%.
- QUANTUM IS/OOS Sharpe 2.73 / 0.87 (`quantum_metrics.json`, recomputed from CSV).
- Grid: chosen overlay config rank 1/225 by full_sr and is_sr; grid full_sr range 1.9887–2.1031 (`phoenix_v2_grid.csv`).
- Adjustment seam: `Adj Close` empty ≤2026-04-03, `== Close` from 2026-04-06 in TMF/TYD/QLD/BIL; BIL 2026 YTD 0.88%; `live_signal.py:91` `auto_adjust=False`.
- VANGUARD calendar: 153 non-NYSE dates (11× Jan-1), 24 nonzero returns; union index 4255 rows vs 4102 NYSE.
- Sleeve CSV first commit 2026-06-20 (`9e9db48`); repo first commit 2026-05-22; pre-2025 rows hash-identical HEAD vs oldest commit.
- Crypto top days = GBTC o2o +68.2% (2017-12-26), +56.9% (2017-05-25), +39.0% (2017-12-22); ETHE +400% (2019-06-20, zero-volume period).
- UPRO synthetic splice: c2c +7.04% phantom jump at 2009-06-25 (`data/etfs_extended/`, non-production).
- Live-vs-backtest overlay wedge: 2026-07-01 recorded `total_mult=0.611` vs live `overlay_mult=0.623` computed the same evening; `paper_summary.json` `ann_tracking_error_bps=2829.8`, `mean_diff=+19.35` bps/day (dominated by the dating misalignment).
- TLT.csv: adjusted 2005 levels (close ≈ 44.22) with raw 2026 tail; commit `225c45b` shows 10k+ line churn in IEF.csv within a single PHOENIX refresh (daily adjusted rewrite by `daily-update.yml` + raw append by `live_signal.py`).
