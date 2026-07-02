# PHOENIX — Complete Handoff Record

**Date:** 2026-07-02 · **Branch:** `claude/phoenix-strategy-review-pybwuz`
**Purpose:** everything done in this engagement — the review of the original PHOENIX, every issue found, everything rebuilt, every experiment run, and where every artifact lives — so any agent (or human) can pick up from here with zero context loss.

---

## 0. Document map (read in this order)

| Document | Contents |
|---|---|
| **This file** | Master index: timeline, change inventory, current state, ops runbook, research ledger, open items |
| `alt/PHOENIX_REVIEW.md` | The full independent review of the ORIGINAL PHOENIX: every finding with severity, file:line evidence, reproduction numbers, affirmed-clean list |
| `alt/PHOENIX_V3.md` | The rebuild: v3/v3.1 design, protocol, corrected results, target-feasibility verdict, reconciliation to old claims, §1b vol-target frontier |
| `alt/RESEARCH_LOG_V3PLUS.md` | The invention loop: all 6 research iterations, every experiment (including all dead ends), adoption/rejection verdicts with evidence |
| `alt/research_v3plus/` | Persisted research artifacts: candidate return streams, lab scripts, agent reports, evaluation series (see §7) |

Session commits (this branch, oldest first): `0b0fb28` review · `89ac2db` v3 rebuild · `c03d58d` iter-1 log · `6ea39dc` v3.1 (vol target dropped) · `50e0800` cash fix · `c499593` iter-5 + dead-throttle removal · `a7e0a7a` iter-6 log · (this commit) handoff + artifacts.

---

## 1. Timeline of the engagement

1. **Review** (commit `0b0fb28`): line-by-line audit of the original 5-sleeve PHOENIX (advertised Sharpe 2.34, CAGR 36.8%, "5 uncorrelated sleeves max |ρ|=0.19") plus its data layer and live pipeline. Verdict: headline numbers were artifacts (§2). No code changed; findings recorded.
2. **Correction + rebuild → v3** (`89ac2db`): every review finding fixed; QUANTUM retired; CRYPTO rebuilt on spot; 3 new sleeves added (designed IS-only); walk-forward allocator replaced the IS-fit weights; pipeline hardened.
3. **v3.1** (`6ea39dc`): the EWMA vol target removed — under the no-margin cap it is pure drag (frontier in PHOENIX_V3.md §1b).
4. **Cash fix** (`50e0800`): implicit sleeve cash routed to BIL (live already holds BIL) — free, risk-identical gain.
5. **Research loop, 6 iterations** (`c03d58d`, `c499593`, `a7e0a7a`): ~160 experiments across every idea family incl. 4 parallel research agents; two survivors (both corrections); a fixed DD-throttle and 5 IS-strong candidate sleeves rejected on the one-shot OOS check; new data channels (High/Low/Volume) shown empty.
6. **Equity-curve artifact**: interactive PHOENIX-vs-SPY chart (also archived at `alt/research_v3plus/series/phoenix_vs_spy.html`).

---

## 2. All issues identified in the ORIGINAL PHOENIX (condensed; full detail in PHOENIX_REVIEW.md)

**Critical**
1. **Cross-sleeve return-dating misalignment** — sleeves booked "day-t" over four different physical windows (VAN/CRY: open[t−1]→open[t]; QUA: close-to-close; ORI: open[t]→open[t+1]; HEL: open[t+1]→open[t+2]). This fabricated the near-zero correlation matrix (true realigned VAN/ORI/HEL ρ = 0.43–0.70), understated blend vol by ~32%, made the blend NAV non-executable, and leaked up to 2 days of future information into the overlay path. Realigned honestly: Sharpe 2.34 → 1.81.
2. **QUANTUM's 2010–2018 history was in-sample XGBoost output** (model trained on the same years it was "backtested" on; IS 2.73 vs honest OOS 0.87), contaminating the IS-fit blend weights. Plus a CV bug: K was never used in the hyperparameter search.
3. **CRYPTO sleeve economics were Grayscale premium moves, not BTC** (top days +39–68% GBTC premium explosions; ETHE zero-volume +400% prints feeding the signal), at 10 bps costs on 50–300 bps-wide OTC trusts, while live traded IBIT — an instrument that cannot earn that stream. Backtest held GBTC after 2024; live held IBIT.

**High**
4. **Adjusted/raw price seam** — since 2026-04-06 the incremental fetcher appended raw rows onto adjusted history (`auto_adjust=False`); every dividend after the seam booked as a fake loss (BIL ~25bps/month, TMF/TYD/UBT ~4%/yr); frozen forever by the history-freeze.
5. **Two racing workflows** wrote the same price CSVs on incompatible bases (daily-update rewrote SPY/TLT/IEF fully-adjusted daily; PHOENIX appended raw), with push-without-rebase races dropping whole days of state.
6. **No split handling** in the append-only fetch — a split would freeze a fake ±N00% day permanently.
7. **Whole record retrospective + selection multiplicity** — repo born 2026-05-22, sleeve CSVs frozen 2026-06-20, ~8 live days; PHOENIX = winner of a ≥25-strategy in-sample tournament; "OOS 2019–2026" was realized history at design time.
8. **Overlay params were the argmax of a 225-point grid with OOS visible** (mitigant: flat grid).
9. **Stale docstring** advertising Sharpe 2.37 / CAGR 57.4% / 20% target / 2.0x cap vs actual code.
10. **HELIOS signals and macro gate unlagged** → live traded a day early on stale signals; backfilled rows contained genuine lookahead.
11. **Live overlay computed on placeholder-zero trailing rows** → permanent live-vs-backtest wedge (verified 0.611 vs 0.623 same evening); paper "tracking error" 2,830 bps/yr was mostly the dating artifact.

**Medium/Low (selection)** — VANGUARD: fabricated `bdate_range` calendar (153 phantom days incl. 11× Jan-1), NaN-asymmetric SPY trigger, 1.5× gross with zero financing, universe picked on full-sample Sharpe with dead `LEV_UNIVERSE` code; ORION: 256 dead warm-up days, fabricated trailing 0.0; HELIOS: costless daily re-truing, IS-tuned gate with contradictory docstring, ships 0.0 placeholders daily; FRED T+1 lag / no point-in-time macro; Sharpe without risk-free; `validate_state.py` non-gating and blind to the real failure modes; `apex/sleeves_phoenix_exact.py` a divergent "EXACT" clone with an assertion-free test; paper ledger mutable; `_macro_snapshot` ignoring as-of; synthetic LETF builder flaws (quarantined from production).
**Found later (iteration 5):** the **DD throttle was inert** — `(1 + dd/DD_FLOOR).clip(0,1)` is ≥1 for every drawdown, so it multiplied by exactly 1.0 forever, in the original AND in v3 (the review had missed this sign bug; a research agent caught it).

**Affirmed clean in the original:** per-sleeve signal lagging (except HELIOS), IS inverse-vol weights exactly as claimed, no synthetic data in production, pre-seam prices properly adjusted, freeze mechanism byte-stable, block-bootstrap + survivorship self-tests existed.

---

## 3. What was built (current production: PHOENIX v3.1 + fixes)

### 3.1 System spec
- **Unified booking convention** (`alt/sleeve_engine.py`): ret[t] = W[t−1]·(open[t]/open[t−1]−1) − costs; W[t] decided from ≤ close[t−1]. All sleeves contemporaneous; last CSV row fully known same-day (all `lookahead_days=0`).
- **7 sleeves** (each a module exposing `build_weights(live_extend)` + `main()` → `data/results/<name>_returns.csv`):
  | Tag | Module | Strategy | Standalone full SR / CAGR |
  |---|---|---|---|
  | VAN | vanguard_strategy.py | monthly rotation {QLD,UGL,TMF,TYD}, 189d mom + 200dma, inv-vol, 4-trigger macro ladder, 1.5× internal gross, residual→BIL | 0.88 / 22.7% |
  | ORI | orion_strategy.py | weekly top-4 of 12 risk LETFs by 252d mom > 200dma, VIX<30 & HY<7 gate + top-2 safe book, residual→BIL | 0.95 / 23.6% |
  | HEL | helios_strategy.py | weekly 42/189d underlying-trend via LETFs, signals lagged 1d (v3 fix), defensive bypass, BIL residual | 0.70 / 24.2% |
  | CRY | phoenix_v2_crypto.py | weekly 63d TSMOM on spot BTC/ETH chained (return-space) into IBIT/ETHA, 30/10 bps era costs, macro gate | 0.88 / 34.0% |
  | REV | reversal_strategy.py | 5d dip-buy in uptrends: QQQ/SPY/SMH z5<−1 → TQQQ/UPRO/SOXL, 5d hold, residual→BIL | 0.77 / 26.4% |
  | TOM | tom_strategy.py | turn-of-month TQQQ (last 4 + first 3 sessions), residual→BIL | 0.54 / 13.6% |
  | BND | bondtrend_strategy.py | TLT 63d mom long/flat via TMF, weekly Wed, residual→BIL | 0.29 / 4.1% |
  QUANTUM retired (walk-forward rebuild in `quantum_strategy.py` shows SR 0.2–0.6, MDD −77%; kept for research, not in production).
- **Allocator** (`phoenix_production.py`): each Jan 1, weights refit from trailing 4y only — budgets `max(SR,0.3)×(1−ρ̄)`, long-only budget-ERC, <5%-vol sleeves excluded, 0.35 cap. Blend starts 2014-01-02. **No IS-fitted weights anywhere.** 2026 weights ≈ VAN .19 / BND .21 / ORI .18 / CRY .14 / HEL .10 / REV .09 / TOM .08.
- **Overlay**: vol-regime gate only (halve when 60d vol > trailing-252d 99th pct, shift(2), 10bps on multiplier changes). No vol target (v3.1 — pure drag under gross≤1 cap, frontier in PHOENIX_V3.md §1b). No DD throttle (historical formula inert; corrected version REJECTED on OOS — de-risks into V-recoveries; RESEARCH_LOG iteration 5).
- **Headline (honest, walk-forward, net):** full 2014–2026 **SR 1.24 / CAGR 25.9% / MDD −24.3%**; 2019–2026 **SR 1.13 / CAGR 24.8%**; excess-SR (vs BIL) 1.16/1.02. SPY same-period ≈ 0.86 / 13.8% / −32%. $10k → $171k vs $50k (chart: `research_v3plus/series/phoenix_vs_spy.html`).

### 3.2 File-by-file change inventory
| File | Change |
|---|---|
| `alt/sleeve_engine.py` | NEW — shared unified-convention backtest + metrics |
| `alt/vanguard_strategy.py` | real NYSE calendar (phantom days gone), NaN-trigger fix, engine booking, BIL residual |
| `alt/orion_strategy.py` | engine booking (+1d re-dating), warm-up dead zone removed, BIL residual |
| `alt/helios_strategy.py` | signals+gate lagged 1d, engine booking (+2d re-dating), warm-up fix, live_extend normalized |
| `alt/quantum_strategy.py` | full walk-forward rewrite (annual expanding refits, embargo, CV-K bug fixed, N picked in first window only) |
| `alt/phoenix_v2_crypto.py` | full rewrite: spot-chained BTC/ETH → IBIT/ETHA, era costs, ETHE junk gone |
| `alt/reversal_strategy.py`, `tom_strategy.py`, `bondtrend_strategy.py` | NEW sleeves (IS-designed; variants documented in modules) |
| `alt/phoenix_production.py` | full rewrite: WF allocator, overlay (gate-only after v3.1+iter5), excess-Sharpe reporting, honest docstring |
| `alt/live_signal.py` | dynamic WF weights via `prod.current_blend_weights()`, 7-sleeve aggregation, overlay parity, **full-history `auto_adjust=True` price rewrites** (kills seam + split risk; ≥90%-rows guard), no-margin hard cap, fetch map for BTC-USD/ETH-USD |
| `alt/refresh_all.py` | 7-sleeve map (lookahead 0), fetch universe, factsheet/audit derive weights per-year from production, **validation gates the run (exit 1)** |
| `alt/validate_state.py` | full rewrite: params-vs-code, per-sleeve freshness/dups/trailing-zeros/absurd-returns, price-basis seam canary, frozen-window pin (2014-18 SR **1.4765** ± 0.02), live sanity (sum≈1, gross≤1.005, mult∈[0,1]) |
| `.github/workflows/phoenix-signal.yml` | pull-rebase before push (race fix) |
| `data/results/*` | all sleeve CSVs + production outputs regenerated = new frozen baseline (deliberate re-baselining, 2026-07-02) |

---

## 4. Operations runbook

- **Daily cron**: `phoenix-signal.yml` 23:30 UTC weekdays → `alt/refresh_all.py`: adjusted full-history price fetch → extend sleeve CSVs (freeze-and-append, all lookahead 0) → `phoenix_production.py` → factsheet/audit → backfill → `live_signal.py` → `paper_trader.py` → `compute_performance.py` → HTML inject → **`validate_state.py` (gating — failed validation aborts the commit)**.
- **Manual full run**: `python3 alt/{vanguard,orion,helios,reversal,tom,bondtrend}_strategy.py && python3 alt/phoenix_v2_crypto.py && python3 alt/phoenix_production.py && python3 alt/live_signal.py --skip-fetch && python3 alt/validate_state.py`.
- **Frozen history**: sleeve returns ≤ committed baseline are protected by `extend_sleeve_preserving_history`; the validator pins 2014–2018 net SR at 1.4765±0.02. If you deliberately re-baseline (change a sleeve), regenerate all CSVs, re-pin, and say so in the commit.
- **Live state**: live signals began 2026-06-20 (old system) / 2026-07-02 (v3.1 — appears as one rebalance). Paper ledger continues in `data/results/paper_*.csv` (`live_positions.csv` is the mutable source; a known audit-trail weakness, see §6).
- **Webapp**: `docs/phoenix.html` gets fresh JSON injected daily; its narrative copy still tells the old "uncorrelated √N" story — **rewrite pending** (open item).

## 5. Research program — protocol and complete ledger

**Protocol** (non-negotiable for the next agent): design/select on IS = 2010-03-11..2018-12-31 only (allocator work: the 2014–2018 walk-forward segment); never look at post-2018 until a configuration is LOCKED; one OOS shot per locked release candidate; log every experiment including failures in `RESEARCH_LOG_V3PLUS.md`; ship only what survives the OOS shot AND beats shipped production on the user's criteria (higher SR *and* CAGR).

**Ledger** (details + exact numbers in RESEARCH_LOG_V3PLUS.md):
- Iter 1: tier-rotation ✗, staggering ✗, z-scaled REV ~, **breadth+credit gate ✓ standalone** (`invent_lab1.py`)
- Iter 2: BC gate into ORION (+0.09 standalone) but **blend-neutral** → not shipped (`invent_lab2.py`)
- Iter 3: spread-mom pairs ✗, bond-TOM ✗, defensive basket ✗, commodities ✗ (`invent_lab3.py`)
- Iter 4: vol-target frontier → **v3.1 shipped** (target removed)
- Iter 5 (4 parallel agents, ~60 experiments): **shipped** cash fix + dead-throttle removal; **rejected on one-shot OOS**: corrected deadband throttle (segment 1.46→1.65 but 2019+ worse) and the 5-sleeve diversifier stack (segment SR up to 2.05 → OOS 0.78–1.01 vs shipped 1.13). ~25 more ideas killed at the sleeve/allocator level.
- Iter 6: High/Low/Volume channels (CLV, NR-N, OBV/$vol) ✗✗✗; softmax/ensembles ✗; Parkinson gate = unshipped candidate (`invent_lab6.py`)

**Three independent convergences**: ~160 experiments; the only survivors were corrections, never new signals. Conclusion on the standing Sharpe-2 target: not honestly reachable with daily next-open long-only ETF rotation at gross ≤ 1 and realistic costs.

## 6. Artifact inventory (`alt/research_v3plus/`)

- `candidates/*.csv` — 25 candidate daily return streams (Date,ret; full window; **post-2018 stats were never inspected during research** — treat them as still-blind if you extend the work): gh52 family, hyg_lead_sso, smh_lead_tqqq, sector_gh_hedge, cal_* seasonality set, crash_recovery, svxy_postpanic, vix_postpanic, credit_breadth*, volregime_sm, disp_lo_mom, alloc_deadband_ddthrottle.
- `labs/*.py` — every research script (mine + agents'): reproducible IS experiments.
- `agent_reports/*.md` — condensed findings of the 6 review agents + 4 research agents.
- `series/` — v3 candidate/WF/v3.2 evaluation series, per-experiment result CSVs, QUANTUM N-sensitivity log, the PHOENIX-vs-SPY chart page.

## 7. Known caveats & open items (ordered)

1. **Webapp narrative** (`docs/phoenix.html`) still carries pre-review claims in prose — rewrite to match PHOENIX_V3.md §3.
2. **Apr–Jun 2026 frozen sleeve rows** were computed on seam-era prices (small, mostly bond-sleeve dividends); price layer self-heals via the new fetcher but those ~60 frozen return rows carry a small known error (documented, conservative direction).
3. **Non-core tickers stale at 2026-04-02** (TBT/TMV/SQQQ/SVXY/PSQ/SH etc.) — only matters if a future sleeve uses them; the fetch list covers production tickers only.
4. `daily-update.yml` (the *other* workflow) still rewrites shared bond CSVs daily — bases now match (both adjusted), but single-writer-per-file would be cleaner.
5. **Paper ledger is rebuilt from a mutable positions log** — make append-only for a true audit trail.
6. `apex/sleeves_phoenix_exact.py` remains a divergent "EXACT" clone with an assertion-free test — rename or pin.
7. Residual honesty caveats on the backtest: structure-level selection bias (which sleeves exist was decided knowing 2010–2026 in aggregate), costless daily re-truing to targets between rebalances, 5–10bps flat costs thin for YINN/TYD-class names, Friday-holiday weeks skip HELIOS/CRYPTO rebalances, live record starts 2026-07-02 for v3.1.
8. **Unshipped candidates worth revisiting with NEW evidence only** (longer selection basis or live data — do NOT re-select on the same OOS): Parkinson H/L gate; ORION-BC gate (standalone +0.09, blend-neutral); turnaround-Tuesday sleeve (perfect IS halves, corr 0.07); GH 52w-high ranking as an ORION upgrade.
9. **Structural paths past the ceiling** (require new capabilities): close-auction/overnight execution, options-premium sleeves (needs options data), futures trend with real leverage, intraday data, FOMC/event dates file for event studies.

## 8. Rules of engagement for the next agent

1. Read PHOENIX_REVIEW.md before trusting any number in this repo — it is the catalog of how this codebase's numbers have lied before.
2. Never mix booking conventions; everything goes through `sleeve_engine.backtest_weights`.
3. Design on IS only; one OOS shot per locked candidate; log failures; the dead ends in RESEARCH_LOG are the honesty budget — do not silently re-run them.
4. Anything that changes sleeve CSVs is a re-baselining: regenerate everything, re-pin the validator, say so explicitly.
5. The paper ledger is the only evidence that cannot be overfit. Guard it.
