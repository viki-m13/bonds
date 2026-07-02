# PHOENIX v3 — Corrected & Rebuilt

**Date:** 2026-07-02
**Predecessor:** the original PHOENIX advertised Sharpe 2.34 / CAGR 36.8% / "5 uncorrelated sleeves (max |ρ| = 0.19)". The independent review (`alt/PHOENIX_REVIEW.md`) showed those numbers were artifacts: misaligned return dating fabricated the correlation matrix and leaked into the overlays, QUANTUM's 2010–2018 history was in-sample ML output, and the crypto sleeve's economics were Grayscale premium moves the live book can't earn. This document records the rebuild: what was fixed, how the new system was designed, what it honestly delivers, and the verdict on the stated performance target.

---

## 1. The target, and the honest verdict up front

**Target set for this rebuild:** OOS Sharpe ≥ 2.0 and CAGR ≥ 25%, ETFs only (leveraged/inverse allowed), next-day-open execution, no portfolio margin (gross ≤ 100%).

**Verdict: not honestly achievable with this signal family — and the evidence says no daily-rebalanced long-only ETF rotation should be trusted if it claims otherwise.** What the rebuilt, leak-free system actually delivers:

| Window | Sharpe | Excess-Sharpe (vs BIL) | CAGR | Vol | MDD |
|---|---|---|---|---|---|
| 2014–2026 (full, walk-forward) | **1.17** | 1.07 | **20.0%** | 16.7% | −23.3% |
| 2014–2018 | 1.32 | 1.29 | 21.5% | 15.7% | −20.9% |
| 2019–2026 | **1.09** | 0.95 | **19.0%** | 17.4% | −23.3% |

For scale: SPY over the same period is Sharpe ~0.9 with a −32% MDD; the old "Sharpe 2.34" was never real (§5 of the review quantifies each artifact's contribution). The gap between an honest ~1.1–1.3 and the target 2.0 is not a tuning problem:

- The best IS blend Sharpe reachable across ~40 disciplined design experiments (§4) was ~1.5, and IS→OOS degradation for the *best-in-IS* configuration was −0.5 (1.49 → 1.02). Anything engineered to print OOS 2.0 from this menu would be fit *to* the OOS window — exactly the sin the review documents.
- The structural reason: at next-day-open, daily-bar frequency with long-only ETF exposure and gross ≤ 1, the available edges (trend, dip-buying, seasonality, crypto momentum) each carry Sharpe 0.5–1.0 and, realigned in real time, correlate 0.3–0.7. Diversification math caps the honest blend near √N_effective × 0.8 ≈ 1.3–1.6 before costs. Getting to 2+ requires genuinely new information (intraday/overnight structure, options premia, cross-asset futures, alternative data) or leverage — both outside the constraints.
- What WAS delivered vs the old system, like-for-like honest measurement: realigned old PHOENIX ≈ Sharpe 1.45–1.81 with contaminated inputs and unimplementable overlays; v3 is fully implementable, leak-free, walk-forward-weighted, and paper-trade reconcilable. CAGR ≥ 25% alone is reachable by raising the vol target (the book is capped at 100% gross and currently targets 18%; at ~24% target vol the same Sharpe implies ~26% CAGR with ~−33% MDD) — but that trades directly against drawdown, and the review's position is that the Sharpe target should govern.

Everything below documents how v3 was built so this claim chain is auditable.

---

## 2. What was fixed (all review findings addressed)

| Review finding | Fix |
|---|---|
| C-1 misaligned booking conventions (fabricated correlations, overlay lookahead) | Single realization-dated convention in a shared engine (`alt/sleeve_engine.py`): the value at date t is the P&L over open[t−1]→open[t] from a position decided at close[t−2]. ORION/HELIOS/QUANTUM converted; VANGUARD/CRYPTO already conformed. `refresh_all.py` lookahead trims now 0 everywhere — no more placeholder-zero rows, and the live overlay input is always complete. |
| C-2 QUANTUM in-sample history | Rebuilt walk-forward (annual expanding refits, N-day embargo, N selected by CV inside the first window only, K=3 declared constant — the K-never-used CV bug is documented in the module). Result: Sharpe 0.2–0.6 across horizons, MDD −77%, fragile → **retired from production** (module kept for research). |
| C-3 crypto premium economics + instrument mismatch | Sleeve rebuilt on **spot BTC-USD/ETH-USD** (NYSE-calendar, return-space chained into **IBIT/ETHA** from their listing dates — no splice jump), 30 bps proxy-era / 10 bps ETF-era costs, ETHE's garbage prints gone. Honest result: full SR 0.88, CAGR 34%, IS 0.95 / OOS 0.82. Live and backtest now hold the same instrument. Crypto stays in the book (the original PHOENIX included it) at whatever weight the allocator assigns (~14% for 2026). |
| H-1 adjusted/raw price seam | `live_signal.fetch_latest` now does **full-history `auto_adjust=True` rewrites** (with a ≥90%-rows guard), same basis as `daily-update.yml`'s writer — no seam, and splits can no longer freeze fake returns. Sleeve-return freezing (unchanged) is what protects the published record from vendor revisions. |
| H-2 HELIOS unlagged signals (live drift + backfill lookahead) | HELIOS momentum/eligibility/macro gate all lagged one day; its `live_extend` semantics now identical to every other sleeve. |
| H-3/H-4 stale docstrings, grid-argmax overlays | Docstrings rewritten with real numbers; overlay params are now a-priori standards (EWMA λ=0.94, round-number 18% target) applied with a strict 2-day shift (decidable at close[t−2] for the open[t−1] fill — the old shift(1) was undecidable for backward-dated sleeves). |
| VANGUARD calendar/trigger bugs | Real NYSE calendar (153 phantom rows incl. Jan-1 gone), NaN-asymmetric SPY trigger fixed, cost booked on trade date. |
| ORION/HELIOS warm-up dead zones | Signals warm up on pre-2010 history; scored windows start hot. |
| No-margin edge case (~112% possible) | Live layer hard-caps risk gross at 100%. |
| Validation not gating / wrong checks | `validate_state.py` rewritten: params vs code, per-sleeve freshness/dup/trailing-zero/absurd-return checks, price-basis seam canary, frozen-window Sharpe stability (±0.02), live sanity (sum≈100%, gross≤100.5%, mult∈[0,1]) — and `refresh_all.py` now **exits nonzero on failure** so the workflow doesn't commit corrupted state. |
| Workflow push race | `git pull --rebase` before push in `phoenix-signal.yml`. |
| Sharpe definition | Excess-Sharpe (vs BIL) reported alongside raw in production metrics. |

## 3. The v3 system

**Seven sleeves** (each a module with the same `build_weights()` contract, returns generated through the shared engine):

| Tag | Module | Strategy | Honest standalone (full) |
|---|---|---|---|
| VAN | `vanguard_strategy.py` | Monthly rotation {QLD,UGL,TMF,TYD}, 189d momentum + 200dma, inverse-vol, 4-trigger macro participation, 1.5× internal gross | SR 0.88, CAGR 22.7% |
| ORI | `orion_strategy.py` | Weekly top-4 momentum of 12 risk LETFs + top-2 safe book, macro-gated | SR 0.95, CAGR 23.6% |
| HEL | `helios_strategy.py` | Weekly underlying-trend (42/189d) expressed via LETFs, defensive bypass | SR 0.70, CAGR 24.2% |
| CRY | `phoenix_v2_crypto.py` | Weekly 63d TSMOM on spot-chained BTC/ETH → IBIT/ETHA, macro-gated | SR 0.88, CAGR 34.0% |
| REV | `reversal_strategy.py` | 5d dip-buying inside uptrends (QQQ/SPY/SMH z<−1 → TQQQ/UPRO/SOXL, 5d hold) | SR 0.77, CAGR 26.4% |
| TOM | `tom_strategy.py` | Turn-of-month TQQQ (last 4 + first 3 sessions) | SR 0.54, CAGR 13.6% |
| BND | `bondtrend_strategy.py` | TLT 63d momentum long/flat via TMF, weekly | SR 0.29, CAGR 4.1% |

**Allocator — walk-forward, no in-sample weights:** each January 1 the sleeve weights are refit from the trailing 4 years only: risk budgets `max(trailing SR, 0.3) × (1 − ρ̄)`, long-only risk-budget ERC on the trailing covariance, sleeves excluded while trailing vol < 5% (inactive), weights capped at 35%. Weights are held for the calendar year. 2026 weights: VAN 0.19, BND 0.21, ORI 0.18, CRY 0.14, HEL 0.10, REV 0.09, TOM 0.08. The blend starts 2014 (first 4-year window). **There is no IS-fitted production weight anywhere.**

**Overlays** (strictly implementable; multiplier scaling the open[t−1]→open[t] return is a function of blend returns through t−2): EWMA(0.94) vol target 18% with multiplier in [0.25, 1.0] (no margin); DD throttle vs 252d HWM with −10% floor (throttles exposure — the strategy's own MDD is −24%, not −10%); 99th-percentile vol gate halving; 10 bps on multiplier changes. Average gross exposure ≈ 85%.

**True correlations** (2014–2026, contemporaneous): the trend trio VAN/ORI/HEL correlate 0.5–0.7 — they are one family, sized accordingly by the allocator; CRY ~0.05 to everything; REV/TOM 0.3–0.5 to trend; BND ~0.1. The webapp's "uncorrelated" language has been retired.

## 4. Design protocol (how overfitting was bounded)

- All sleeve/parameter selection ran on **IS = 2010–2018 only** (research scripts in the session scratchpad print no OOS). Candidates evaluated: bond long/short trend (TMV/TBT short legs — rejected on IS), 5-name reversal (rejected: 3-name better), trend-filtered TOM (rejected: filter destroys the effect), SVXY carry (rejected), crisis-convex TMF/SQQQ basket (rejected), ATLAS per-asset TSMOM (rejected: ρ=0.85 to ORION), crypto momentum-speed blends and vol-scaling (rejected: no IS improvement), dynamic risk-parity capital allocation (rejected: starves CAGR at gross ≤ 1), mean-variance weights (rejected: unstable, zeroes sleeves).
- OOS (2019–2026) was evaluated **once** for the locked IS-best config: IS 1.49 → OOS 1.02. That degradation was reported, not iterated on. The production allocator was then switched to the walk-forward protocol above — which never fits on the full sample at all — and the shipped numbers (§1) are that protocol's.
- Residual honesty caveats, disclosed: (a) the *structure* (which 7 sleeves exist, overlay form) was chosen knowing 2010–2026 history in aggregate — structure-level selection bias cannot be walked forward away; (b) the repo's ~25 failed sibling strategies mean PHOENIX itself survives a tournament; (c) costs are modeled at 5–10 bps/side on LETFs (30 bps proxy-era crypto) with costless daily re-truing to targets between rebalances; (d) HELIOS/ORION Friday-holiday weeks skip rebalance; (e) live history begins 2026-06-20 — everything else is backtest.

## 5. Reconciliation to the old claims

| Claim | Old | Honest |
|---|---|---|
| Sharpe (headline) | 2.34 | 1.17 full / 1.09 recent-era (v3 walk-forward) |
| CAGR | 36.8% (docstring said 57.4%) | 20.0% |
| Max pairwise sleeve \|ρ\| | 0.19 | 0.70 (VAN–ORI, realigned) |
| MDD | −17.6% | −24.3% |
| Crypto sleeve economics | GBTC premium (+68% days), untradable live | spot/ETF, tradable |
| QUANTUM IS Sharpe | 2.73 (in-sample) | 0.2–0.6 walk-forward → retired |
| Overlay implementability | shift(1) on mixed dating = up to 2 days of lookahead | strict shift(2), live formula identical |
| Production weights | fit once on contaminated IS | refit each January from trailing 4y only |

## 6. Operational notes

- All seven sleeve CSVs in `data/results/` were regenerated under the new convention in this change — this is a deliberate re-baselining; the freeze mechanism protects history from here forward. `validate_state.py` pins the 2014–2018 net Sharpe at 1.3179 ± 0.02.
- `quantum_returns.csv` still exists (walk-forward version) but is not consumed by production.
- BTC-USD/ETH-USD spot CSVs are stale (2026-04-05); harmless — post-2024 signals and returns use IBIT/ETHA, and the fetcher now refreshes spot too.
- The next cron run will rewrite all price CSVs on the single adjusted basis, repairing the April–June 2026 dividend seam in the price layer (the affected ~60 days of frozen sleeve returns carry a small known error, conservative for bond sleeves' momentum signals; documented rather than rewritten).
- `docs/phoenix.html` had fresh JSON injected; its narrative copy (the "uncorrelated √N" story) still needs a rewrite to match §3 — flagged as follow-up.
- Paper trading continues on the same ledger; the strategy change appears as one rebalance on 2026-07-02's open.
