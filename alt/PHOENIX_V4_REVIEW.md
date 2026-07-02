# PHOENIX v4 — Independent Re-Review of the v3.1 Rebuild, and the Fixes

**Date:** 2026-07-02
**Scope:** independent, adversarial re-review of the PHOENIX v3/v3.1 rebuild
(`claude/phoenix-strategy-review-pybwuz`, commits `0b0fb28`…`86390c0`) — the
system the previous engagement produced after finding the original PHOENIX's
Sharpe 2.34 to be an artifact. Nothing was taken on faith: every critical
claim of the previous review was re-verified against code and data, and the
replacement itself was audited with the same hostility it applied to the
original.

**Verdict up front:** the previous agent's *review of the original* is
accurate — every critical finding reproduced exactly (see §1). Its *rebuild*,
however, shipped with five significant defects of its own and one strictly
dominated design choice (see §2). v4 fixes all of them. The honest headline
drops again: **v2 advertised 2.34 → v3.1 advertised 1.24 → v4 measures 1.09**
(full 2014–2026, net; 2019+ 1.11; since-IBIT 2024+ 1.19). Each step down is a
lie removed, not performance lost.

---

## 1. The v3 review of the original PHOENIX — independently verified

Re-derived from the original code/CSVs at the pre-review baseline (`ad8db50`),
not from the review's text:

| Claim | Independent check | Verdict |
|---|---|---|
| C-1 dating misalignment fabricated the correlations | Recomputed: corr(HEL[t+1],ORI[t]) = **0.648**, corr(ORI[t],VAN[t+1]) = **0.563**, corr(HEL[t+2],VAN[t]) = **0.426** vs lag-0 \|ρ\| ≤ 0.19; blend vol 17.0% labeled vs 23.1% realigned; raw blend SR 2.19 → 1.60 realigned | **CONFIRMED** |
| C-2 QUANTUM 2010-18 history was in-sample ML output | `quantum_strategy.py`: model trained on all of IS, then "Predict over full window" includes the same rows → CSV; K never used inside CV scoring | **CONFIRMED** |
| C-3 crypto economics were Grayscale premium moves | Top booked days +68.2%/+56.9%/+36.8% are GBTC single days (premium explosions, not BTC) | **CONFIRMED** |
| H-1 adjusted/raw price seam since 2026-04-06 | `auto_adjust=False` at old `live_signal.py:91`; BIL/TMF `Adj Close == Close` exactly from 2026-04-06 | **CONFIRMED** |

The decision to retire the original numbers was correct and is kept.

## 2. What the re-review found wrong in v3.1 itself

Four independent audit passes (sleeve code, crypto/HELIOS, new sleeves +
allocator, live pipeline), each with empirical reproduction. Everything below
was verified by running code, not by reading it.

### 2.1 CRITICAL — TOM's live signal was permanently long TQQQ
`tom_strategy.py` marked "the last 4 available days of the month" as
turn-of-month. Any data tail mid-month — and every `live_extend` appended day
— is by construction among the last 4 available, so the live sleeve was long
TQQQ every single day (verified for arbitrary mid-month cutoffs), and each
mid-month cron would have frozen a wrong long-TQQQ row into the published
record daily (the freeze mechanism has no repair path). The committed history
was still clean only because the CSVs were generated on a month boundary.
**Fix:** month windows are evaluated on the full projected NYSE session
calendar (observed sessions + remaining business days minus exchange
holidays). History byte-identical; mid-month and live evaluation now correct.

### 2.2 CRITICAL — dead ETH feed fabricated ~19 flat days at 50% weight
`ETHA.csv` ends 2026-04-02 and the spot `ETH_USD.csv` at 2026-04-05; the
crypto sleeve's chained close then froze, its 63d momentum flipped positive
on the frozen numerator (2026-05-08), the sleeve "re-entered" ETH at 50%,
and 19 trading days of NaN returns were silently booked as 0.0 while ETH
actually sold off with BTC (IBIT −22% over the window). The committed May–Jun
2026 crypto tail was overstated by ~9-10% of sleeve NAV — the same *class* of
fabrication the v3 review condemned in ETHE's zero-volume prints.
**Fix:** a staleness guard (no real print in the in-use source for 3 sessions
→ asset ineligible at rebalance AND force-exited to cash daily, decided at
t−1 like every other signal), a held-asset-NaN-return counter exported to
metrics (`fabricated_days`, must be 0), and regenerated tail rows (May–Jun
2026 now book the honest −22.2% vs the fabricated −11.8%). Pre-2026 rows
unchanged (verified: 18 rows differ, all in the broken tail).

### 2.3 HIGH — the backtest booked margin the live book refuses
VANGUARD ran a constant **1.5× internal gross with zero financing cost**
(1.5× on 60% of all days; the module's own docstring admitted the multiplier
was "scaled to hit the 20% CAGR target" — in-sample calibration financed by
free margin). The blend therefore booked a mean **104.7% / max 113.9%
notional** while `phoenix_production.py` claimed "gross never exceeds 100%",
and the live layer's proportional cap silently held up to 12% less than the
backtest on exactly the days it binds.
**Fix:** VANGUARD gross capped at 1.0. Its standalone CAGR falls 22.7% →
15.6% with Sharpe unchanged (0.88) — the difference was margin, not edge.
Every sleeve is now internally gross ≤ 1, the blend notional is exactly
100%, and the live no-margin cap is an assertion instead of an active
constraint.

### 2.4 MEDIUM — the walk-forward allocator was dominated by doing nothing
The v3.1 allocator (annual refits, trailing-SR budgets `max(SR, 0.3)·(1−ρ̄)`,
budget-ERC, 35% cap) **lost to plain equal weight on every reported window**:

| | shipped WF | equal weight (same sleeves, same overlay) |
|---|---|---|
| full 2014–2026 SR / CAGR | 1.25 / 25.9% | **1.29 / 27.7%** |
| 2019+ SR / CAGR | 1.13 / 24.8% | **1.28 / 30.3%** |
| MDD | −24.3% | **−21.2%** |

Decomposition: the SR floor guaranteed budget to a losing sleeve — BND drew
15–25% of the book through six straight years of negative trailing Sharpe
(2025 weight 0.247 at fit-window SR **−0.59**; ~−20pp cumulative
contribution) — and ERC risk-equalization systematically over-funded low-vol,
low-return sleeves. Notably, the v3 research ledger tested risk parity and
mean-variance allocators but never the 1/N baseline. Two latent bugs in the
same code: the cap/renormalize loop violates the 0.35 cap whenever ≤2 sleeves
are active (0.50/0.50), and the year-boundary refit uses the Dec-31-labeled
return that completes at the same open where the new year's first position
fills.
**Fix:** v4 allocates **1/n over active sleeves** — zero estimated
parameters, nothing to overfit, live ≡ backtest weights by construction. The
WF allocator is retained behind `ALLOCATOR = "walk_forward"` with the cap
bug fixed. *Disclosed caveat:* the switch was decided after seeing the
comparison above (one full-sample structural choice, the same class as
v3.1's vol-target removal); the a-priori estimation-error case for 1/N is
the justification.

### 2.5 MEDIUM — the headline counted a decade of non-investable crypto
The rebuilt crypto sleeve books **spot BTC/ETH** returns from 2014 (+1,610%
in 2017 alone) at 30 bps costs. No ETF-mandate investor could hold that
stream before 2024 — the only listed vehicle was the Grayscale trusts, whose
premium chaos the v3 review itself rejected. The spot-proxy era materially
props up the published blend history: with crypto allocated only from IBIT's
listing, the equal-weight blend's 2019+ SR drops 1.28 → 1.10.
**Fix:** headline metrics allocate to CRY only from **2024-01-11** (IBIT's
listing — a fixed, ex-ante activation date). The spot-proxy variant is still
published in `phoenix_production_metrics.json` under
`reference_spot_proxy_crypto` with an explicit NOT-INVESTABLE warning.

### 2.6 MEDIUM — smaller defects, all fixed
- **Vol-gate NaN warm-up:** `np.where(sv <= sv_thr, 1, 0.5)` evaluates
  NaN≤NaN as False → the backtest ran the first **118 days of 2014 at half
  exposure** on an artifact, and live (correctly) treats NaN as pass — a
  live-vs-backtest edge inconsistency inside the pin window. Fixed: gate
  passes while either side is NaN, matching live.
- **Push races:** `phoenix-signal.yml`'s retry loop ended on `sleep` and
  exited 0 when every push failed (state silently lost); `daily-update.yml`
  had no rebase at all. Both now rebase-retry and fail the job loudly on
  exhaustion; daily-update's SPY writer pins `auto_adjust=True`.
- **Validator blind spots:** the seam check compared `Adj Close` to `Close`
  — freshly appended raw rows always match, so the *actual* seam passed. New
  BIL dividend-drop canary (any BIL open-to-open < −10 bps in the trailing
  90d fails; warn-only until 2026-08-01 while the heal completes), plus
  production-output currency check, and gate/EWMA/allocator params added to
  the params-drift check. The frozen pin is re-set to the v4 baseline
  (2014-2018 net SR **1.0904** ± 0.02).
- **Dead code / stale docs:** unreachable `dd` NameError branch removed from
  `live_signal.py`; HELIOS docstrings corrected (VIX_Z_CAP is 1.5, not
  0.75).

### 2.7 Data-layer state (inherited, being healed)
The price seam (raw rows appended to adjusted history since 2026-04-06) is
still in the committed CSVs — v3's full-adjusted-rewrite fetcher has never
run in CI, and the Apr–Jun 2026 sleeve rows were computed on seamed prices
(BIL ex-div days booked as losses; mostly conservative, except the crypto
fabrication in §2.2). Because the freeze mechanism would otherwise lock this
in forever, `refresh_all.py` now carries a **one-time freeze floor**
(2026-04-03, expires 2026-08-01): until expiry, crons re-derive all sleeve
rows after the floor from the healed adjusted prices. After the heal, the
BIL canary becomes a hard gate so this class of corruption cannot silently
return.

### 2.8 Affirmed clean in v3.1 (verified, no change)
The unified booking convention (`sleeve_engine.py`) is correct and
implementable; all seven sleeves pass truncation leak tests (weights before a
2024-01-01 data cutoff identical to full-data weights, max |ΔW| = 0.0);
committed sleeve CSVs reproduce from source byte-identically; the overlay's
shift(2) is decidable before the fill it scales; live weight aggregation
reproduces backtest weights exactly; QUANTUM's retirement was justified
(walk-forward SR 0.2–0.6, MDD −77%); the freeze-and-append mechanism, FRED
merge guards, and paper-trader fill logic are sound. The v3 research log's
one-shot OOS discipline (rejecting the deadband throttle and the 5-sleeve
diversifier stack) held up under scrutiny and those rejections are kept.

---

## 3. PHOENIX v4 — what production is now

- **Sleeves (7):** unchanged roster — VAN (gross now ≤ 1.0), ORI, HEL, CRY
  (staleness-guarded), REV, TOM (calendar-projected), BND. Roster changes
  were deliberately NOT made: dropping BND/TOM after observing their OOS
  decay would be selection on the same data; their damage is capped at 1/7
  and disclosed.
- **Allocation:** equal weight over active sleeves; CRY activates
  2024-01-11 (IBIT listing), everything else at blend start 2014-01-02.
- **Overlay:** 99th-percentile vol gate only (halve, shift(2)); no vol
  target, no DD throttle; 10 bps on multiplier changes.
- **Honest headline (net, investable universe):**

| Window | SR | excess-SR (vs BIL) | CAGR | Vol | MDD |
|---|---|---|---|---|---|
| 2014–2026 | **1.09** | 1.01 | **22.4%** | 20.4% | **−17.9%** |
| 2014–2018 | 1.09 | 1.06 | 18.3% | 16.7% | −16.9% |
| 2019–2026 | **1.11** | 1.00 | **25.1%** | 22.6% | −17.9% |
| 2024+ (7 sleeves live) | 1.19 | 1.03 | 29.9% | 24.4% | −16.8% |

  SPY same window: SR 0.86, CAGR 13.8%, MDD −32.0%. Reference
  (spot-proxy crypto from 2014, NOT investable): SR 1.33 / CAGR 27.5%.
- **Live layer:** unchanged mechanics; weights now trivially match the
  backtest (1/7); risk gross ≈ 79% today; v4 appears as one rebalance on the
  next trading day's open.

## 4. Reconciliation of every published headline

| Version | Advertised | What was inflating it |
|---|---|---|
| v2 (original) | SR 2.34 / CAGR 36.8% | dating misalignment (≈ −0.5 SR), in-sample QUANTUM, GBTC premium economics, IS-fitted weights, 225-grid overlay |
| v3.1 | SR 1.24 / CAGR 25.9% | unfinanced 1.5× VANGUARD gross, non-investable pre-2024 crypto, 2014 gate artifact (net of these: allocator actually *hurt*) |
| **v4** | **SR 1.09 / CAGR 22.4% / MDD −17.9%** | remaining known biases are listed below, all in the conservative-to-disclosed column |

Remaining, disclosed, unfixable-by-code: structure-level selection bias
(sleeve roster chosen knowing 2010–2026 history; PHOENIX survives a ~25-
strategy repo tournament), thin cost model on illiquid LETFs, costless daily
re-truing between rebalances, ~8 live days of paper history. Expect live
performance below backtest; the paper ledger is the only evidence that
cannot be overfit.

## 5. Operational runbook deltas (vs the v3 handoff)

1. Cron flow unchanged (`phoenix-signal.yml` → `refresh_all.py` → gate on
   `validate_state.py`), but the first runs after deploy will print
   `[REBASE]` lines while the freeze floor re-derives Apr–Jun 2026 rows from
   healed prices — expected until 2026-08-01, then the floor expires and the
   BIL canary hardens.
2. Manual full run: unchanged commands; `phoenix_production.py` now prints
   the 2024+ window and the labeled non-investable reference line.
3. Re-pinned: `validate_state.FROZEN_2014_2018_SHARPE = 1.0904`.
4. `docs/phoenix.html` fully rewritten: honest numbers, the version-history
   table (2.34 → 1.24 → 1.09 with reasons), investability rules, and live
   positions — injected daily as before (`const F/A/L/LIVE/PAPER`).
5. Next agent: read `alt/PHOENIX_REVIEW.md` (how the original lied),
   `alt/RESEARCH_LOG_V3PLUS.md` (what already failed honestly), and this
   file (how the rebuild lied) before trusting or "improving" anything. The
   rules of engagement in `PHOENIX_HANDOFF.md` §8 stand, with one addition:
   **any comparison used to justify a design change must include the
   do-nothing baseline** (1/N, no-overlay, hold-BIL) — the v3.1 allocator
   shipped because that row was missing.
