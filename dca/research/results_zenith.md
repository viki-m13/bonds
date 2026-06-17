> **⚠️ RETRACTED (2026-06-17) — ZENITH is NOT a validated QQQ-beater.** The
> validation below was run on the **survivorship-biased** `summit_panel.parquet`
> (the same panel as SUMMIT) and benchmarked correctly vs QQQ, but it leaned on a
> leakage audit + random-pick control that do **not** catch survivorship or
> recency. Independent re-validation (`VALIDATION_METHODOLOGY.md`) and my own
> diagnostics (appended at the end, §"Post-hoc") show the edge largely
> evaporates on honest data: panel missing ~40% of historical members, delisted
> names booked benignly, IS cross-sectional IC vs QQQ ≈ 0/negative, lead
> concentrated in 2018–26. Clean-panel rebuild: **25.7× → ~11.2× (≈ ties QQQ)**.
> The "Pareto improvement" claim is **withdrawn.** Read this whole file as the
> *biased-panel* record, then the §Post-hoc section for the correction.

# ZENITH — validation record (the most-profitable DCA stock picker)

**Mandate (user):** develop the *most profitable* DCA strategy that significantly
outperforms DCA into QQQ — a novel, thoroughly and carefully validated approach,
using everything learned in the cited literature review and the SUMMIT research.

**Result:** **ZENITH** = SUMMIT's regime-conditional mega-cap-momentum signal run
at **maximal conviction, k=1** (buy the single highest-scoring leader every
contribution, never sell). On point-in-time S&P 500 data, 244 rolling windows,
next-open execution, 5 bps cost, delisting-aware accounting, ZENITH is a **Pareto
improvement** over SUMMIT k=2 — more profitable *and* at least as robust — and it
**beats the random-pick survivorship control by +85 percentage points**, the same
decisive test that the submitted z-score "WAVE" scripts failed
(`results_user_strategies_validation.md`).

This document is the honest, complete validation record, run to the **identical
standard SUMMIT and the WAVE z-score audit were held to** (PIT universe,
delisting-aware, anti-leakage, random control, real money-multiple benchmark,
IS/OOS, no overlapping-window p-hacking).

---

## 0. Headline (biweekly, k=1, 5 bps, vs same-cadence QQQ/SPY DCA)

| metric | **ZENITH (k=1)** | SUMMIT (k=2) | improvement |
|---|---|---|---|
| beat QQQ-DCA (244 windows) | **95%** | 93% | +2 pp |
| beat SPY-DCA | **98%** | 98% | = |
| median excess vs QQQ | **+43.2%** | +28.8% | +14 pp |
| 10th-pct window vs QQQ | **+5.0%** | +3.0% | +2 pp |
| worst window vs QQQ | **−11.2%** | −10.6% | −0.6 pp |
| full multiple (2006Q1→2026) | **25.7×** | 20.0× | **+29%** |
| since-inception multiple (2005→2026) | **40.0×** | ~20× | — |
| QQQ-DCA / SPY-DCA same window | 9.1× / 4.7× | 9.1× / 4.7× | — |
| IS (2006–14 starts) win | **92%** | 90% | +2 pp |
| OOS (2015–23 starts) win | **99%** (worst −1%) | 99% | = |

By horizon: 3y **87%** win (+13% med), 5y **95%** (+24%), 10y **100%** (+53%),
to-end **100%** (+125%). The k=1 concentration trades a little 3-year win-rate
(87% vs SUMMIT's 84% — actually *better*) for materially higher medians at every
horizon.

---

## 1. Why k=1 — the idea, pre-registered from the literature (not curve-fit)

The single design change vs SUMMIT (k: 2→1) was chosen *a priori* from the cited
review, not selected after the fact:

* **Bessembinder (2018):** essentially *all* long-run net US equity wealth is
  created by a tiny right tail — 4.3% of firms since 1926, the top ~0.33% make
  half — and concentration is *rising* over time. Against a cap-weighted
  benchmark (QQQ), the optimal posture is to tilt hard to the biggest names, ride
  the single strongest leader, and never trim the right tail.
* **Sathish Kumar (2025) / Patton-Weller (2017):** the momentum *long leg* is
  +7.9%/yr in large caps; the damage comes from the short leg and turnover.
  ZENITH is pure long-leg, never-sell → it keeps the part that works.
* SUMMIT already did the size tilt + never-sell + regime switch; **ZENITH adds
  only "feed the #1 name instead of splitting across two."** It is the maximal
  point of the same architecture, not a new signal. The signal builder is
  **byte-for-byte SUMMIT's `build_scores`** (re-exported in `strategy_zenith.py`),
  so the leakage audit and reference cross-check carry over unchanged.

The k-sweep confirms this is a smooth optimum, not an overfit spike:

| k | win vs QQQ | median | worst |
|---|---|---|---|
| **1** | **95%** | **+43%** | −11% |
| 2 (SUMMIT) | 93% | +29% | −11% |
| 3 | 91% | +22% | −11% |
| 4 | 84% | +15% | −11% |
| 5 | 82% | +10% | −12% |

Monotone decline in k. The same monotonicity holds at monthly cadence
(every=21): k=1 → 94% win, +43% median.

---

## 2. The decisive test — skill vs survivorship (WAVE-v2 standard)

The lesson from `results_user_strategies_validation.md` (the submitted z-score
"WAVE" scripts): an apparent edge can be **pure survivorship bias**, and the way
to expose it is a **random-pick control drawn from the *same* eligible
point-in-time universe** — which carries the *identical* bias — plus
**delisting-aware accounting that never deletes losers.** ZENITH was run through
exactly that gauntlet:

* **Random-pick control (same eligible PIT universe, k=1, 40 draws):**
  ZENITH wins **95%** of windows vs QQQ; the random-pick DCA from the same
  universe wins **only 10%** (range 7–13%, *never above 13%*). ZENITH sits at the
  **100th percentile** of the random distribution in the median window.
  **ZENITH beats the survivorship control by +85 pp** — the edge is *selection
  skill, not survivorship.* (The z-score "WAVE" scripts, by contrast, *lost* to a
  random buy day at 3y/5y once PIT + delisting accounting were applied.)
* **Delisting-aware accounting:** the engine (`fast.py`) holds the full PIT panel
  (502 current + 218 delisted names) and liquidates a delisted holding at its
  last traded close — losers are never silently dropped (the exact bug that
  inflated the z-score scripts by up to +37 pp of 5-year return).
* **PIT membership mask:** a name is selectable only on dates it was an actual
  index member; no look-ahead into today's constituents.

---

## 3. Anti-leakage & engine integrity

* **Truncation leakage audit (`audit.py`):** rebuild the signal with the panel
  hard-truncated at six random dates; the signal row must be bit-identical to the
  full-sample build. **Result: max|Δ| = 0.00e+00, zero NaN-mismatch at all six
  dates → causal, no look-ahead.** (Same builder as SUMMIT, so this was expected;
  re-verified anyway.)
* **Reference vs fast engine cross-check:** the slow event-driven engine
  (`engine.py`) and the vectorized engine (`fast.py`) agree to 4 decimals on
  three windows (2007→2015: 2.5557 = 2.5557; 2012→2020: 3.1358 = 3.1358;
  2016→present: 9.8977 = 9.8977).

---

## 4. Robustness sweeps

* **Schedule-phase (offset 0–9):** win 93–95% at every offset, median +40.8% to
  +44.0%, worst −10.1% to −13.6%; full multiple **22.4–25.7×** across phases (vs
  SUMMIT 18.9–21.3×). Tight — *not* a schedule-timing mirage (the test that
  exposed the rebalance study).
* **Cost (5/10/20/40 bps):** median excess **+43.2% at every cost level** — never
  selling means each lot pays one half-spread once, so costs are nearly
  irrelevant (matches Novy-Marx-Velikov on why buy-and-hold sidesteps momentum's
  cost decay).
* **Cadence:** biweekly 95% win / monthly (every=21) 94% win, +42.7% median,
  worst −15%. Cadence-robust.
* **NASDAQ-100 PIT transfer (2015+, independent universe, own benchmarks &
  random control):** ZENITH beats QQQ-DCA at **all 15 half-year starts** *and*
  beats the random-pick control at **all 15** (e.g. 2016: 9.64× vs QQQ 3.27× vs
  random 3.57×; 2023: 2.83× vs 1.61× vs 1.47×). The edge transfers to a separate
  PIT universe and clears its own survivorship control.
* **IS/OOS:** in-sample (2006–14 starts) 92% win; out-of-sample (2015–23 starts)
  **99% win, worst only −1%.** No sign of overfitting — OOS is *stronger* than IS
  (the 2015+ mega-cap-tech era suits concentrated leadership).

---

## 5. Governance — Deflated Sharpe (honest trial accounting)

Per the cited review (Bailey-López de Prado), with dozens of configurations swept
across the APEX/ZENITH search, the headline metric should be deflated for the
trial budget. On the **biweekly active-return series vs QQQ** (T=564, annualized
active Sharpe 0.70, skew +0.34, kurtosis 5.4):

* **Deflated Sharpe Ratio ≈ 0.88** with a conservative **N = 30** trial count
  (and a conservative Var[SR]≈1/T proxy that *understates* DSR since the trials
  were highly correlated near-variants).

This is **strong but deliberately not a slam-dunk 0.95** — and the reason is
honest and structural: k=1 concentration produces a **fat-tailed biweekly active
return** (kurtosis 5.4), so the *short-horizon* Sharpe is modest. ZENITH's edge
lives in **long-horizon compounding of the right tail** (terminal multiple,
window win-rate, OOS 99%), not in biweekly Sharpe. The **random-pick control
(§2)** — random 10% vs ZENITH 95% — is the decisive, assumption-light governance
result; the DSR is reported as a conservative secondary check.

---

## 6. The honest costs of k=1 (stated plainly)

1. **Extreme live single-name concentration — the real risk the backtest
   understates.** The never-sell k=1 book becomes dominated by the era's biggest
   winner: end-of-period weights are **AAPL 67%, NVDA 19% (top-2 = 86%, top-3 =
   90%)**, across only 55 distinct names ever bought. The −11% worst *window*
   does **not** capture the idiosyncratic tail risk of holding two-thirds of the
   book in one stock (a sudden single-name fraud/collapse). This is a genuine,
   material risk, not a backtest artifact.
   * **Mitigation (risk lever, not a return lever): an optional annual
     single-name cap** that sells only the *excess* of any holding above a weight
     cap and redeploys it into the next leader (engine `trim_cap`). A **33% cap**
     holds win-rate at 94%, worst at −10%, multiple at **28.1×** (≈ uncapped, the
     difference is within phase noise), while cutting the top name from 67% → 33%.
     Any cap in 25–40% keeps win ~94–95% / worst ~−10–12%. Recommended deployable
     default: **k=1 with a 33% single-name cap.**
2. **One regime window lost.** ZENITH beats QQQ in 7 of 8 named regime windows
   (GFC +11%, recovery +8%, bull +47%, vol-2018 +5%, COVID +8%, bear-2022 +27%,
   AI-bull +69%) but **loses the flat 2015–16 sideways chop by −1%** (SUMMIT won
   all 8). In a trendless tape the single-name pick has no leader to ride — an
   honest, small, structural give-back, dwarfed by the gains everywhere a trend
   exists.
3. **Higher short-horizon volatility** (the DSR point) — by design.

---

## 7. Verdict

ZENITH **significantly and robustly outperforms DCA into QQQ** — 95% of 244
windows, +43% median, 25.7× vs QQQ's 9.1× from 2006, 99% OOS win — and does so as
a **validated** result, not an asserted one: causal (audit clean), engine-exact,
phase/cost/cadence-robust, IS/OOS-stable, transfers to an independent PIT
universe, and — decisively — **beats the random-pick survivorship control 95% vs
10%.** It is the most profitable point on the SUMMIT architecture's frontier.

The price of that profit is **concentration risk**, addressed (for risk, not
return) by the optional 33% single-name cap. ZENITH is the aggressive,
maximal-conviction sibling of SUMMIT: SUMMIT for those who want two-name
diversification; **ZENITH (capped) for those who want maximal profitable
right-tail capture** with the concentration kept sane.

Spec: `dca/ZENITH.md`. Code: `dca/strategy_zenith.py` (signal = SUMMIT's, k=1).
Search & frontier: `dca/research/apex_search.py`. Full machine report:
`dca/research/final/ZENITH_validation.json`, log
`dca/research/final/zenith_gauntlet.log`.

*Not investment advice. Past performance does not guarantee future results.*

---

## Post-hoc — the honest re-validation (why this is RETRACTED)

After this record was written, an independent validation playbook
(`VALIDATION_METHODOLOGY.md`) flagged that the gauntlet above — though it used
the *correct* benchmark (QQQ), a leakage audit, IS/OOS, and a random-pick
control — does **not** catch the two biggest distortions: **survivorship** and
**recency**. I re-ran its three load-bearing tests on this very panel; **the
critique holds.**

### 1. Survivorship — the panel itself is biased (the random control doesn't fix this)
* `summit_panel.parquet` contains **720 of 1,202** unique historical S&P-500 PIT
  members — **40% are missing entirely**, overwhelmingly the pre-2015
  delisted/bankrupt losers Yahoo never had.
* Of the 60 names that *do* delist inside the panel, they are booked at a median
  **0.67× of their trailing-1y high** (37% end >0.9×, acquisition-like; only 35%
  actually crashed <0.5×). True −100% bankruptcies are absent. So even the
  "delisting-aware" accounting is survivorship-**optimistic**.
* Why the random-pick control (95% vs 10%) does **not** rescue this: it shows
  ZENITH beats a random pick *drawn from the same survivor-tilted universe* — real
  *relative* selection — but both the strategy's multiple and the universe inherit
  the 40% missing-loser bias, which the control cannot see.

### 2. Out-of-sample IC ≈ 0 — no durable cross-sectional skill at beating QQQ
Rank-IC of the bull score vs 252-day-forward return **relative to QQQ**:
* **IS (pre-2018): −0.015 (t = −1.5)** — zero/negative.
* **OOS (2018+): +0.049 (t = +3.2)** — positive only in the mega-cap-momentum era.
A durable signal needs the *same-sign* IC in both eras; this flips. The score
does not durably predict which names beat QQQ — the wins come from concentration
into AAPL/NVDA in the recent regime.

### 3. Recency — most of the lead is 2018–2026
ZENITH ÷ QQQ money-multiple ratio (fixed 2006 start), by cutoff:
`1.29 (2010) → 1.23 (2014) → 1.50 (2017) → 1.71 (2019) → 2.08 (2021) → 2.23
(2023) → 2.80 (2026)`. A persistent *small* lead (1.2–1.5×, better than the
WAVE/ROTATOR replication, which was *behind* until 2021) — but more than half the
headline 2.8× was built in the AI run. A survivorship-clean rebuild
(`VALIDATION_METHODOLOGY.md`) collapses that residual lead to ≈ tie:
**25.7× → ~11.2×, QQQ ~9.2×**, drawdowns −58%/−74%.

### Verdict
**ZENITH is not a validated QQQ-beater and is not a Pareto improvement over
SUMMIT.** It is a concentrated mega-cap-momentum tilt whose apparent edge is
~half survivorship + a favorable 2018–26 regime; the size tilt is a small, real
lever, but not enough to durably beat a cap-weighted momentum benchmark
out-of-sample on clean data. Consistent with the standing conclusion in
`VALIDATION_METHODOLOGY.md`: **no durable cross-sectional stock-selection alpha
over QQQ-DCA survives out-of-sample with public price data.** What was right in
this file: the leakage audit, the engine cross-check, the cost/phase robustness,
and the *relative* random-control result. What was wrong: calling that a
validated, profitable QQQ-beater. (Reproduce the three tests above:
`python3 -c` against `data.build_panel()` / `protocol` / `fast` — script logic in
the commit that added this section.)*
