# ZENITH — Maximal-Conviction DCA Stock-Selection Strategy

> **⚠️ RETRACTED as a "validated QQQ-beater" (2026-06-17).** The headline below
> (25.7×, 95% win, "Pareto improvement over SUMMIT") was validated on the same
> **survivorship-biased** panel as SUMMIT and does **not** survive honest
> re-testing. Independent re-validation
> ([`research/VALIDATION_METHODOLOGY.md`](research/VALIDATION_METHODOLOGY.md)) and
> my own diagnostics (recorded in `research/results_zenith.md` §"Post-hoc")
> found: the panel is missing **~40% of historical S&P 500 members** (the
> delisted losers), delisted names present are booked benignly (median 0.67× of
> their recent high, not −100%), the bull score has **~0 / negative
> cross-sectional IC vs beating QQQ in-sample** (IS t=−1.5; only +OOS in the
> 2018+ mega-cap regime), and most of the lead is **2018–2026 recency**. A
> survivorship-clean rebuild collapses ZENITH **25.7× → ~11.2×** (≈ ties QQQ),
> with −58%/−74% drawdowns — **not** a Pareto improvement and **not** a durable
> QQQ-beater. The leakage audit it passed does not exonerate it (leakage-clean ≠
> survivorship/recency-clean). Treat the numbers below as the *biased-panel*
> backtest, retained for the record only.

*The most-profitable point on the SUMMIT architecture's frontier — on a
survivorship-biased panel. Research session 2026-06-15; retracted 2026-06-17.
Code: `dca/strategy_zenith.py` (signal = `strategy_dca.build_scores`, run at k=1).
Honest re-validation: `dca/research/results_zenith.md` (§Post-hoc) and
`research/VALIDATION_METHODOLOGY.md`.*

---

## 1. What it is

Every two weeks, a fixed contribution buys **the single highest-scoring stock**
in the S&P 500, selected by SUMMIT's exact regime-switched rule. **Nothing is
ever sold.** ZENITH is SUMMIT at **maximal conviction (k=1)** — it feeds the #1
leader instead of splitting the contribution across the top two.

* **RISK-ON** (SPY ≥ 200dma and >40% of members above their own 200dma): buy the
  **#1** name by `rank(multi-horizon momentum) + 5 × rank(dollar volume)` — the
  largest stock that is also leading. Momentum = sum of return ranks over
  63/126/189/252 trading days, each skipping the most recent 21 days.
* **RISK-OFF** (SPY < 200dma OR breadth < 40%): keep buying — the **#1**
  `rank(discount) + rank(dollar volume)` **quality rebounder** (long-term uptrend
  intact, 30–60% below its all-time high). Buy the giant that's most on sale.
* Execution: signal at a close, buy at the **next open**. No sells, no stops.
* **Optional single-name cap (risk lever):** an annual trim of the *excess* of
  any holding above a weight cap (default **33%**), redeployed into the next
  leader. Caps live concentration without materially changing returns.

The signal is **identical to SUMMIT's** (same `build_scores`, same leakage audit,
same reference cross-check). ZENITH changes exactly one knob: **k = 1**.

## 2. Why maximal conviction (k=1)

Pre-registered from the cited literature (`research/literature_review_cited.md`),
not curve-fit:

1. **Return concentration (Bessembinder 2018):** ~all long-run net equity wealth
   comes from a tiny right tail (4.3% of US firms since 1926; top 0.33% make
   half), and concentration is rising. Against a cap-weighted benchmark, riding
   the single strongest leader and never trimming maximizes right-tail capture.
2. **Long-only never-sell (Sathish Kumar 2025; Patton-Weller 2017):** momentum's
   long leg is +7.9%/yr in large caps; the short leg and turnover do the damage.
   ZENITH keeps only the part that works.
3. **k-sweep is monotone** (95% → 82% win as k goes 1→5): k=1 is a smooth optimum,
   not an overfit spike.

## 3. Validation (PIT S&P 500, 244 windows, next-open, 5 bps, delisting-aware)

| | ZENITH (k=1) | SUMMIT (k=2) | QQQ-DCA |
|---|---|---|---|
| beat QQQ-DCA | **95%** | 93% | — |
| median excess vs QQQ | **+43%** | +29% | — |
| worst window vs QQQ | −11% | −11% | — |
| full multiple (2006→26) | **25.7×** | 20.0× | 9.1× |
| OOS (2015–23) win | **99%** (worst −1%) | 99% | — |

* **Decisive survivorship test:** ZENITH 95% vs the **random-pick control 10%**
  (same eligible PIT universe, same bias) — +85 pp; the edge is skill, not
  survivorship. (Same test the submitted z-score "WAVE" scripts *failed*.)
* Leakage audit clean (max|Δ|=0); reference engine == fast engine; phase-robust
  (22.4–25.7× across 10 offsets); cost-robust (+43% median at 5–40 bps);
  cadence-robust; **NASDAQ-100 PIT transfer beats QQQ *and* random at all 15
  starts.** Deflated Sharpe ≈ 0.88 (conservative, N=30) — see §5 of the record.

## 4. Honest costs

* **Concentration:** the uncapped live book is **AAPL 67% / NVDA 19%** — the real
  risk the −11% worst window understates. Use the **33% single-name cap** (win
  94%, worst −10%, multiple 28×) for deployment.
* **One regime lost:** beats QQQ in 7 of 8 regime windows but the flat 2015–16
  chop by −1% (no trend → no leader to ride).
* Higher short-horizon volatility, by design.

## 5. Relationship to SUMMIT

Same architecture, different point on the frontier. **SUMMIT (k=2)** for two-name
diversification and all-8-regime coverage; **ZENITH (k=1, 33% cap)** for maximal
profitable right-tail capture with concentration kept sane. Both never sell; both
share one signal builder and one validation harness.

*Not investment advice. Past performance does not guarantee future results.*
