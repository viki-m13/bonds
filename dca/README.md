# SUMMIT — Biweekly DCA Stock Selection: complete project record

This directory contains a from-scratch research program that designed,
validated, and deployed **SUMMIT**, a concentrated dollar-cost-averaging (DCA)
stock-selection strategy that beats DCA into QQQ and SPY across the large
majority of historical windows — plus a faithful replication of an external
strategy ("ROTATOR") for comparison, and a large battery of honest experiments
probing whether the strategy can be improved.

This README is the authoritative, complete record. Detailed numbers for each
study live in `research/results_*.md`; this document ties everything together.

---

## 0. TL;DR

* **SUMMIT** buys 2 S&P-500 stocks every two weeks (or monthly), holds forever,
  and switches what it buys by market regime. On point-in-time S&P 500 data,
  244 rolling windows, next-open execution, 5 bps cost:
  **beats QQQ-DCA in 93% of windows, SPY-DCA in 98%, median lead +28.8% vs QQQ,
  worst window −10.6%, all 8 regime windows positive, full-period 20.0× money
  multiple (24.7% IRR) vs QQQ 9.1× and SPY 4.7×.**
* The edge is **concentration into persistent mega-cap momentum winners**,
  bought while leading (bull regime) or while cheap (bear regime), never sold.
* We tested dozens of "improvements" drawn from quant literature and first
  principles. **Almost everything that dilutes the concentration hurts.** The
  strategy sits at a robust local optimum; only mild, optional tail buffers
  (each costing a little terminal return) move a needle.
* Live, self-updating factsheets: `docs/summit.html` and `docs/rotator.html`
  (https://viki-m13.github.io/bonds/summit.html). They refresh daily after
  market close via GitHub Actions.

---

## 1. The mission

Design a DCA stock-selection strategy that consistently and significantly
outperforms DCA into QQQ and SPY across *every* timeframe tested — capital
deployed biweekly (also monthly), buying 1–a-few concentrated names, holding
long-term, with optional regime exits. Strict anti-leakage, walk-forward / OOS
validation, no survivorship bias, realistic costs.

---

## 2. The final strategy: SUMMIT

Full spec and rationale: **`SUMMIT.md`**. Code: **`strategy_dca.py`**.

Every two weeks (or month), the contribution buys **2 stocks**, split 50/50, at
the next open. **Nothing is ever sold.**

* **Regime:** risk-off if SPX < 200-day MA *or* breadth (share of members above
  their own 200dma) < 40%; else risk-on.
* **Risk-on ("leaders"):** top-2 by `rank(multi-horizon momentum) + 5 ×
  rank(dollar-volume)`. Momentum = sum of return ranks over 63/126/189/252
  trading days, each skipping the most recent 21 days. The dollar-volume term is
  a strong **mega-cap tilt** — it is the decisive ingredient that closes the gap
  to cap-weighted QQQ.
* **Risk-off ("on sale"):** keep buying, but among names with a long-term
  uptrend intact (above 400dma *or* +24-month return) trading 30–60% below their
  all-time high, take the top-2 by `rank(discount) + rank(dollar-volume)`.
* **No sells, no stops, no recovery triggers** — all tested and rejected.

Why it works (validated, not asserted):
1. The benchmark is cap-weighted; an equal-weight picker lags it in mega-cap-led
   eras. The size tilt fixes that, then momentum adds selection alpha.
2. Momentum is regime-conditional (own EDA: 6m-momentum rank-IC +0.024 above the
   200dma, −0.054 below), so the rule switches *what* it buys, not *whether*.
3. Never selling compounds era-winners with zero turnover cost and full tax
   deferral; "add to existing winners" each period is the literal engine of the
   returns (see §6.8).

---

## 3. Infrastructure (the research engine)

| module | purpose |
|---|---|
| `data.py` | Builds aligned OHLCV + point-in-time membership panels. `build_panel()` = S&P 500 PIT (730 tickers, 2004–present); `build_panel_n100()` = NASDAQ-100 PIT (2015+). Includes bad-tick repair and ticker-recycling guards. |
| `download_pit_universe.py` | Downloads every historical S&P 500 constituent (fja05680 PIT membership) from Yahoo. |
| `engine.py` | Reference event-driven DCA backtester: next-open execution, delisting handling, costs, optional sell matrix, exposes current holdings. |
| `fast.py` | Vectorized DCA engine (~400× faster), verified to match the reference. Supports sell, single-name trim, sector cap / diversify, "new-only", and holdings return. |
| `protocol.py` | Standard evaluation: 244-window grid (quarterly starts × 3y/5y/10y/full horizons) + 8 regime windows vs QQQ/SPY DCA; random-pick survivorship control; by-horizon breakdown. |
| `regime.py` | Causal market-regime features (SPX 200dma, breadth, VIX percentile, HY-OAS). |
| `audit.py` | Truncation-based leakage audit: rebuild a signal with the future deleted; the row must be bit-identical. |
| `strategy_dca.py` | SUMMIT (the final strategy). |
| `strategy_rotator.py` | Faithful replication of the external ROTATOR strategy. |
| `build_factsheet.py` | Generates the live factsheet JSON (curves, returns, holdings, win-rates, regimes, cadence table, trim variants) for any strategy config. |
| `update_summit.py` | Daily cron entrypoint: refreshes prices, rebuilds both factsheets. |
| `validate_final.py` | Finalist gauntlet: leakage audit + reference cross-check + offset/cost/k/cadence sweeps + NASDAQ-100 transfer. |

### Validation governance (non-negotiable, applied throughout)
* **Anti-leakage:** signals use data only through the close before the execution
  open; verified by truncation audit (`audit.py`), zero difference.
* **Point-in-time universe:** real historical S&P 500 membership; no name
  selectable outside its membership window; delisting-aware accounting.
* **Survivorship control:** a random-pick DCA from the *same* eligible universe
  carries the identical bias; SUMMIT must (and does) beat it by a wide margin
  (93% win vs ~8% for random).
* **Robustness:** every finalist checked across 10 schedule phases, 5–40 bps
  costs, k=1/2/3, biweekly/monthly cadence, and an independent NASDAQ-100 PIT
  universe.
* **IS/OOS:** 2006–2014 starts vs 2015–2023 starts reported separately to catch
  overfitting.

### Data caveats (stated honestly everywhere)
Free Yahoo data lacks ~26% of historical constituents (mostly pre-2015
delistings); coverage rises 57% (2005) → 99% (today). Mitigated by PIT masks,
the random-pick control, and the mega-cap tilt (which concentrates picks where
coverage is ~complete). Two data bugs were found and fixed early — recycled
tickers and garbage price spikes on delisted names — which had inflated every
pre-fix backtest by up to +14pp win-rate.

---

## 4. SUMMIT validation results

(biweekly, k=2, 5 bps, vs same-cadence QQQ/SPY DCA; see `SUMMIT.md`)

| horizon | windows | beat QQQ | beat SPY | median vs QQQ | worst vs QQQ |
|---|---|---|---|---|---|
| 3 years | 70 | 84% | 91% | +10.0% | −10.6% |
| 5 years | 62 | 92% | 100% | +19.1% | −7.4% |
| 10 years | 42 | 100% | 100% | +31.6% | +8.4% |
| to present | 70 | 100% | 100% | +120% | +45.9% |
| **all 244** | | **93%** | **98%** | **+28.8%** | **−10.6%** |

All 8 regime windows beat QQQ (GFC +9%, recovery +7%, bull +27%, sideways +4%,
vol-2018 +1%, COVID +14%, bear-2022 +18%, AI-bull +49%). Full period 2006→2026:
$515k contributed → **$10.3M (20.0×, 24.7% IRR)** vs QQQ 9.1× and SPY 4.7×.
Leakage audit clean; reference and fast engines match exactly; NASDAQ-100 PIT
transfer beats QQQ at all 15 starts; IS 90% / OOS 99% win.

Honest weakness: 16/244 windows still lose to QQQ (worst −10.6%) — all are
3–5-year windows starting 2010–2013, the peak AAPL-concentration QQQ era. This
is structural (QQQ out-concentrated the strategy), not a risk event.

---

## 5. ROTATOR — external strategy replicated & compared

Code: `strategy_rotator.py`. Page: `docs/rotator.html`. Results:
`research/head2head.json`, `research/results_cadence*.md`.

Faithful rebuild of an external biweekly "leadership rotation" strategy
(leadership score = avg(3m,6m return) × rising-week fraction; top-3 hold with
rank-8 retention; SPX<210dma → cash). Run on the **identical** harness:

| metric | SUMMIT (k2) | ROTATOR (k3) |
|---|---|---|
| beat QQQ (244 windows) | **93%** | 65% |
| 10th-pct window vs QQQ | **+3.0%** | −23.4% |
| worst window vs QQQ | **−10.6%** | −57.5% |
| full-period multiple | 20.0× | **30.0×** |
| trades/year · taxes | ~0 · deferred | ~26 · short-term |

ROTATOR has the higher ceiling but far higher floor risk, and its edge is
fragile: the 30× depends on the biweekly cadence (full multiple swings
12.8–44.2× across schedule phases) and concentrates in the 2023-26 AI run. On
its **native** S&P+Nasdaq universe it's stronger (76-79% win, 34-51×) but the
tail stays bad. Verdict: SUMMIT = consistency/robustness; ROTATOR =
high-variance satellite.

---

## 6. The complete research log — what was explored and learned

Each item links to its detailed write-up. Listed roughly in the order explored.

### 6.1 Signal-family search (initial fan-out)
`research/results_momentum.md`, `results_ram.md`, `results_vcb.md`,
`results_volume.md`, `results_bear.md`, `eda_parabolic.md`, `literature_review.md`.
* **Momentum/trend** — the core selector. 9-1/12-1 with a skip-month best; 52w-high
  proximity, trend-smoothness, and momentum-acceleration as selectors all
  underperform. Plateaus ~60-66% win alone.
* **Risk-adjusted momentum (Sharpe, vol-scaled, low-vol, anti-lottery)** — all
  **hurt** vs QQQ (low-vol tilts are fatal because QQQ is high-beta growth).
* **Volatility-compression breakouts** — clean negative; compression is a
  negative overlay on momentum, monotone in the dose.
* **Volume/accumulation** — veto-grade at best; redundant after the size tilt.
* **EDA on parabolic precursors** — winners come from high-vol/high-beta names,
  not coiled springs; momentum is regime-conditional (the key architectural
  finding); deep-drawdown quality names lead *below* the 200dma.
* **Bear-regime behavior** — "quality rebounders at a discount" beats holding
  cash and every defensive sleeve; **all sell rules and recovery triggers hurt**
  (HY-OAS panic exit tripled the worst window).

### 6.2 The decisive ingredient — the mega-cap (size) tilt
Adding `+5 × rank(dollar-volume)` to momentum lifted window win-rate from ~60%
to 93% and closed the structural gap to cap-weighted QQQ. This, plus the regime
switch and bear sleeve, *is* SUMMIT. (`SUMMIT.md`, `research/SUMMARY.md`.)

### 6.3 Machine learning & foundation models — `results_ml.md`, `results_chronos.md`
* **Walk-forward LightGBM cross-sectional ranker** (20 trailing features): OOS
  rank-IC ≈ 0.002 (t=0.5); learns a defensive beta/vol tilt; **loses decisively
  to a single momentum column.**
* **Chronos-bolt time-series foundation model** re-ranking the momentum
  candidates: **worse than the matched momentum control** at every k. Both
  negative; excluded.

### 6.4 Cadence — `results_cadence.md`, `results_cadence_followup.md`
SUMMIT is **cadence-robust**: ~93-94% win, ~−11% worst, ~18-20× at daily /
weekly / biweekly / monthly. (ROTATOR, by contrast, is cadence-fragile.) Phase
sweep confirms SUMMIT's full multiple is tight (18.9–21.3×) regardless of which
day you start.

### 6.5 Universe — `results_universe.md`, `results_etf.md`
* **Broader large+mid-cap (Russell-1000-style, 1,005 names):** essentially no
  change (93% win, 20.1×) — the mega-cap tilt keeps SUMMIT in the biggest names;
  only 17/100 final positions are non-S&P, all tiny. Lowering the tilt to admit
  mid-caps raises the multiple slightly but doubles the drawdown and is
  survivorship-inflated.
* **ETFs (sector / all-types / leveraged + inverse):** SUMMIT **does not work** —
  beats QQQ in only 1-11% of windows. The size tilt makes it buy SPY/QQQ/GLD (the
  biggest "names" are the index funds), so it becomes a worse-diversified
  benchmark. Leveraged ETFs are never selected. SUMMIT is a stock strategy; for
  leverage the repo's PHOENIX/APEX are purpose-built.

### 6.6 Rebalancing vs never-sell — `results_rebalance.md`
Periodically selling everything and redeploying looks great on one path
(quarterly 130×) but is a **timing mirage**: the full multiple swings 15–130×
on the schedule phase, the robust win-rate/median barely change, worst-case
tails worsen, and it would trigger heavy short-term-gains tax the backtest
ignores. Never-sell wins on robustness and after-tax.

### 6.7 Trimming (optional concentration cap) — `results_trim.md`
Selling only the *excess* of a holding above a cap (vs full liquidation) is the
clean version: an **annual 25–33% single-name trim** caps the biggest position
(36% → 19–32%) while keeping win-rate 92-94%, median +25-30%, worst ~−11% — all
~unchanged — and stays phase-robust and tax-light. Shipped as an optional toggle
on the page.

### 6.8 Add-to-existing — `results_add_to_existing.md`
**The most decisive result.** Adding each contribution to the current top
names (even if already held) is the engine of the edge. Forcing every buy into
*new* names collapses the strategy: win-rate 93% → **7%**, median +28.8% →
−10.9%, multiple 20× → 4.9×, book balloons to 589 names. Momentum persistence
means "buy the current leaders" = "keep feeding the proven winners."

### 6.9 Sector caps — `results_improvements.md`, `sector_experiments.json`
Hard sector caps **hurt monotonically** (tech led the era, so capping it
sacrifices winners). A **loose 50% sector cap** is near-free (tech 77% → 62%, no
performance cost) — a risk-reduction lever, not a return booster. Forcing
distinct-sector picks barely diversifies (still 74% tech) and shaves the median.

### 6.10 Technical / risk-adjusted / Sharpe sweep — `results_improvements.md`
19 selection variants, IS/OOS-gated. Sharpe selection, vol-adjustment, the
fair-value mean-reversion z-score, the MAX filter, and RSI-pullback all **hurt**.
A "skip RSI≥80" gate and a mild trend-quality tilt were marginal/neutral. An
overfit trap was caught: `mom_12_1_only` looked best in-aggregate but was worse
out-of-sample — rejected.

### 6.11 Literature-grounded enhancements — `literature_enhancements.md`, `results_enhancements_tested.md`
Tested the six top-ranked ideas from the momentum-enhancement literature
(frog-in-the-pan, volatility-managed momentum, residual momentum, 52-week-high
gate, double-sort, conviction sizing) plus turn-of-month timing. Selection-level
ideas **hurt or were neutral**. The one defensible enhancement: a **targeted
Daniel-Moskowitz panic-defer gate** (SPX<200dma AND 20d-vol>80th pct AND 2y
return<0; 178 days in 22 years) takes the worst window −10.6% → −9.0% with
**identical OOS** and unchanged win-rate, at a cost of ~1.1× of terminal
multiple. An optional tail buffer, not a free improvement.

### Overarching lesson
From every direction, the same truth: **SUMMIT's returns come from concentrated
mega-cap momentum bought-and-held, and almost everything that dilutes that
either does nothing or trades return for diversification.** The design is at a
robust optimum; the only honest knobs left are optional, mild tail buffers
(panic gate, loose trims/caps), each with a small return trade-off.

---

## 7. The live site & automation

* `docs/summit.html` — the live SUMMIT factsheet: current picks + biweekly/monthly
  buy-date schedule, current holdings (with an optional concentration-cap toggle),
  growth chart (× multiple / $ value, selectable timeframes), returns-by-start-date,
  year-by-year, cadence table, win-rate, regime windows, rules, validation.
  Everything is data-driven from JSON the cron regenerates daily.
* `docs/rotator.html` — the ROTATOR factsheet, same format, with the head-to-head.
* **Automation** (`.github/workflows/`): `daily-update.yml` runs after market
  close, refreshes prices and rebuilds both factsheets via `update_summit.py`;
  `deploy-pages.yml` publishes `docs/` to GitHub Pages on push and after the cron
  (a guard bug that skipped push-deploys was fixed during this work).

---

## 8. File index

* **Strategy & spec:** `SUMMIT.md`, `strategy_dca.py`, `strategy_rotator.py`
* **Infrastructure:** `data.py`, `engine.py`, `fast.py`, `protocol.py`,
  `regime.py`, `audit.py`, `build_factsheet.py`, `update_summit.py`,
  `validate_final.py`, `download_pit_universe.py`, `RESEARCH_PROTOCOL.md`
* **Research write-ups (`research/`):** `SUMMARY.md` (running narrative),
  `literature_review.md`, `eda_parabolic.md`, `results_momentum.md`,
  `results_ram.md`, `results_vcb.md`, `results_volume.md`, `results_bear.md`,
  `results_ml.md`, `results_chronos.md`, `results_cadence.md`,
  `results_cadence_followup.md`, `results_universe.md`, `results_etf.md`,
  `results_rebalance.md`, `results_trim.md`, `results_add_to_existing.md`,
  `results_improvements.md`, `literature_enhancements.md`,
  `results_enhancements_tested.md`
* **Experiment scripts (`research/`):** `signals_*.py`, `cadence_study.py`,
  `cadence_phase_universe.py`, `rebalance_study.py`, `trim_study.py`,
  `universe_broad_study.py`, `etf_universe_study.py`, `sector_experiments.py`,
  `improve_experiments.py`, `improve_robustness.py`, `enhance_experiments.py`,
  `eda_parabolic.py`
* **Data (`../data/pit/`):** `sp500_pit_membership.csv`, `n100_pit_membership.csv`,
  `sectors.json`, coverage JSONs (price panels and downloaded price dirs are
  gitignored — reproducible from Yahoo).
* **Live pages (`../docs/`):** `summit.html`, `rotator.html`, and their `_data` /
  `_returns` / `_signal` JSON.

---

## 9. Reproduce

```bash
pip install -r ../requirements.txt        # + pyarrow scipy lightgbm
python download_pit_universe.py           # ~30 min, Yahoo (PIT S&P 500)
python -c "import data; data.build_panel(force=True)"
# headline validation:
python -c "import data,protocol,strategy_dca; P=data.build_panel(); \
  protocol.evaluate_signal(strategy_dca.build_scores(P),'SUMMIT',k=2)"
# full gauntlet:
python -c "import validate_final,strategy_dca; \
  validate_final.validate(strategy_dca.build_scores,'SUMMIT',k=2)"
# regenerate live factsheets:
python update_summit.py
```

---

*Not investment advice. Past performance does not guarantee future results.
All results carry the data caveats in §3.*
