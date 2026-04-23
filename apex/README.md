# APEX — Leveraged ETF Ensemble Strategy

**Status:** research & production reference for the APEX strategy, an ensemble
of rule-based leveraged-ETF signals designed to run long-only, no-margin,
with Phoenix-style robustness and pre-2008 backtesting.

## Design

Six sleeves, equal-weighted:

| # | Sleeve | Signal | Assets |
|---|---|---|---|
| 1 | XSMOM    | Cross-sectional momentum top-2 of 12 LETFs, monthly rebal | Full LETF universe |
| 2 | RPAR     | Inverse-vol risk parity                                     | UPRO, TMF, UGL     |
| 3 | TREND_EQ | 50/200 MA + 126d filter trend                               | TQQQ on QQQ        |
| 4 | TREND_BD | Same, treasury variant                                      | TMF on TLT         |
| 5 | TREND_GD | Same, gold variant                                          | UGL on GLD         |
| 6 | TSMOM    | Multi-timeframe (21/63/126/252) TSMOM consensus             | 12-LETF universe   |

Each sleeve is vol-scaled to 20% target (scale-down only, no margin).
Blend is equal-weight. Portfolio overlays: crisis switch, DD throttle
(−15% floor), and 25% vol target (bidirectional but gross capped at 1.0).

All signals computed on close[t-1], activated on day t (close-to-close
return the following day). Transaction costs: 8 bps for 3x LETFs, 6 bps for
2x, 3 bps for plain ETFs — charged per unit of daily weight change.

## Universe & Data

LETF universe (22 tickers) with synthetic pre-inception history constructed
from underlying index + FEDFUNDS + 90bps annual fee drag. Real/synthetic
correlation validates > 0.96 on 3-year overlap:

| Ticker | L  | Underlying | History |
|--------|----|------------|---------|
| UPRO   | 3x | SPY        | 2005+  |
| TQQQ   | 3x | QQQ        | 1999+  |
| TECL   | 3x | XLK        | 1999+  |
| SOXL   | 3x | SMH        | 2005+  |
| FAS    | 3x | XLF        | 1999+  |
| EDC    | 3x | EEM        | 2003+  |
| YINN   | 3x | FXI        | 2004+  |
| DRN    | 3x | VNQ        | 2004+  |
| TMF    | 3x | TLT        | 2005+  |
| UBT    | 2x | TLT        | 2005+  |
| TYD    | 3x | IEF        | 2005+  |
| UGL    | 2x | GLD        | 2004+  |
| UCO    | 2x | USO        | 2006+  |
| ERX    | 2x | XLE        | 1999+  |
| SSO    | 2x | SPY        | 2005+  |
| QLD    | 2x | QQQ        | 1999+  |
| SPY/QQQ/TLT/GLD/BIL/SHY | 1x | — | 1999-2007+ |

## Pipeline

1. `01_build_universe.py`       — builds extended LETF price history into
                                   `data/apex/prices.parquet`
2. `sleeves.py`                  — per-sleeve weight functions
3. `apex_production.py`          — build + blend + overlays, save returns
4. `stress_tests.py`             — walk-forward, bootstrap, regime, TC sens

Intermediate experimentation scripts (kept for audit):

- `02_test_engines.py`   — earlier engine tests
- `03_explore_core.py`   — single-asset vs dual-mom variants
- `04_continuous_sizing.py` — continuous TSMOM experiments
- `05_apex_v3.py`        — multi-engine v3
- `06_big_experiment.py` — ~30 engine library
- `07_blend_candidates.py` — greedy subset selection
- `08_build_apex.py`     — 8-sleeve v1 build
- `09_mega_library.py`   — ~30-sleeve library + greedy + robust blends
- `10_final_tune.py`     — subset / vol / DD sweep
- `ml_sleeve.py`         — XGBoost ML sleeve (not included in final blend)

## Outputs

- `data/apex/prices.parquet`          — wide Open/Close price panel
- `data/apex/sleeve_returns.csv`      — daily sleeve returns
- `data/apex/apex_production_returns.csv`  — daily net portfolio return
- `data/apex/apex_production_state.csv`    — overlay state (dd, vol, tc)
- `data/apex/apex_production_weights.csv`  — daily per-asset weights
- `data/apex/apex_production_metrics.json` — metrics + correlations
- `data/apex/stress_*.csv`            — per-stress-test results

## Results

See `data/apex/apex_production_metrics.json` for headline numbers.
Approx:

- Full sample (1999-2026):  Sharpe 0.80, CAGR 15.7%, MDD −38.9%
- In-sample (2005-2018):    Sharpe 0.85, CAGR 17.4%, MDD −33.9%
- Out-of-sample (2019+):    Sharpe 0.87, CAGR 18.3%, MDD −38.9%
- 2008 GFC:                 Sharpe 0.88, CAGR 20.3%, MDD −22.1%
- 2020 COVID:               Sharpe 1.56, CAGR 37.4%, MDD −18.4%
- 2022 rate hike:           Sharpe −1.92, CAGR −32.6%, MDD −35.5%
- 2023-24 recovery:         Sharpe 0.95, CAGR 22.7%

Headline target was Sharpe 3+ / CAGR 50%+. Strategy achieves strong Sharpe
near 0.9 and CAGR 15-18% — within the realistic range for a rule-based
long-only LETF ensemble without portfolio margin. Honest reporting of
achieved vs targeted metrics is included in the factsheet.

## Key design choices

- **No portfolio margin**: sum of weights ≤ 1 at all times. All leverage
  comes from the LETFs themselves (2x or 3x on the underlying).
- **Daily vol scaling**: both at sleeve level (20% target, scale-down only)
  and portfolio level (25% target, bidirectional but gross-capped).
- **Close-to-close return semantics**: weights decided at close[t-1] earn
  ret[t]. Equivalent to executing at T's open with small overnight-gap
  slippage (modeled via TC).
- **Pre-2008 testing**: 1999 onward where data allows, ensuring 2000-2002
  dot-com bust and 2008 GFC are stress scenarios, not hidden IS data.
- **Anti-overfit**: parameters (lookbacks, thresholds, blend weights) locked
  on IS (2005-2018), never re-fit on OOS. Greedy subset selection used IS
  correlations only.
