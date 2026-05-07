# MERIDIAN — Three-Strategy Family

A family of three momentum strategies on the same 90-stock + 33-ETF universe,
differing in concentration, sleeve count, and whether leveraged ETFs are
allowed. Pick by your risk tolerance.

## Strategy comparison (2010-2026)

| Strategy | Sharpe | CAGR (raw) | CAGR (haircut) | MDD | Calmar |
|---|---|---|---|---|---|
| **COMPOSITE** | **1.28** | 26.7% | **24.6%** | -19.8% | 1.35 |
| **PURE** | 1.17 | 30.5% | **27.5%** | -29.6% | 1.03 |
| **LEV** | 1.10 | 33.9% | **31.8%** | -36.4% | 0.93 |
| Phoenix (ref) | 2.39 | 38.5% | — | -17.6% | 2.18 |

CAGR (haircut) applies a 3% survivorship-bias correction proportional to
stock-portion weight (30%, 100%, and 70% respectively).

## Hard constraints (all simultaneous, all strategies)

1. NO portfolio margin or borrowing — gross ≤ 1.0 every day.
2. NO forward-looking signals — close[t-1] only.
3. NO shorting, NO options.
4. ETF universe (where used) fixed ex-ante by liquidity + inception.
5. Stock universe disclosed as survivorship-biased; 3% CAGR haircut applied.

LEV variant additionally allows leveraged ETFs in the ETF rotation.

## Survivorship-bias accounting

The 90-stock universe is current S&P 500 large-caps with data back to 2010
in `data/stocks/`. Bankrupt/delisted/merged-out names (Lehman, WaMu, etc.)
are not in the dataset. Concentrated top-K amplifies the bias — we apply
a conservative **3% CAGR haircut** to the stock portion, blended at:

| Strategy | Stock weight | Blended haircut | Forward CAGR |
|---|---|---|---|
| COMPOSITE | 70% | 2.1% | 24.6% |
| PURE | 100% | 3.0% | 27.5% |
| LEV | 70% | 2.1% | 31.8% |

## COMPOSITE (alt/meridian_strategy.py)

5-sleeve diversified ensemble. Best Sharpe, smallest MDD.

| Sleeve | Universe | Top-K | Lookback | Rebal | Weight |
|---|---|---|---|---|---|
| S1 STOCK_3_W | 90 stocks | 3 | 126d | weekly | 23.3% |
| S2 STOCK_5_W | 90 stocks | 5 | 126d | weekly | 23.3% |
| S3 STOCK_7_M | 90 stocks | 7 | 252d | monthly | 23.3% |
| S4 ETF_FAST | 33 ETFs (1x) | 1 | 21d | daily | 15.0% |
| S5 ETF_SLOW | 33 ETFs (1x) | 1 | 126d | daily | 15.0% |

## PURE (alt/meridian_pure_strategy.py)

Single sleeve: top-3 stocks by 126d momentum, weekly Wed rebal. Simpler
than COMPOSITE; higher CAGR but lower Sharpe and deeper MDD due to
concentration.

## LEV (alt/meridian_lev_strategy.py)

3-sleeve standalone strategy with leveraged ETFs. Does NOT use Phoenix.

| Sleeve | Universe | Top-K | Lookback | Rebal | Weight |
|---|---|---|---|---|---|
| S1 STOCK_2_W | 90 stocks | 2 | 126d | weekly | 40% |
| S2 LETF_2_W | 17 LETFs | 2 | 126d | weekly | 30% |
| S3 STOCK_3_M | 90 stocks | 3 | 252d | monthly | 30% |

LETF universe: TQQQ, UPRO, SOXL, TECL, QLD, SSO, FAS, ERX, EDC, YINN, DRN
(3x equity); TMF, TYD, UBT (3x bonds); UGL, UCO (2x); NUGT (3x miners).

LEV's OOS CAGR (40.6%) **beats Phoenix's OOS CAGR (36.3%)**, though
Phoenix wins on Sharpe (2.39 vs 1.10). Phoenix's specific 5-sleeve
structure with mean correlation 0.02 is hard to beat without copying it.

## Risk overlays (de-risk only, all strategies)

- DD throttle: linear scale toward 0 below 252d HWM, floor at:
  - COMPOSITE: -15%
  - PURE: -25%
  - LEV: -25%
- Vol-regime gate: halve exposure when 60d realized vol > 99th pct.

## Files

| Path | Description |
|---|---|
| `alt/meridian_strategy.py` | COMPOSITE strategy |
| `alt/meridian_pure_strategy.py` | PURE strategy |
| `alt/meridian_lev_strategy.py` | LEV strategy |
| `alt/MERIDIAN_DESIGN.md` | This document |
| `data/results/meridian_metrics.json` | COMPOSITE metrics |
| `data/results/meridian_pure_metrics.json` | PURE metrics |
| `data/results/meridian_lev_metrics.json` | LEV metrics |
| `docs/meridian.html` | Editorial-style factsheet (with strategy toggle) |
| `docs/meridian_data.json` | Webpage data (auto-generated, all 3) |
