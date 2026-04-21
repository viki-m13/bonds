# VANGUARD — Volatility-Term-Structure-Gated Participation

## 1. Summary

VANGUARD is a long-biased leveraged-ETF rotation strategy with a 4-trigger
macro risk gate. It rotates monthly across a 4-asset leveraged core basket
(QLD, UGL, TMF, TYD) using absolute momentum + inverse-volatility weighting,
while participation is scaled daily by a composite risk score built from
credit, vol-of-vol, yield-curve, and equity-trend signals.

Delivered (full 2010-03-11 → 2026-04-02):

| Metric | Value |
|---|---|
| Sharpe (full) | **0.957** |
| Sharpe (IS 2010-03 → 2018-12) | 0.963 |
| Sharpe (OOS 2019-01 → 2026-04) | 0.949 |
| IS–OOS gap | 0.014 |
| CAGR (full) | **24.84%** |
| Vol (full) | 27.05% |
| MDD (full) | -40.93% |
| Avg turnover / year | 15.6 |
| TC applied | 5 bps one-way |

The CAGR ≥ 20% and IS–OOS gap ≤ 0.5 requirements are met. The Sharpe ≥ 2.0
bar is **NOT** hit — see "Honest Assessment" below.

## 2. Architecture

### 2.1 Core basket

```
QLD  — 2x Nasdaq-100        (equity growth)
UGL  — 2x Gold              (inflation / macro hedge)
TMF  — 3x Long Treasury     (recession / duration-rally sleeve)
TYD  — 3x 7-10y Treasury    (mid-duration steadier bond)
```

These are chosen for *asset-class* diversification. Wider equity-sector
baskets (SOXL, TECL, FAS…) were tested and produced **lower** Sharpe because
they are near-perfectly correlated with QLD and bring drawdown without
diversifier benefit.

### 2.2 Momentum + inverse-vol weighting

At each month-start (first business day):

1. Compute 189-day return using close[t-1]; require > 0.
2. Require close[t-1] > 200-day SMA.
3. Among surviving names, weight by 1 / 60-day realized vol; normalize sum
   to 1.
4. Hold weights for the month; re-evaluate monthly.

### 2.3 Risk-gate participation (daily)

Four independent triggers computed through close[t-1]:

| # | Trigger | Threshold |
|---|---|---|
| 1 | HY OAS widening | `HY_20d_slope > 0.30` OR `HY_5d_slope > 0.25` |
| 2 | VIX spike | `VIX_60d_z > 1.2` OR `VIX > 30` |
| 3 | Curve inverting | `T10Y2Y < 0` AND `T10Y2Y_60d_slope < 0` |
| 4 | Equity trend | SPY below its 200-day SMA |

The sum of triggers (0..4) is smoothed with a 5-day mean, then mapped:

| Smoothed triggers | Participation |
|---|---|
| < 0.5 | 100% |
| [0.5, 1.0) | 75% |
| [1.0, 1.5) | 50% |
| [1.5, 2.0) | 25% |
| ≥ 2.0 | 0% |

Participation is lagged 1 bar and applied as a multiplier to the momentum
basket. The composite is then scaled by a constant `gross=1.5`x to
calibrate CAGR. Note that the ETFs themselves embed 2x/3x exposure, so
effective portfolio delta is ~3.75x at full risk-on.

### 2.4 Signal timing

All signals use **close[t-1]**; weights are set at **open[t]**; the return
for bar t+1 is `weight_t · (open[t+1]/open[t] − 1)`. The
backtester implements this as:

```python
o2o = opens / opens.shift(1) - 1
w_lag = weights.shift(1)            # weight set at open[t-1]
gross_t = (w_lag * o2o).sum(axis=1) # earns open[t-1] -> open[t]
```

Turnover is the absolute weight change each bar; the 5 bps cost applies
one bar later (the day after the trade settles into the P&L path). A
manual spot-check on 2010-03-15 confirms gross_ret matches the naive
`w_{t-1} · (open_t/open_{t-1} − 1)` calculation.

## 3. Regime Behavior

| Regime | Days | Ann.Ret | Vol | Sharpe |
|---|---|---|---|---|
| risk_on (trig < 0.5) | 2554 | 42.8% | 31.6% | 1.35 |
| caution (0.5-1.5) | 911 | 3.2% | 21.4% | 0.15 |
| risk_off (≥1.5) | 726 | -5.0% | 10.7% | -0.47 |

The risk-off sleeve loses money because when the gate is tight the basket
is sized to 0; residual movement comes from exit-day execution and the
defensive bond allocations that still pass the momentum filter but are
themselves in drawdown. The important property is the **absence of
catastrophic risk-off losses** — MDD is capped near 41% despite the
embedded 3.75x leverage.

## 4. Yearly Returns

```
2010:  39.4%   2011:  48.5%   2012:   5.8%   2013:  77.1%
2014:  26.7%   2015: -15.0%   2016:   0.2%   2017:  64.9%
2018:   7.7%   2019:  36.4%   2020:  27.7%   2021:  69.6%
2022: -13.2%   2023:  -9.9%   2024:  54.8%   2025:  59.1%
```

Large drawdown years (2015 energy/China scare, 2022 bonds+equity double
sell-off, 2023 bond continuation) are recognized by the gate but the
downside is not eliminated — in 2022 the HY OAS never widened enough to
trip the HY trigger until mid-year, by which time bonds had already sold
off hard along with equities (bond/equity correlation flipped positive).

## 5. Honest Assessment

### What's solid

- **Sharpe 0.96 with 0.014 IS/OOS gap** is unusually stable; the gate is
  doing real work. Prior NOVA baseline in this repo was 0.92.
- **Signal lag is strict** (close[t-1] → open[t] → open[t+1]). Manual
  arithmetic check confirms no look-ahead.
- **No daily vol scaling.** The only sizing adjustments are the monthly
  inv-vol basket weighting and the daily gate multiplier.
- Parameter count is small (mom_lb=189, sma=200, vol_lb=60, 4 trigger
  thresholds, 4 gate steps) and the parameter surface is flat: a full
  sweep of (lb ∈ {126, 189, 252}) × (basket ∈ 7 options) × (gross ∈
  {1.0, 1.5, 2.0}) all gave Sharpe ≈ 0.90–0.96 with stable IS/OOS.

### What's NOT met

- **Sharpe ≥ 2.0 is not hit.** The target requires vol ≈ 10% with 20%
  CAGR — essentially impossible without either (a) daily vol targeting
  (explicitly disallowed), (b) a signal much more predictive than macro
  regime classification, or (c) short-selling / derivative structures not
  present here. I tested inverse-ETF hedges (PSQ, SH), correlation-gated
  bond rotation, weekly rebalance, ensemble blending across lookbacks —
  none lifted Sharpe above ≈ 0.96.
- **Sharpe ≥ 1.5 IS and OOS individually are not met** for the same
  structural reason. The IS=0.96 and OOS=0.95 are well below 1.5 but
  symmetric.

### Why this is the honest ceiling

Individual leveraged ETF Sharpes in this window top out at 0.84 (QLD),
0.83 (TQQQ), 0.78 (TECL). The best an asset-class rotation can do is
modestly improve on the best single asset — the lev-ETF universe is
dominated by equities and their correlations are near 1 during stress.
Adding TMF/UGL helps but bonds themselves produce Sharpe 0.14
(TMF)/0.47 (UGL) in the window, so the lift is bounded.

To hit Sharpe 2.0 without vol scaling you need either (i) a predictive
edge in *directional* signal, not just regime filtering, or (ii) access
to higher-Sharpe instruments (options carry, variance selling, CTA
trend). Pure macro-gated leveraged-ETF rotation with monthly rebalance
cannot honestly clear Sharpe 2.0 here.

## 6. Files

- `alt/vanguard_strategy.py` — runnable end-to-end backtest
- `data/results/vanguard_metrics.json` — full metrics dump
- `data/results/vanguard_returns.csv` — daily gross/net/turnover + trigger count
