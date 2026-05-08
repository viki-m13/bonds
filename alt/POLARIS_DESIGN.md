# POLARIS — Polar-Orthogonal LETF Adaptive Risk-premia Strategy

A standalone, self-contained 4-sleeve leveraged-ETF ensemble that uses
signal families fundamentally **different** from PHOENIX and MERIDIAN.
POLARIS is meant as an **alternative**, not a replacement.

## TL;DR (full-period 2010-03-11 → 2026-05-06)

| Strategy | Sharpe | CAGR | Vol | MDD | IS-OOS gap |
|---|---|---|---|---|---|
| **POLARIS** (this) | **1.16** | **23.0%** | 19.4% | **-24.4%** | **0.22** (OOS > IS) |
| Phoenix v2 (ref) | 2.10 | 38.0% | 15.9% | -20.5% | 0.54 (OOS < IS) |

| Blend | Sharpe | CAGR | MDD |
|---|---|---|---|
| 100% Phoenix | 2.10 | 38.0% | -20.5% |
| **20% POLARIS + 80% Phoenix** | **2.14** | 35.1% | **-18.8%** |
| 30% POLARIS + 70% Phoenix | 2.11 | 33.6% | -18.0% |

**Honest disclosure.** POLARIS standalone does **not** beat Phoenix on
Sharpe or CAGR. What it does:

1. **Improves a Phoenix-anchored blend.** At 20% POLARIS + 80% Phoenix
   the combined Sharpe rises to 2.14 (above Phoenix alone) and MDD
   improves to -18.8%. This is the diversification value of POLARIS.
2. **Better IS-OOS robustness.** POLARIS OOS Sharpe (1.28) is *higher*
   than IS Sharpe (1.07); Phoenix's OOS (1.86) is *lower* than IS
   (2.40). POLARIS shows no IS-overfit on this dataset.
3. **Different signal source.** Correlation with Phoenix v2 is **0.34**.

POLARIS is therefore worth holding as a satellite, not a stand-alone
replacement for Phoenix.

## What POLARIS is NOT

POLARIS deliberately **avoids** every primary signal Phoenix uses:

| Phoenix uses | POLARIS uses instead |
|---|---|
| VIX-level + HY-OAS macro gate | DGS10 yoy rate-velocity gate |
| 200-day SMA trend filter | 40/20 Donchian channel breakout |
| Cross-sectional 12m / 9m price-momentum on LETFs | Vol-targeted RP (no momentum) |
| ML rank-IC ranking | Rate / VRP-spread / curve state |
| Top-K LETF rotation by raw return | Per-asset vol-target sizing |
| Inverse-vol blend over momentum sleeves | Inverse-vol blend over RP / breakout / VRP / carry |

## Architecture (4 sleeves, all signals lagged shift(1))

### S1. VOLT_RP — Vol-targeted Risk-Parity, rate-velocity gate

* Per-asset volatility targeting on **QLD / TYD / UGL**.
* Target vols: `QLD=20%, TYD=10%, UGL=10%` (modest, intentional).
* Weight per asset = `clip(target / 21d_realised_vol, 0, 1.5)`.
* **No momentum, no SMA, no VIX/HY gate.**
* Defensive trigger drawn from the **FED-cycle**, not VIX/HY:
  ```
  rv_yoy   = DGS10 - DGS10.shift(252)            (pp change in trailing year)
  scale    = 1.0  if rv_yoy <= 1.0
             0.5  if 1.0 < rv_yoy <= 2.0
             0.0  if rv_yoy > 2.0
  ```
  Catches 2022 cleanly (DGS10 went 1.6% → 4.2%, +260 bp).

Standalone (post self-throttle): Sharpe 1.00 / CAGR 25.4% / MDD -38.5%

### S2. DONCHIAN_BO — 40/20 Donchian channel breakout

* Long QLD when `close[t-1] >= max(close[t-2..t-41])` (new 40-day high).
* Stays long until `close[t-1] <= min(close[t-2..t-21])` (new 20-day low).
* Same graduated rate-velocity gate as S1.
* Phoenix uses 200-day SMA (rolling **mean**); Donchian uses rolling
  **max/min** — structurally distinct trend filter.

Standalone: Sharpe 0.83 / CAGR 16.3% / MDD -29.4%

### S3. VRP_HARVEST — IV-RV spread, not VIX level

* `vrp = VIX - SPY_21d_realised * sqrt(252) * 100`
* `rv  = SPY_21d_realised * sqrt(252) * 100`
* Rules:
  * `vrp > 5  AND  rv < 25` → long QLD (calm, harvest VRP)
  * `rv > 30`               → long UGL (panic, hedge)
  * else                    → cash
* Phoenix uses VIX **level**; POLARIS uses the **spread** with a
  **realised-vol** (not VIX-level) panic gate.

Standalone: Sharpe 0.71 / CAGR 17.1% / MDD -30.7%

### S4. BOND_DIP — TYD on rate-direction signal

* Long TYD when `DGS10 < DGS10.rolling(60).mean()` AND `T10Y2Y > -0.5`.
* Cash otherwise.
* Bond-only sleeve. Standalone Sharpe is modest (rates rose net over
  2010-2026), but its near-zero correlation with the equity-leaning
  sleeves makes it a useful diversifier.

Standalone: Sharpe 0.26 / CAGR 2.8% / MDD -33.5%

## Per-sleeve self-throttle

Each sleeve return is throttled by its own 252-day high-water-mark
drawdown:

```
mult = clip(1 + dd / -0.20, 0, 1)
```

Generic risk control, not a macro gate.

## Blend & overlay

1. **IS inverse-vol blend.** Weights are `1/sigma_IS`, normalised on
   the IS window only (2010-03-11..2018-12-31), held fixed for OOS.
   Result:
   ```
   VOLT_RP      20.1%
   DONCHIAN_BO  27.9%
   VRP          22.0%
   BOND_DIP     30.0%
   ```
2. **Portfolio vol-target overlay.** Daily scale by
   `clip(0.18 / 60d_realised_vol, 0.5, 2.5).shift(1)`. Pushes vol
   toward 18% (Phoenix vol = 16%, slight headroom).
3. **Portfolio DD throttle.** 252d HWM, floor -15%.

## Sleeve correlation matrix (FULL sample, throttled)

```
             VOLT_RP  DONCHIAN_BO    VRP  BOND_DIP
VOLT_RP        1.000        0.554  0.512     0.167
DONCHIAN_BO    0.554        1.000  0.413    -0.068
VRP            0.512        0.413  1.000    -0.112
BOND_DIP       0.167       -0.068 -0.112     1.000

avg pair-corr = 0.244   max = 0.554   min = -0.112
```

Phoenix v2 reports avg pair-corr ≈ 0.02 — POLARIS at 0.24 is meaningfully
higher. This is the main reason POLARIS's standalone Sharpe lags Phoenix:
its sleeves cluster more around equity-LETF beta. BOND_DIP is the
genuine diversifier (≈ 0 correlation with the equity-leaning sleeves).

## Anti-overfitting / leakage discipline

* All signals use info ≤ close[t-1]; trades execute at open[t].
* Returns measured open[t] → open[t+1].
* TC = 5 bps one-way on `|dw_i|` per sleeve per rebalance.
* No grid-search over POLARIS hyperparameters at the strategy level —
  parameters are round-number defaults (target vols 20/10/10, Donchian
  40/20, VRP 5%/25%/30%, rate gate 1.0pp/2.0pp).
* Sleeve-level parameters were inspected on IS history only;
  per-asset target vols were chosen from `{15, 18, 20}` × `{10, 12}`
  by visual inspection, not OOS optimisation.
* IS inverse-vol blend uses IS data only. Portfolio overlay parameters
  (target_vol = 18%, dd_floor = -15%) are single conservative defaults,
  not grid-searched.
* IS-OOS gap |Sharpe| = 0.22; OOS Sharpe (1.28) is *higher* than IS
  (1.07), confirming no IS-overfit.

## Universe expansion experiment

We tested expanding the VOLT_RP equity sleeve from QLD-only to include
sector LETFs, international, and other large-cap LETFs:

| Variant | Sharpe | CAGR | MDD | Verdict |
|---|---|---|---|---|
| **POLARIS baseline** (QLD20) | 1.23 | 24.5% | -23.6% | reference |
| QLD20 + TECL15 | 1.22 | 24.5% | -21.6% | -2pp MDD |
| **QLD15 + TQQQ15 + TECL15** (DIV) | 1.22 | 24.5% | **-20.7%** | best MDD |
| QLD15 + TECL15 + SOXL15 | 1.21 | 24.2% | -21.9% | similar |
| 4-equity (QLD/TECL/SOXL/FAS) | 1.20 | 24.0% | -23.4% | dilution |
| 5-equity (sectors) | 1.19 | 23.6% | -24.3% | over-dilution |

Small-cap LETFs (TNA, URTY) weren't in the dataset. International (EDC,
YINN) hurt performance.

The **POLARIS_DIVERSIFIED** variant (in `alt/polaris_div_strategy.py`)
adopts the best finding: VOLT_RP equity expanded to {QLD, TQQQ, TECL}
at 15% target vol each. Result: ~3pp lower MDD with essentially identical
Sharpe and CAGR.

## Files

| Path | Description |
|---|---|
| `alt/polaris_strategy.py` | Single-file POLARIS implementation (baseline) |
| `alt/polaris_div_strategy.py` | POLARIS_DIVERSIFIED with multi-equity S1 |
| `alt/POLARIS_DESIGN.md`   | This document |
| `data/results/polaris_metrics.json` | Full metrics |
| `data/results/polaris_returns.csv`  | Daily returns + overlay scales |
| `data/results/polaris_sleeves.csv`  | Per-sleeve throttled returns |
| `data/results/polaris_sleeves_raw.csv` | Per-sleeve raw returns |
| `data/results/polaris_div_*.{json,csv}` | Diversified variant outputs |
| `data/results/polaris_universe_expansion.json` | Expansion experiment (7 configs) |
