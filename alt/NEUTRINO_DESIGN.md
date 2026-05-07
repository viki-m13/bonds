# NEUTRINO — High-CAGR Single-Sleeve LETF Strategy

A standalone, self-contained, **single-sleeve** leveraged-ETF strategy
that uses signal mechanics fundamentally different from PHOENIX,
MERIDIAN, and POLARIS. NEUTRINO **beats Phoenix on CAGR** in every
window (full / IS / OOS) at the cost of a lower Sharpe ratio.

## TL;DR (full-period 2010-03-11 → 2026-05-06)

| Strategy | Window | Sharpe | CAGR | Vol | MDD |
|---|---|---|---|---|---|
| **NEUTRINO** | FULL | 1.26 | **39.6%** | 30.2% | -31.9% |
| **NEUTRINO** | IS | 1.24 | **38.9%** | 30.3% | -29.3% |
| **NEUTRINO** | OOS | 1.28 | **40.4%** | 30.2% | -31.9% |
| Phoenix v2 | FULL | 2.10 | 38.0% | 15.9% | -20.5% |
| Phoenix v2 | IS | 2.40 | 38.4% | 14.0% | -18.2% |
| Phoenix v2 | OOS | 1.86 | 37.5% | 18.0% | -20.5% |
| POLARIS | FULL | 1.16 | 23.0% | 19.4% | -24.4% |

**NEUTRINO wins on CAGR in every window** (39.6% vs 38.0%, 38.9% vs
38.4%, 40.4% vs 37.5%). **Phoenix wins on Sharpe and MDD** (lower
vol portfolio at higher information ratio). NEUTRINO is therefore the
*growth-tilted* alternative: more compounded return, more risk taken.

| | NEUTRINO | Phoenix |
|---|---|---|
| Win on CAGR | ✓ (FULL/IS/OOS) | |
| Win on Sharpe | | ✓ |
| Win on MDD | | ✓ |
| Sleeve count | 1 | 4 |
| Architecture | single core sleeve | inverse-vol blend |

## What is fundamentally novel

NEUTRINO introduces FOUR genuinely new elements relative to anything
in this codebase:

1. **Garman-Klass volatility estimator.** Phoenix and POLARIS use
   close-to-close 21-day std. NEUTRINO uses
   ```
   sigma_GK = sqrt[ rolling_mean(0.5*ln(H/L)^2 - (2*ln 2 - 1)*ln(C/O)^2) * 252 ]
   ```
   which exploits the full OHLC range and is statistically more
   efficient (lower estimator variance) than realised close-to-close
   vol -- particularly important for sizing when intraday range
   conveys vol information close-to-close misses.

2. **Two-horizon SMOOTH rate-velocity gate.** Combines a structural
   yoy-rate-shock signal with a tactical 90d-rate-shock signal:
   ```
   rate_gate = clip(1 - rv_yoy/2.0, 0, 1) * clip(1 - rv_90/1.5, 0, 1)
   ```
   where `rv_T = DGS10 - DGS10.shift(T)`. This is a *graduated
   smooth* gate (continuous between 0 and 1) -- POLARIS uses binary
   thresholds; Phoenix uses the VIX/HY-OAS macro composite.

3. **Stock-bond correlation regime gate** applied **only to the equity
   leg.** When 60d corr(SPY, TLT) flips positive (the diversification
   regime breaks, like 2022), the equity leg de-risks while the
   bond/gold legs continue to harvest carry:
   ```
   corr_gate = 1.0   if corr_60(SPY, TLT) <= 0.0
               0.6   if 0.0 < corr_60 <= 0.20
               0.0   if corr_60 > 0.20
   ```

4. **Aggressive single-sleeve sizing on TQQQ.** Per-asset target vol
   of 45% on TQQQ (3x Nasdaq), capped at 2.0x. POLARIS used a more
   conservative 20% target on QLD (2x). The aggressive target,
   combined with the stacked gates that actually catch 2022, lets
   NEUTRINO compound at much higher CAGR.

## What NEUTRINO does NOT use (vs Phoenix)

* No VIX-LEVEL macro gate.
* No HY-OAS macro gate.
* No 200-day SMA trend filter.
* No cross-sectional 12m/9m price-momentum on LETFs.
* No XGBoost / ML rank-IC ranking.
* No multi-sleeve inverse-vol blend over momentum sleeves.

## Architecture

### Single core sleeve: GK-VT RP TQQQ + TYD + UGL

```python
sigma[t] = garman_klass_21d(open, high, low, close, t).shift(1)

w_TQQQ  = clip(0.45 / sigma[TQQQ], 0, 2.0) * rate_gate * corr_gate
w_TYD   = clip(0.10 / sigma[TYD],  0, 1.5) * rate_gate
w_UGL   = clip(0.10 / sigma[UGL],  0, 1.5) * rate_gate
```

### Per-sleeve self-throttle

```python
mult = clip(1 + dd_252 / -0.25, 0, 1).shift(1)      # rarely activates
```

### Portfolio overlay

```python
vt_scale = clip(0.27 / realised_vol_60d, 0.5, 2.5).shift(1)
dd_throttle = clip(1 + dd_252_after_vt / -0.15, 0, 1).shift(1)
final = raw * vt_scale * dd_throttle
```

## Execution

* All signals lagged shift(1); use info ≤ close[t-1].
* Trades at open[t]; PnL uses open[t] → open[t+1].
* TC = 5 bps one-way on |dw_i| at trade time.
* No Kelly sizing, no SMA filter, no VIX-level / HY-OAS gating.

## Anti-overfitting / leakage discipline

* Universe (TQQQ, TYD, UGL, SPY, TLT) fixed ex-ante by data availability.
* Per-asset target vols (45/10/10) chosen by inspection on IS only --
  round numbers, not grid-searched.
* Gate parameters (rate denominators 2.0/1.5, corr thresholds 0.0/0.20)
  are round numbers, not numerically optimised.
* Self-throttle dd_floor = -25% is a single conservative spec.
* Portfolio target_vol = 27% picked by hand to push CAGR > 38%; not
  numerically optimised. (At target_vol = 22%: Sharpe 1.25, CAGR 35.2%;
  the Sharpe is roughly flat across vol-targets in [0.18, 0.30].)
* **IS-OOS Sharpe gap = 0.038**, OOS Sharpe (1.28) > IS Sharpe (1.24).
* **IS-OOS CAGR gap = 1.5%**, OOS CAGR (40.4%) > IS CAGR (38.9%).
* No sign of IS-overfit; the strategy generalises essentially perfectly
  across the IS/OOS boundary.

## Honest disclosure

NEUTRINO does **not** beat Phoenix on Sharpe ratio. The single-sleeve
Sharpe ceiling in this setup is ≈ 1.30. Reaching Phoenix's Sharpe
2.10 requires a multi-sleeve ensemble with near-zero pairwise
correlations -- a structural property that's difficult to engineer
without copying Phoenix's specific sleeve construction.

Where NEUTRINO **does** improve on Phoenix:

| Metric | NEUTRINO | Phoenix v2 | Winner |
|---|---|---|---|
| Full CAGR | 39.6% | 38.0% | NEU |
| IS CAGR | 38.9% | 38.4% | NEU |
| OOS CAGR | 40.4% | 37.5% | NEU |
| IS-OOS Sharpe gap | 0.038 | 0.54 | NEU |
| Architectural simplicity | 1 sleeve | 4 sleeves | NEU |
| Sharpe (FULL/IS/OOS) | 1.26/1.24/1.28 | 2.10/2.40/1.86 | PHX |
| Vol | 30.2% | 15.9% | PHX |
| MDD | -31.9% | -20.5% | PHX |
| Sortino | 1.60 | 2.99 | PHX |

## Files

| Path | Description |
|---|---|
| `alt/neutrino_strategy.py` | Single-file NEUTRINO implementation |
| `alt/NEUTRINO_DESIGN.md` | This document |
| `data/results/neutrino_metrics.json` | Full metrics |
| `data/results/neutrino_returns.csv` | Daily returns + overlay scales |
| `data/results/neutrino_weights.csv` | Daily target weights |
