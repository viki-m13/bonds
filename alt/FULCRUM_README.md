# FULCRUM — an UNLEVERED VIX carry-&-hedge overlay that beats SPY

*An honest, IS/OOS-gated improvement on the Concretum "Volatility Edge" VIX-ETN strategy.*
**No leverage, no margin — fully invested, gross ≤ 100% at all times.**

Code: `alt/fulcrum_voledge.py` · validator: `alt/vix_voledge_validate.py` · chart: `fulcrum_equity_curve.png`
Data: SPY, VIX (FRED), VIX3M (CBOE), VIXY (+1x short-term VIX futures = VXX-equivalent), SVXY, 3M T-bills. 2011→2026.

## The diagnosis: the base strategy is 69% dead cash

The Concretum strategy is a *good signal* (harvests the variance risk premium, gated by
the VIX/VIX3M term structure and an eVRP realized-vol filter; standalone Sharpe ≈ 0.84).
But it sizes at "VIX%", so it deploys only ~31% of capital and **parks ~69% in cash.**
It's a cash-collateralized *overlay* run as if the collateral were the product. That dead
cash is why it trails SPY despite a fine Sharpe.

## The edge (unlevered): treat it as two sleeves and allocate by risk

Split a 100% budget between an **equity sleeve (SPY)** and the **vol sleeve** (the
unchanged regime-driven SVXY/VXX position). `w_spy + vol_notional = 1` → no leverage.

- **FULCRUM-U** — vol sleeve sized by the paper's VIX% rule, funded out of equity. Zero tuned params.
- **FULCRUM-RB (recommended)** — **risk-budgeted**: weight the two sleeves by inverse
  trailing volatility (risk parity), so each contributes equal risk and the short-vol
  sleeve auto-shrinks as it becomes dangerous. The IS grid chose `k_short=1` (pure
  inverse-vol, no fudge factor) and a 60-day window.

## Results (5 bps/side costs, unlevered)

| Period | Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| **OOS 2019-26** | SPY | 16.0% | 0.86 | 1.05 | −33.7% | 0.47 |
| | **FULCRUM-RB** | **21.9%** | **1.08** | **1.28** | **−23.3%** | **0.94** |
| **Full 2011-26** | SPY | 13.4% | 0.82 | 1.01 | −33.7% | 0.40 |
| | **FULCRUM-RB** | **19.4%** | **1.04** | **1.27** | **−23.3%** | **0.83** |

vs SPY this is **+6 pts of CAGR, Sharpe 1.04 vs 0.82 (+27%), and a drawdown cut by a
third** — out-of-sample as well as in-sample, **with no leverage.** The two FULCRUM
variants roughly **triple** SPY's terminal wealth over the period (see chart).

## What I tested and **rejected** (overfitting / honesty control)

Every "clever" enhancement, tuned IS and checked OOS, FAILED or was rejected by the grid:
amplified hedge (`k_h>1`), crash-throttle on the short leg, converting the cash regime to
a hedge, term-structure equity de-risking, a Moreira–Muir vol-managed wrapper, and brute
scaling of the vol sleeve (saturates ~Sharpe 0.98). **The robust edge is capital structure
+ risk allocation, not signal complexity.**

### A look-ahead bug I caught (documented on purpose)
An exploratory version shifted the whole sleeve-return series, pairing *tomorrow's* regime
signal with *today→tomorrow's* return. It printed a fake **Sharpe ≈ 2.0 / CAGR ≈ 90%.**
That is a bug, not alpha — I found it because the number was implausibly good, fixed the
alignment (signal at close_t → return close_t→close_{t+1}), and the honest Sharpe is
~1.0–1.08. If anyone shows you a backtested VIX strategy with Sharpe ≫ 1.5, suspect this.

## Honest risks (not a free lunch)

- **Short-vol-tilted, with fat left tails.** Its worst years are *worse* than SPY's:
  2018 (−14% vs −5%) and 2026-YTD (−9% vs −4%) — vol-explosion regimes where the overlay
  loses *and* equity falls. Risk-parity dampens but does not remove this.
- Edge concentrates in vol-normalization years (2012, 2020, 2021, 2023). A long calm
  stretch with no spike to mean-revert would compress it.
- Correlation to SPY ≈ 0.78 (it is a SPY-*plus* product, not a diversifier).
- VIXY/SVXY embed real roll decay and the post-2018 SVXY −0.5x leverage; pre-2018 exposure
  uses the index-equivalent method. Live MOC slippage / borrow and the notebook's
  operational bugs (see validation) are not fully modeled.

## Reproduce

```bash
pip install pandas numpy matplotlib
python alt/vix_voledge_validate.py   # validates the base strategy
python alt/fulcrum_voledge.py        # builds unlevered FULCRUM, prints tables, writes chart
```
