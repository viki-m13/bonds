# FULCRUM — a capital-efficient VIX carry-&-hedge overlay that beats SPY

*An honest, IS/OOS-gated improvement on the Concretum "Volatility Edge" VIX-ETN strategy.*

Code: `alt/fulcrum_voledge.py` (builds on the validation harness `alt/vix_voledge_validate.py`).
Data: SPY, VIX (FRED), VIX3M (CBOE), VIXY (+1x short-term VIX futures = VXX-equivalent),
SVXY, 3M T-bills. Window 2011-01 → 2026-04 (overlap of all series).

## The diagnosis: the base strategy is 69% dead cash

The Concretum strategy is a *good signal* — it harvests the variance risk premium,
gated by the VIX/VIX3M term structure and an eVRP realized-vol filter, and an
independent backtest gives it Sharpe ≈ 0.84. **But it sizes positions at only
"VIX%" of capital, so it deploys ~31% of the book on average and parks ~69% in
cash.** It is a cash-collateralized *overlay* being run as if the collateral were
the product. That dead cash is exactly why it trails SPY (CAGR 12.5% vs 14.6%)
despite a respectable Sharpe.

The vol sleeve also has beta ≈ 0.13 to SPY — so it is almost perfectly suited to
be *stacked on top of equity* rather than financed by cash.

## The edge: run it as portable alpha on an equity core

**FULCRUM-U (unlevered, recommended):** fund the vol sleeve out of the equity
sleeve — `w_spy = 1 − vol_notional`. Gross ≤ 100%, no margin, no financing.

**FULCRUM-L (levered / true portable alpha):** hold a full SPY core and stack the
vol overlay on top, financed at T-bills + 1.0%/yr.

The vol overlay itself is the **unchanged** Concretum regime engine (R1 short-full,
R2 short-half, R3 long-vol hedge). **No new tuned parameters in FULCRUM-U.**

## Results (5 bps/side costs; financing T-bills+1% for the levered variant)

| Period | Strategy | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| **Out-of-sample 2019-26** | SPY | 17.4% | 0.92 | −33.7% | 0.52 |
| | Base VolEdge | 15.6% | 0.84 | −29.7% | 0.53 |
| | **FULCRUM-U** | **23.4%** | **0.99** | **−22.8%** | **1.03** |
| | **FULCRUM-L** | **29.3%** | **1.05** | −30.3% | 0.97 |
| **Full 2011-26** | SPY | 14.6% | 0.88 | −33.7% | 0.43 |
| | **FULCRUM-U** | **19.6%** | **0.95** | **−22.8%** | **0.86** |
| | **FULCRUM-L** | **24.3%** | **1.00** | −30.3% | 0.80 |

FULCRUM-U beats SPY on **return, Sharpe, and drawdown simultaneously**, both
in-sample and out-of-sample, with zero tuned parameters. FULCRUM-L maximizes
absolute return at roughly SPY-level drawdown. (See `fulcrum_equity_curve.png`.)

## What I tested and **rejected** (overfitting control)

This is the part that matters. Every "clever" enhancement I invented was tuned on
in-sample 2011-2018 and then **failed** out-of-sample or was rejected by the IS
grid (which kept selecting the neutral setting):

| Enhancement | Verdict |
|---|---|
| Amplify long-vol hedge leg (`k_h > 1`) | Rejected — carry bleed > hedge benefit |
| Crash-throttle the short leg on VIX spikes | Rejected — no OOS improvement |
| Convert eVRP>0 & backwardation "cash" → hedge | Rejected — hurts |
| Term-structure equity de-risking (cut SPY when inverted) | Rejected — IS grid picks "no cut" |
| Moreira–Muir volatility-managed wrapper | Rejected — lagged scalar adds whipsaw, Sharpe 0.95→0.79 |

**Conclusion: the robust edge is capital structure, not signal complexity.** Adding
knobs only added overfitting risk. The honest improvement is a one-line idea —
stop holding 69% cash — executed with discipline.

## Honest risks (this is not a free lunch)

- **FULCRUM is short-vol-tilted and carries fat left-tail risk.** Its worst years
  are *worse* than SPY's: 2018 (−13% vs −5%) and 2026-YTD (−10% vs +8%) — both
  vol-explosion regimes where the overlay loses *and* equity falls.
- The aggregate edge concentrates in vol-normalization years (2012, 2020, 2021,
  2023). A multi-year stretch without a vol spike to mean-revert would compress it.
- Correlation to SPY is 0.66 (it is a SPY-plus product, not a diversifier).
- VIXY/SVXY price series embed real roll decay and the post-2018 SVXY −0.5x
  leverage; pre-2018 exposure is expressed via the index-equivalent method.
- Live frictions (MOC slippage, borrow/financing, the notebook's operational bugs
  flagged in the validation) are not fully modeled.

## Reproduce

```bash
pip install pandas numpy matplotlib
python alt/vix_voledge_validate.py   # validates the base strategy
python alt/fulcrum_voledge.py        # builds FULCRUM, prints tables, writes the chart
```
