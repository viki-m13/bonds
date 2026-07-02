"""Shared sleeve backtest engine — single booking convention for all sleeves.

Convention (realization-dated, next-day-open execution):
    W[t]   = target weights DECIDED from data available at/before close[t-1]
             (every sleeve's signal layer carries its own >=1-bar lag),
             i.e. the weights to hold from open[t].
    o2o[t] = open[t] / open[t-1] - 1
    ret[t] = sum_i W[t-1, i] * o2o[t, i]  -  cost[t]
    cost[t]= sum_i |W[t, i] - W[t-1, i]| * tc_bps_i / 1e4   (trade at open[t])

So the return booked on date t is the P&L realized over open[t-1] -> open[t],
by a position entered at open[t-1] using information through close[t-2].
Every sleeve books on the same physical window for the same date label, which
makes cross-sleeve correlations, the blend NAV, and the portfolio overlays
contemporaneous and executable.

The last row of a sleeve returns CSV is therefore fully known at the close of
that date — no forward-looking placeholder rows, and refresh_all's
lookahead_days is 0 for every sleeve.

Cash convention: a 'BIL' (or other cash ticker) column, when present in W and
in the price panel, earns its own o2o return like any other asset. Implicit
residual cash (weights summing < 1 with no cash column) earns 0 —
conservative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_weights(
    W: pd.DataFrame,
    opens: pd.DataFrame,
    cost_bps: float | dict[str, float] = 5.0,
) -> pd.DataFrame:
    """Run the unified booking convention over a decision-dated weight frame.

    Args:
        W:        daily target weights, decision-dated (W[t] = weights to hold
                  from open[t], decided from data <= close[t-1]).
        opens:    open prices, same or superset of W's columns/index.
        cost_bps: one-way transaction cost; scalar applied to every ticker, or
                  a {ticker: bps} dict for per-ticker costs.

    Returns:
        DataFrame indexed like W with columns [ret, gross_ret, cost, turnover].
    """
    cols = [c for c in W.columns if c in opens.columns]
    W = W[cols].reindex(opens.index).fillna(0.0)
    o2o = opens[cols].pct_change(fill_method=None)

    w_prev = W.shift(1).fillna(0.0)
    # Positions held over open[t-1] -> open[t] are w_prev; skip NaN prices
    # conservatively as 0 contribution.
    gross = (w_prev * o2o).sum(axis=1)

    dw = (W - w_prev).abs()
    if isinstance(cost_bps, dict):
        rates = pd.Series({c: cost_bps.get(c, 5.0) for c in cols}) / 1e4
        cost = dw.mul(rates, axis=1).sum(axis=1)
    else:
        cost = dw.sum(axis=1) * (cost_bps / 1e4)

    net = gross - cost
    return pd.DataFrame(
        {"ret": net, "gross_ret": gross, "cost": cost,
         "turnover": dw.sum(axis=1)}
    )


def sharpe(r: pd.Series, freq: int = 252) -> float:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(freq))


def cagr(r: pd.Series, freq: int = 252) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    yrs = len(r) / freq
    eq = float((1 + r).prod())
    return float(eq ** (1 / yrs) - 1) if yrs > 0 and eq > 0 else float("nan")


def max_dd(r: pd.Series) -> float:
    eq = (1 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def ann_vol(r: pd.Series, freq: int = 252) -> float:
    return float(r.dropna().std() * np.sqrt(freq))


def perf_block(r: pd.Series, label: str = "") -> dict:
    p = f"{label}_" if label else ""
    return {
        f"{p}sharpe": sharpe(r),
        f"{p}cagr": cagr(r),
        f"{p}vol": ann_vol(r),
        f"{p}mdd": max_dd(r),
        f"{p}n": int(r.dropna().shape[0]),
    }
