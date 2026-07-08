"""Backtest engine for portfolios of individual Treasury CUSIPs.

Execution model (deliberately conservative, no look-ahead):
- Signals are computed from data up to and including rebalance date t.
- Trades execute at the close of t; the new weights earn returns from t+1.
- Between rebalances weights drift with each bond's total return.
- Transaction cost per trade = |dweight| * half-spread, where the half-spread
  is that CUSIP's FedInvest (buy-sell)/2 as a fraction of mid on trade date,
  with a per-side floor (cost_floor_bp). Missing spreads fall back to the
  cross-sectional median spread of that date's same-type securities.
- A held security that disappears from the panel (matures) is liquidated at
  its last observed price: its weight is dropped, earning zero that day.
  Treasuries redeem at par on maturity, so this is a small conservative bias.

All portfolio returns are per $1 of NAV. For long-short portfolios weights
sum to ~0 and the series is a self-financing excess return. For long-only
portfolios subtract the cash series to get excess returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BtResult:
    ret: pd.Series           # daily portfolio return (net)
    ret_gross: pd.Series     # before costs
    cost: pd.Series          # daily cost drag
    turnover: pd.Series      # one-sided turnover at each rebalance
    weights: pd.DataFrame    # rebalance-date target weights
    meta: dict = field(default_factory=dict)


def _half_spread(day: pd.DataFrame, cost_floor_bp: float) -> pd.Series:
    hs = day["spread_pct"] / 2.0 / 100.0
    med = hs.median()
    hs = hs.fillna(med if np.isfinite(med) else 0.0005)
    return np.maximum(hs, cost_floor_bp / 1e4)


def run(
    panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_floor_bp: float = 1.0,
    cost_mult: float = 1.0,
) -> BtResult:
    """panel: rows (date, cusip, ret, spread_pct). target_weights: rows
    (date, cusip, weight) at rebalance dates (close-of-day execution)."""
    dates = np.array(sorted(panel["date"].unique()))
    rets = panel.pivot_table(index="date", columns="cusip", values="ret", aggfunc="first")
    rets = rets.reindex(dates)

    # per-date half-spread lookup
    hs_w = panel.assign(hs=lambda d: d["spread_pct"] / 2.0 / 100.0).pivot_table(
        index="date", columns="cusip", values="hs", aggfunc="first"
    ).reindex(dates)
    med_by_date = hs_w.median(axis=1)

    tw = target_weights.pivot_table(index="date", columns="cusip", values="weight", aggfunc="first")
    reb_dates = tw.index

    port_ret, port_gross, port_cost, turns = [], [], [], []
    idx_out = []
    cur = pd.Series(dtype=float)  # current weights keyed by cusip

    for t in dates:
        r_t = rets.loc[t]
        if len(cur):
            held = cur.index
            r_held = r_t.reindex(held).fillna(0.0)
            gross = float((cur * r_held).sum())
            cur = cur * (1.0 + r_held)
            # drop matured/vanished (no future returns)
        else:
            gross = 0.0
        cost = 0.0
        if t in reb_dates:
            tgt = tw.loc[t].dropna()
            allw = pd.concat([cur, tgt], axis=1, keys=["cur", "tgt"]).fillna(0.0)
            dw = (allw["tgt"] - allw["cur"]).abs()
            hs = hs_w.loc[t].reindex(allw.index)
            hs = hs.fillna(med_by_date.loc[t]).fillna(0.0005)
            hs = np.maximum(hs, cost_floor_bp / 1e4) * cost_mult
            cost = float((dw * hs).sum())
            turns.append((t, float(dw.sum()) / 2.0))
            cur = tgt.copy()
        idx_out.append(t)
        port_gross.append(gross)
        port_cost.append(cost)
        port_ret.append(gross - cost)

    idx = pd.DatetimeIndex(idx_out)
    turnover = pd.Series(dict(turns), dtype=float)
    return BtResult(
        ret=pd.Series(port_ret, index=idx),
        ret_gross=pd.Series(port_gross, index=idx),
        cost=pd.Series(port_cost, index=idx),
        turnover=turnover,
        weights=tw,
    )


def cash_series(panel: pd.DataFrame) -> pd.Series:
    """Daily cash return proxy: median YTM of ~13-week bills, applied per day."""
    bills = panel[(panel["sec_type"] == "MARKET BASED BILL")
                  & panel["tsy_years"].between(0.2, 0.3)]
    y = bills.groupby("date")["ytm"].median()
    dates = y.index
    dt_frac = pd.Series(dates, index=dates).diff().dt.days.fillna(1) / 365.25
    return ((1 + y.shift(1)) ** dt_frac - 1).rename("cash")


def metrics(ret: pd.Series, cash: pd.Series | None = None, label: str = "") -> dict:
    r = ret.dropna()
    if cash is not None:
        r = (r - cash.reindex(r.index).fillna(0.0)).rename(r.name)
    ann = 252
    mu = r.mean() * ann
    sd = r.std() * np.sqrt(ann)
    nav = (1 + r).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    return {
        "label": label,
        "ann_ret": round(float(mu), 5),
        "cagr": round(float(cagr), 5),
        "ann_vol": round(float(sd), 5),
        "sharpe": round(float(mu / sd), 3) if sd > 0 else np.nan,
        "max_dd": round(float(dd), 4),
        "n_days": int(len(r)),
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
    }
