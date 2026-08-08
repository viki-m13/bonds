"""Vectorized daily cross-sectional backtester.

Weights contract: weights.loc[d] is the target portfolio decided from
information through the close of day d.

Execution modes:
  - "open":  trade at the open of d+1; the position earns open(d+1)->open(d+2)
             each day (open-to-open returns, shifted so no lookahead).
  - "close": trade at the close of d+1 (MOC); earns close(d+1)->close(d+2).

Costs: cost_bps applied to traded notional |w_new - w_old| per rebalance,
i.e. one-way per-side cost in basis points of NAV.

All series are simple returns on NAV with gross exposure = sum |w| (caller
controls leverage; Sharpe is leverage-invariant for the long-short part).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _exec_returns(panel, mode):
    if mode == "open":
        px = panel["open"]
    elif mode == "close":
        px = panel["close"]
    else:
        raise ValueError(mode)
    return px.pct_change(fill_method=None)


def run(weights, panel, mode="open", cost_bps=5.0):
    """See module docstring. mode="sameclose": MOC execution at the close of
    the signal day itself (order submitted into the closing auction from
    ~3:45pm data; assumes the last minutes don't flip the signal — an
    approximation that must be stress-tested against mode="open")."""
    if mode == "sameclose":
        r = panel["close"].pct_change(fill_method=None)
        w = weights.reindex(index=r.index, columns=r.columns).fillna(0.0)
        pos = w.shift(1).fillna(0.0)
        pnl = (pos * r).sum(axis=1)
        dw = w.diff().abs().sum(axis=1).fillna(0.0)
        cost = dw * cost_bps / 1e4
        return {"gross": pnl, "net": pnl - cost, "turnover": dw, "cost": cost}
    return _run_lagged(weights, panel, mode, cost_bps)


def _run_lagged(weights, panel, mode="open", cost_bps=5.0):
    """Backtest. Returns dict with daily net/gross returns and turnover.

    weights: DataFrame dates x tickers (target weights decided at close of d).
    For mode="open": position entered at open(d+1). Daily P&L attribution:
      ret(t) = sum_i w_i(t-2 decided) ... implemented via shift(2) on
      open-to-open returns o(t-1)->o(t): the weight decided at close d starts
      earning from open(d+1) to open(d+2), which is r_oo(d+2) = o(d+2)/o(d+1)-1,
      i.e. weights.shift(2) * r_oo. Cost charged when weights change.
    For mode="close": weights.shift(2) on close-to-close r_cc(d+2)?  No:
      entered at close(d+1), earns c(d+1)->c(d+2) = r_cc(d+2) -> shift(2).
    """
    r = _exec_returns(panel, mode)
    w = weights.reindex(index=r.index, columns=r.columns).fillna(0.0)
    pos = w.shift(2).fillna(0.0)
    pnl = (pos * r).sum(axis=1)
    # traded notional at execution day d+1 = |w(d) - w(d-1)| summed
    dw = w.diff().abs().sum(axis=1).shift(1).fillna(0.0)
    cost = dw * cost_bps / 1e4
    net = pnl - cost
    return {
        "gross": pnl,
        "net": net,
        "turnover": dw,
        "cost": cost,
    }


def sharpe(ret, ann=TRADING_DAYS):
    ret = ret.dropna()
    if len(ret) < 20 or ret.std() == 0:
        return np.nan
    return float(ret.mean() / ret.std() * np.sqrt(ann))


def metrics(ret, ann=TRADING_DAYS):
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(ret) / ann
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else np.nan
    downside = ret[ret < 0].std()
    return {
        "sharpe": round(sharpe(ret, ann), 3),
        "cagr": round(float(cagr), 4),
        "vol": round(float(ret.std() * np.sqrt(ann)), 4),
        "maxdd": round(float(dd), 4),
        "sortino": round(float(ret.mean() / downside * np.sqrt(ann)), 3) if downside and downside > 0 else np.nan,
        "days": int(len(ret)),
        "worst_day": round(float(ret.min()), 4),
    }


def yearly_sharpes(ret):
    return {str(y): round(sharpe(g), 2) for y, g in ret.dropna().groupby(ret.dropna().index.year) if len(g) > 60}


def is_oos(ret, split="2019-01-01"):
    r = ret.dropna()
    return {"IS": round(sharpe(r[r.index < split]), 3),
            "OOS": round(sharpe(r[r.index >= split]), 3)}


def norm_ls(raw, member, long_frac=0.1, short_frac=0.1, gross=2.0):
    """Standard long-short portfolio from a raw score frame.

    Ranks within members each day; longs top `long_frac` quantile, shorts
    bottom `short_frac`; equal weight, gross exposure = `gross` (1 per side
    if gross=2), dollar-neutral.
    """
    s = raw.where(member)
    rk = s.rank(axis=1, pct=True)
    n_l = (rk >= 1 - long_frac)
    n_s = (rk <= short_frac)
    wl = n_l.div(n_l.sum(axis=1), axis=0).fillna(0.0) * (gross / 2)
    ws = n_s.div(n_s.sum(axis=1), axis=0).fillna(0.0) * (gross / 2)
    return wl - ws


def norm_longonly(raw, member, frac=0.1):
    s = raw.where(member)
    rk = s.rank(axis=1, pct=True)
    n_l = (rk >= 1 - frac)
    return n_l.div(n_l.sum(axis=1), axis=0).fillna(0.0)
