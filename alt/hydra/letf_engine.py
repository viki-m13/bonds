"""Core backtest engine for standalone LETF strategy exploration.

Design goals (from the user requirement list):
  * Universe: leveraged ETFs + crypto (not hand-picked), with & without crypto.
  * No daily vol scaling — static or periodically rebalanced weights only.
  * Rebalance cadence 3-21 business days.
  * Next-day OPEN execution: signal uses data up through close of T-1;
    weights effective starting open of T; return captured is close-to-close
    of day T onward.  We implement this with a 1-bar weight lag.
  * 15 bps per unit turnover (one-way).
  * All strategies share one engine so comparisons are apples-to-apples.

Outputs a pd.Series of daily portfolio returns and a weight DataFrame for
inspection.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


TC_BPS_DEFAULT = 15.0
ROOT = Path("/home/user/bonds")


def load_universe(tickers, start=None, end=None):
    """Load close prices for a list of tickers; drop tickers with no data.
    Return a DataFrame aligned on the union index."""
    frames = {}
    for t in tickers:
        s = load_etf(t)
        if s is None or len(s) < 50:
            continue
        frames[t] = s
    px = pd.DataFrame(frames).sort_index()
    if start is not None:
        px = px.loc[start:]
    if end is not None:
        px = px.loc[:end]
    return px


def common_window_returns(px, require_all=True):
    """Return daily pct_change on the common-non-null window.
    If require_all=True, truncate to dates where every column has a value."""
    if require_all:
        px = px.dropna(how="any")
    rets = px.pct_change().fillna(0)
    return rets


def run_backtest(rets, weights_fn, rebal_days=21, tc_bps=TC_BPS_DEFAULT,
                 exec_lag=1):
    """Generic rebal-every-N-days backtester.

    Arguments:
      rets        : DataFrame[date x ticker] of daily pct returns
      weights_fn  : callable(date, history_rets) -> pd.Series or None
                    history_rets is rets.iloc[:i] (strictly pre-date data).
                    Return a Series indexed by ticker, or None to keep prior.
      rebal_days  : integer cadence
      tc_bps      : one-way transaction cost in bps on |turnover|
      exec_lag    : weights decided at close T, effective on day T+exec_lag.
                    Default 1 = next-day open execution.

    Returns (port_ret, weights_effective).
    """
    idx = rets.index
    cols = rets.columns
    n = len(idx)

    # Only iterate over rebalance days (the weight_fn is expensive) — then
    # broadcast last_target to all intervening days using a rebal-day table
    # and ffill.
    rebal_iloc = list(range(0, n, rebal_days))
    rebal_targets = {}   # iloc -> Series(len cols)
    last_target = pd.Series(0.0, index=cols)
    for i in rebal_iloc:
        hist = rets.iloc[:i]
        new_w = weights_fn(idx[i], hist)
        if new_w is not None:
            last_target = new_w.reindex(cols).fillna(0.0)
        rebal_targets[i] = last_target

    target_w = pd.DataFrame(np.nan, index=idx, columns=cols)
    for i, w in rebal_targets.items():
        target_w.iloc[i] = w.values
    target_w = target_w.ffill().fillna(0.0)

    # Apply execution lag: weights decided as of day T become effective on day T+exec_lag
    w_eff = target_w.shift(exec_lag).fillna(0.0)

    # Turnover = |ΔW_eff|, charged on the day the new weight activates
    tc = w_eff.diff().abs().sum(axis=1).fillna(0) * (tc_bps / 1e4)

    port_ret = (w_eff * rets).sum(axis=1) - tc
    return port_ret, w_eff


def summarise(r, label=""):
    """Compute CAGR, vol, MDD, Sharpe, CAGR/|MDD|, NAVx."""
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"label": label, "n": len(r), "cagr": 0.0, "vol": 0.0,
                "mdd": 0.0, "sharpe": 0.0, "cagr_mdd": 0.0, "navx": 1.0}
    nav = (1 + r).cumprod()
    cagr = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sr = r.mean() * 252 / vol if vol > 0 else 0
    mdd = (nav / nav.cummax() - 1).min()
    cagr_mdd = cagr / abs(mdd) if mdd < 0 else float("inf")
    return {"label": label, "n": len(r),
            "cagr": round(cagr * 100, 2),
            "vol": round(vol * 100, 2),
            "mdd": round(mdd * 100, 2),
            "sharpe": round(sr, 2),
            "cagr_mdd": round(cagr_mdd, 2),
            "navx": round(float(nav.iloc[-1]), 1)}


# --- weighting scheme factories ---

def w_fixed(weights_dict):
    """Static target weights (do not renormalise)."""
    w = pd.Series(weights_dict, dtype=float)
    def fn(d, hist):
        return w.copy()
    return fn


def w_inv_vol(tickers, lookback=63, sum_to=1.0):
    """Inverse-vol across a FIXED list of tickers. Weights sum to `sum_to`."""
    def fn(d, hist):
        if len(hist) < lookback + 5:
            return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0:
            return None
        vol = r.std()
        if (vol <= 0).all():
            return None
        inv = (1 / vol.replace(0, np.nan)).fillna(0)
        w = inv / inv.sum() * sum_to
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


def w_momentum_topn(tickers, lookback=126, top_n=3, sum_to=1.0):
    """Pick top N by compound return over lookback; equal weight among picks."""
    def fn(d, hist):
        if len(hist) < lookback + 5:
            return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0:
            return None
        cum = (1 + r).prod() - 1
        picks = cum.sort_values(ascending=False).head(top_n).index.tolist()
        out = pd.Series(0.0, index=hist.columns)
        if picks:
            out.loc[picks] = sum_to / len(picks)
        return out
    return fn


def w_risk_parity(tickers, lookback=63, vol_target=None):
    """Allocate 1/vol_i to each ticker. If vol_target given, scale total so the
    naive-independent portfolio vol ~= target (ignores covariance)."""
    def fn(d, hist):
        if len(hist) < lookback + 5:
            return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0:
            return None
        vol = r.std() * np.sqrt(252)
        if (vol <= 0).all():
            return None
        inv = 1 / vol.replace(0, np.nan).fillna(0)
        if vol_target is None:
            w = inv / inv.sum()
        else:
            w = inv * (vol_target / len(inv))
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


if __name__ == "__main__":
    # Smoke test: 60/40 UPRO/TMF monthly rebal, 2011-2026, next-day open exec
    px = load_universe(["UPRO", "TMF"], start="2011-01-01")
    rets = common_window_returns(px)
    r, W = run_backtest(rets, w_fixed({"UPRO": 0.6, "TMF": 0.4}),
                        rebal_days=21, exec_lag=1)
    s = summarise(r, "HFEA 60/40 @ 21d, exec_lag=1")
    print(s)
