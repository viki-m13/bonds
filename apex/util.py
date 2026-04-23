"""APEX — shared utilities.

Conventions:
  • All signals are computed using close prices through day t-1.
  • Weights are applied to the *next day's* return: ret[t] = Close(t)/Close(t-1) - 1.
    (This mirrors Phoenix: rebalance-at-open is approximated by the close-to-close
    return starting the day the signal activates. Open-fill slippage is charged
    separately as transaction cost.)
  • No lookahead: `weights.shift(1)` multiplies the return on day t.

Key functions:
  load_prices()           — wide Open/Close prices for universe
  ret_cc / ret_oo         — close-to-close or open-to-open returns
  metrics(r)              — SR, CAGR, MDD, vol, Calmar, Sortino
  daily_vol_target(r, v)  — scale a return series to target realized vol
  apply_weights(w, r)     — portfolio return from (T, N) weights * returns,
                             with tx-cost deduction
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
APEX = ROOT / "data/apex"

# -- Strategy parameters (locked once) --------------------------------------
IS_END = "2018-12-31"
OOS_START = "2019-01-02"

# Transaction cost (round-trip on daily target weight change)
TC_BPS_ETF = 3.0     # cheap unleveraged
TC_BPS_LETF = 8.0    # 3x LETF bid-ask + impact
DPY = 252

# Tickers by leverage
LETFS_3X = ["UPRO", "TQQQ", "TECL", "FAS", "SOXL", "DRN", "EDC", "YINN", "TMF", "TYD"]
LETFS_2X = ["SSO", "QLD", "ERX", "UBT", "UGL", "UCO"]
PLAIN = ["SPY", "QQQ", "TLT", "GLD", "BIL", "SHY"]


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (Open, Close) wide DataFrames."""
    df = pd.read_parquet(APEX / "prices.parquet")
    # multi-index columns: ("Open"/"Close", ticker)
    op = df["Open"].copy()
    cp = df["Close"].copy()
    return op, cp


def ret_cc(close: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close daily returns."""
    return close.pct_change()


def ret_oo(open_: pd.DataFrame) -> pd.DataFrame:
    """Open-to-open daily returns (for signals)."""
    return open_.pct_change()


def metrics(r: pd.Series, ann: int = DPY) -> dict:
    r = r.dropna().astype(float)
    if len(r) < 30:
        return {}
    mu = r.mean() * ann
    sd = r.std() * np.sqrt(ann)
    sr = mu / sd if sd > 0 else 0.0
    c = (1 + r).cumprod()
    dd = (c / c.cummax() - 1).min()
    yrs = len(r) / ann
    cagr = float(c.iloc[-1] ** (1 / yrs) - 1) if c.iloc[-1] > 0 else -1
    neg = r[r < 0]
    sortino = mu / (neg.std() * np.sqrt(ann)) if len(neg) > 1 and neg.std() > 0 else 0
    calmar = cagr / abs(dd) if dd < 0 else 0
    # Hit rate (daily)
    hit = (r > 0).mean()
    return {
        "sharpe":  round(float(sr), 4),
        "sortino": round(float(sortino), 4),
        "cagr":    round(float(cagr), 4),
        "vol":     round(float(sd), 4),
        "mdd":     round(float(dd), 4),
        "calmar":  round(float(calmar), 4),
        "hit":     round(float(hit), 4),
        "nav":     round(float(c.iloc[-1]), 3),
        "n":       int(len(r)),
    }


def daily_vol_target(ret: pd.Series, target_vol: float = 0.15,
                     win: int = 60, cap: float = 1.5, floor: float = 0.25) -> tuple[pd.Series, pd.Series]:
    """Scale a daily return series to target realized vol.

    mult[t] = clip(target_vol / rv_{t-1}, floor, cap).
    Applied with .shift(1) so no leakage.
    """
    rv = ret.rolling(win).std() * np.sqrt(DPY)
    mult = (target_vol / rv).clip(lower=floor, upper=cap).shift(1).fillna(1.0)
    return ret * mult, mult


def apply_weights(weights: pd.DataFrame, ret: pd.DataFrame,
                  tc_bps_map: dict[str, float] | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Compute daily portfolio return.

    weights: T x N daily target weights (sum <= 1 if desired). Applied with
             .shift(1) so weights decided at close[t-1] earn ret[t].
    ret: T x N daily returns.
    tc_bps: per-asset cost in bps per unit of daily weight change.

    Returns: (portfolio_ret, state_df) where state has [gross, tc_drag, net].
    """
    common = weights.columns.intersection(ret.columns)
    w = weights[common].fillna(0.0)
    r = ret[common].fillna(0.0)
    # Align to common index and carry forward weights on non-trading days
    w_eff = w.shift(1).fillna(0.0)
    gross = (w_eff * r).sum(axis=1)
    # Daily turnover (L1 change)
    dw = w.diff().abs().fillna(w.abs())
    if tc_bps_map is None:
        tc_bps_map = {c: 5.0 for c in common}
    tc_vec = pd.Series({c: tc_bps_map.get(c, 5.0) for c in common})
    tc_drag = (dw * tc_vec / 1e4).sum(axis=1)
    tc_drag = tc_drag.shift(1).fillna(0.0)
    net = gross - tc_drag
    state = pd.DataFrame({"gross": gross, "tc_drag": tc_drag, "net": net})
    return net, state


def tc_map() -> dict[str, float]:
    m = {}
    for t in LETFS_3X:
        m[t] = TC_BPS_LETF
    for t in LETFS_2X:
        m[t] = TC_BPS_LETF * 0.75
    for t in PLAIN:
        m[t] = TC_BPS_ETF
    return m


def regime_slice(r: pd.Series, start: str, end: str) -> pd.Series:
    return r.loc[(r.index >= start) & (r.index <= end)]


def summarize(r: pd.Series, label: str = "") -> None:
    m = metrics(r)
    print(f"{label:20s}  SR={m['sharpe']:>5.2f}  CAGR={m['cagr']*100:>5.1f}%  "
          f"Vol={m['vol']*100:>5.1f}%  MDD={m['mdd']*100:>6.1f}%  "
          f"Sortino={m['sortino']:>5.2f}  Calmar={m['calmar']:>4.2f}  "
          f"NAV={m['nav']:>7.2f}x  n={m['n']:>5d}")
