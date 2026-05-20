"""Compute period returns (MTD / QTD / YTD / 1Y / 3Y / 5Y / 10Y / ITD) for both
the Phoenix backtest line and the live paper-trading account, plus SPY for
each window. Writes the result into phoenix_factsheet.json under the
`performance` key so the webapp can render it as `F.performance`.

Runs standalone, and is invoked from refresh_all.regenerate_factsheet so the
daily cron keeps the numbers current.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
R = ROOT / "data/results"
ETF = ROOT / "data/etfs"


def _open_to_open_returns(ticker: str) -> pd.Series:
    df = pd.read_csv(ETF / f"{ticker}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return pd.to_numeric(df["Open"], errors="coerce").pct_change().dropna()


def _cum_ret(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) == 0:
        return 0.0
    return float((1.0 + s).prod() - 1.0)


def _ann(total_ret: float, n_days: int) -> float:
    if n_days <= 0 or (1.0 + total_ret) <= 0:
        return 0.0
    yrs = n_days / 252.0
    if yrs < 1e-9:
        return 0.0
    return float((1.0 + total_ret) ** (1.0 / yrs) - 1.0)


# (label, kind, param) — kind: "cal" calendar-anchored MTD/QTD/YTD,
# "trail" trailing N trading days, "itd" inception-to-date.
WINDOWS = [
    ("MTD",  "cal",   "month"),
    ("QTD",  "cal",   "quarter"),
    ("YTD",  "cal",   "year"),
    ("1M",   "trail", 21),
    ("3M",   "trail", 63),
    ("6M",   "trail", 126),
    ("1Y",   "trail", 252),
    ("3Y",   "trail", 252 * 3),
    ("5Y",   "trail", 252 * 5),
    ("10Y",  "trail", 252 * 10),
    ("ITD",  "itd",   None),
]


def _window_mask(idx: pd.DatetimeIndex, kind: str, param) -> pd.Series:
    """Boolean mask selecting rows in [start, end] for the given window."""
    end = idx.max()
    if kind == "cal":
        if param == "month":
            start = pd.Timestamp(end.year, end.month, 1)
        elif param == "quarter":
            q_start_month = ((end.month - 1) // 3) * 3 + 1
            start = pd.Timestamp(end.year, q_start_month, 1)
        elif param == "year":
            start = pd.Timestamp(end.year, 1, 1)
        else:
            raise ValueError(param)
        return (idx >= start) & (idx <= end)
    if kind == "trail":
        # Take the last `param` rows of the series the mask is applied to.
        # We can't compute that from idx alone — return a length sentinel and
        # let caller slice via tail(param).
        return None  # handled separately
    if kind == "itd":
        return idx <= end
    raise ValueError(kind)


def _period_metrics(ret: pd.Series, bench: pd.Series, label: str, kind: str, param):
    """Compute Phoenix + benchmark cumulative (and annualized for >=1Y / ITD)."""
    if kind == "trail":
        ph = ret.tail(int(param))
        bh = bench.reindex(ph.index).fillna(0)
        if len(ph) < min(5, int(param)):  # not enough data
            return None
    elif kind in ("cal", "itd"):
        mask = _window_mask(ret.index, kind, param)
        ph = ret[mask]
        bh = bench.reindex(ph.index).fillna(0)
        if len(ph) == 0:
            return None
    else:
        return None

    p_tot = _cum_ret(ph)
    b_tot = _cum_ret(bh)
    out = {
        "label": label,
        "n_days": int(len(ph)),
        "start": ph.index[0].strftime("%Y-%m-%d"),
        "end": ph.index[-1].strftime("%Y-%m-%d"),
        "phoenix": round(p_tot, 6),
        "spy": round(b_tot, 6),
        "excess": round(p_tot - b_tot, 6),
    }
    # Annualize for windows >=1Y or ITD
    if label in ("1Y", "3Y", "5Y", "10Y", "ITD"):
        out["phoenix_ann"] = round(_ann(p_tot, len(ph)), 6)
        out["spy_ann"] = round(_ann(b_tot, len(ph)), 6)
    return out


def compute_backtest_performance() -> dict:
    """Period returns for the Phoenix backtest line vs SPY (open-to-open)."""
    prod = pd.read_csv(R / "phoenix_production_returns.csv",
                       parse_dates=["Date"]).set_index("Date").sort_index()
    ret = prod["net_ret"]
    spy_o2o = _open_to_open_returns("SPY").reindex(ret.index).fillna(0)

    periods = []
    for label, kind, param in WINDOWS:
        m = _period_metrics(ret, spy_o2o, label, kind, param)
        if m is not None:
            periods.append(m)
    return {
        "as_of": ret.index[-1].strftime("%Y-%m-%d"),
        "inception": ret.index[0].strftime("%Y-%m-%d"),
        "periods": periods,
    }


def compute_paper_performance() -> dict:
    """Period returns for the paper-trading account vs SPY (NAV-based)."""
    ps_path = R / "paper_summary.json"
    if not ps_path.exists():
        return {}
    ps = json.loads(ps_path.read_text())
    nav = ps.get("nav_series", [])
    spy = ps.get("spy_series", [])
    if len(nav) < 2 or len(spy) < 2:
        return {}

    nav_df = pd.DataFrame(nav).rename(columns={"d": "Date", "v": "nav"})
    nav_df["Date"] = pd.to_datetime(nav_df["Date"])
    nav_df = nav_df.set_index("Date").sort_index()
    spy_df = pd.DataFrame(spy).rename(columns={"d": "Date", "v": "spy"})
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.set_index("Date").sort_index()
    nav_s = nav_df["nav"]
    spy_s = spy_df["spy"].reindex(nav_s.index).ffill()

    p_ret = nav_s.pct_change().fillna(0)
    b_ret = spy_s.pct_change().fillna(0)

    periods = []
    # Only the windows that make sense given paper's short history
    for label, kind, param in [("MTD", "cal", "month"),
                                ("1M",  "trail", 21),
                                ("ITD", "itd",  None)]:
        m = _period_metrics(p_ret, b_ret, label, kind, param)
        if m is not None:
            periods.append(m)
    return {
        "as_of": str(nav_s.index[-1].date()),
        "inception": str(nav_s.index[0].date()),
        "current_nav": float(nav_s.iloc[-1]),
        "periods": periods,
    }


def compute_all() -> dict:
    """Return the combined performance dict to be embedded as F.performance."""
    return {
        "backtest": compute_backtest_performance(),
        "paper": compute_paper_performance(),
    }


def update_factsheet_json() -> dict:
    """Update phoenix_factsheet.json in-place with a fresh `performance` block.
    Returns the computed performance dict."""
    fpath = R / "phoenix_factsheet.json"
    perf = compute_all()
    if fpath.exists():
        data = json.loads(fpath.read_text())
        data["performance"] = perf
        fpath.write_text(json.dumps(data, separators=(",", ":")))
    return perf


if __name__ == "__main__":
    perf = update_factsheet_json()
    bt = perf["backtest"]
    print(f"Backtest as_of {bt['as_of']} (inception {bt['inception']}):")
    for p in bt["periods"]:
        extra = ""
        if "phoenix_ann" in p:
            extra = f" | ann {p['phoenix_ann']*100:+.1f}% vs SPY ann {p['spy_ann']*100:+.1f}%"
        print(f"  {p['label']:5s} ({p['n_days']:5d}d): Phoenix {p['phoenix']*100:+7.2f}%  SPY {p['spy']*100:+7.2f}%{extra}")
    pa = perf.get("paper", {})
    if pa:
        print(f"\nPaper as_of {pa['as_of']} (inception {pa['inception']}, NAV ${pa['current_nav']:,.2f}):")
        for p in pa["periods"]:
            print(f"  {p['label']:5s} ({p['n_days']:3d}d): Paper {p['phoenix']*100:+7.2f}%  SPY {p['spy']*100:+7.2f}%")
