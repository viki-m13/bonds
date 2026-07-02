"""Shared calendar-sleeve research library. IS = 2010-03-11 .. 2018-12-31 ONLY.

Never prints post-2018 stats. Full-window returns kept only for saving CSVs.
"""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol

DATA = "/home/user/bonds/data/etfs"
IS_START = pd.Timestamp("2010-03-11")
IS_END = pd.Timestamp("2018-12-31")
SPLIT = pd.Timestamp("2014-12-31")          # sub-period boundary
SAVE_END = pd.Timestamp("2026-04-30")       # ignore raw tail post-2026-04
COSTS = {"TQQQ": 10.0, "QLD": 10.0, "UPRO": 10.0, "SSO": 10.0,
         "SOXL": 10.0, "TMF": 10.0, "QQQ": 5.0, "SPY": 5.0, "BIL": 2.0}

_TICKERS = ["SPY", "QQQ", "TQQQ", "QLD", "TMF", "BIL", "TLT"]


def load_panel(tickers=_TICKERS):
    opens, closes = {}, {}
    for t in tickers:
        df = pd.read_csv(f"{DATA}/{t}.csv", parse_dates=["Date"], index_col="Date")
        opens[t] = df["Open"]
        closes[t] = df["Close"]
    opens = pd.DataFrame(opens)
    closes = pd.DataFrame(closes)
    cal = opens["SPY"].dropna().index          # NYSE calendar = SPY dates
    cal = cal[(cal >= IS_START) & (cal <= SAVE_END)]
    return opens.reindex(cal), closes.reindex(cal), cal


def blank_W(cal, cols):
    return pd.DataFrame(0.0, index=cal, columns=cols)


def add_bil_residual(W):
    """Park unused gross in BIL (long-only, gross<=1)."""
    W = W.copy()
    risk = W.drop(columns=["BIL"], errors="ignore").sum(axis=1).clip(0, 1)
    W["BIL"] = 1.0 - risk
    return W


def run(W, opens, bil_residual=True):
    if bil_residual:
        W = add_bil_residual(W)
    res = backtest_weights(W, opens, cost_bps=COSTS)
    return res


def is_stats(res, name, phoenix=None):
    """Print IS metrics + sub-period split. Returns dict. Never touches post-2018."""
    r = res["ret"].loc[IS_START:IS_END]
    r1 = r.loc[:SPLIT]
    r2 = r.loc[SPLIT + pd.Timedelta(days=1):]
    to = res["turnover"].loc[IS_START:IS_END]
    yrs = len(r) / 252
    out = {
        "name": name,
        "SR": sharpe(r), "CAGR": cagr(r), "vol": ann_vol(r), "MDD": max_dd(r),
        "SR_10_14": sharpe(r1), "SR_15_18": sharpe(r2),
        "CAGR_10_14": cagr(r1), "CAGR_15_18": cagr(r2),
        "turnover_ann": float(to.sum() / yrs),
    }
    if phoenix is not None:
        both = pd.concat([r, phoenix], axis=1, join="inner").dropna()
        both = both.loc[:IS_END]
        out["corr_phx"] = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) > 100 else np.nan
    print(f"{name:38s} SR={out['SR']:5.2f}  CAGR={out['CAGR']*100:6.2f}%  "
          f"vol={out['vol']*100:5.1f}%  MDD={out['MDD']*100:6.1f}%  "
          f"| 10-14 SR={out['SR_10_14']:5.2f}  15-18 SR={out['SR_15_18']:5.2f}  "
          f"| TO/yr={out['turnover_ann']:5.1f}"
          + (f"  corr_phx={out.get('corr_phx', float('nan')):5.2f}" if phoenix is not None else ""))
    return out


def load_phoenix():
    p = pd.read_csv("/home/user/bonds/data/results/phoenix_production_returns.csv",
                    parse_dates=["Date"], index_col="Date")
    return p["raw_ret"].loc[:IS_END]           # IS guard


def save_candidate(res, short_name):
    r = res["ret"].loc[IS_START:SAVE_END]
    path = ("/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09"
            f"/scratchpad/candidates/{short_name}.csv")
    r.rename("ret").to_csv(path, index_label="Date")
    print(f"saved -> {path}  ({len(r)} rows, no post-2018 stats viewed)")


# ---------------- calendar feature helpers ----------------

def holiday_flags(cal):
    """pre1[d]=True if next session is >1 bday away (holiday follows d)."""
    d = pd.Series(cal, index=cal)
    nxt = d.shift(-1)
    gap = np.busday_count(d.dt.date.values.astype("datetime64[D]"),
                          pd.Series(nxt.dt.date.values).fillna(pd.Timestamp("2030-01-01").date()).values.astype("datetime64[D]"))
    pre1 = pd.Series(gap > 1, index=cal)
    pre1.iloc[-1] = False
    return pre1


def third_friday(year, month):
    d = pd.Timestamp(year, month, 1)
    fridays = pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2]


def week_of(cal, anchor_friday):
    """Sessions in the Mon..Fri calendar week containing anchor_friday."""
    mon = anchor_friday - pd.Timedelta(days=4)
    return cal[(cal >= mon) & (cal <= anchor_friday)]


def month_session_offsets(cal):
    """For each session: positive offset from month start (1=first) and
    negative offset from month end (-1=last)."""
    s = pd.Series(cal, index=cal)
    ym = s.dt.to_period("M").astype(str)
    fwd = s.groupby(ym.values).cumcount() + 1
    n_in_month = ym.map(ym.value_counts())
    bwd = fwd - n_in_month.values - 1          # -1 = last session of month
    return pd.Series(fwd.values, index=cal), pd.Series(bwd.values, index=cal)
