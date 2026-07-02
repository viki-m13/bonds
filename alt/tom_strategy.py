"""TOM — turn-of-month equity seasonality, expressed via TQQQ.

Rule (designed on IS 2010-2018 only; see PHOENIX_V3.md):
  - Long TQQQ during the last 4 and first 3 trading days of each calendar
    month; cash (0%) otherwise. No other conditions (an SPY trend filter
    was tested on IS and REJECTED — it destroyed the effect).
  - 10 bps/side.

The signal is purely calendar-based: the NYSE session calendar is known in
advance, so W[t] is decidable at close[t-1] by construction.

Booking: unified realization-dated convention (alt/sleeve_engine.py).

Outputs:
  data/results/tom_returns.csv  (Date, ret)
  data/results/tom_metrics.json
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF = ROOT / "data/etfs"
R = ROOT / "data/results"

VEHICLE = "TQQQ"
N_BEFORE = 4   # last N trading days of the month
N_AFTER = 3    # first N trading days of the month
COST_BPS = 10.0
START_DATE = pd.Timestamp("2010-03-11")
IS_END = "2018-12-31"


def load_etf(t):
    df = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).set_index("Date")
    return df[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")


def _nyse_calendar():
    """Approximate NYSE holiday calendar, used ONLY to project the remaining
    sessions of the in-progress month (complete months use observed dates).

    NYSE quirk: New Year's Day falling on a Saturday is NOT observed on the
    preceding Friday (Dec 31 stays a session), so sunday_to_monday — not
    nearest_workday — is the correct observance there; getting this wrong
    would misclassify Dec 31 inside the last-4 window."""
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar, Holiday, nearest_workday, sunday_to_monday,
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        USLaborDay, USThanksgivingDay)

    class _NYSE(AbstractHolidayCalendar):
        rules = [
            Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
            USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
            Holiday("Juneteenth", month=6, day=19,
                    start_date=pd.Timestamp("2022-06-19"),
                    observance=nearest_workday),
            Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
            USLaborDay, USThanksgivingDay,
            Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        ]

    return _NYSE()


def _project_month_sessions(observed: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Full expected session list for the month of observed[-1]: the sessions
    already observed plus projected business days (minus NYSE holidays)
    through month-end. Needed so 'last N sessions of the month' is evaluated
    on the true month window even while the month is in progress."""
    last = observed[-1]
    month_end = last + pd.offsets.MonthEnd(0)
    if last >= month_end:
        return observed
    future = pd.bdate_range(last + pd.Timedelta(days=1), month_end)
    hols = _nyse_calendar().holidays(start=future[0], end=future[-1]) if len(future) else []
    future = future.difference(pd.DatetimeIndex(hols))
    return observed.append(future)


def build_weights(live_extend: bool = False) -> pd.DataFrame:
    """Decision-dated daily target weights ({VEHICLE} column).

    The month windows are evaluated on the full month's session calendar:
    observed sessions for complete months, observed + projected (deterministic
    NYSE calendar) for the in-progress month. Without the projection, the
    'last N sessions' rule degenerates to 'last N AVAILABLE sessions', which
    marks every data tail — and every live_extend appended day — as
    turn-of-month, i.e. permanently long.
    """
    dates = load_etf("SPY")["Open"].dropna().index
    if live_extend and len(dates) > 0:
        nxt = dates[-1] + pd.tseries.offsets.CustomBusinessDay(calendar=_nyse_calendar())
        dates = dates.append(pd.DatetimeIndex([nxt]))
    ym = dates.to_period("M")
    pos = pd.Series(0.0, index=dates)
    for i, p in enumerate(ym.unique()):
        m = dates[ym == p]
        if i == len(ym.unique()) - 1:
            full = _project_month_sessions(m)
            long_days = full[-N_BEFORE:].union(full[:N_AFTER])
            pos.loc[m.intersection(long_days)] = 1.0
        else:
            pos.loc[m[-N_BEFORE:]] = 1.0
            pos.loc[m[:N_AFTER]] = 1.0
    # Residual cash held in BIL (live holds BIL, not 0%-yield cash)
    W = pd.DataFrame({VEHICLE: pos, "BIL": 1.0 - pos})
    return W.loc[START_DATE:]


def main():
    from sleeve_engine import backtest_weights, perf_block

    W = build_weights()
    opens = pd.DataFrame({VEHICLE: load_etf(VEHICLE)["Open"],
                          "BIL": load_etf("BIL")["Open"]}).reindex(W.index).ffill(limit=3)
    bt = backtest_weights(W, opens, {VEHICLE: COST_BPS, "BIL": 2.0})
    ret = bt["ret"]

    m = {"full": perf_block(ret), "is": perf_block(ret.loc[:IS_END]),
         "oos": perf_block(ret.loc["2019-01-02":]),
         "params": {"vehicle": VEHICLE, "n_before": N_BEFORE, "n_after": N_AFTER,
                    "cost_bps": COST_BPS}}
    out = ret.to_frame("ret")
    out.index.name = "Date"
    out.to_csv(R / "tom_returns.csv")
    (R / "tom_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print(f"TOM: full SR {m['full']['sharpe']:.2f}  CAGR {m['full']['cagr']*100:.1f}%  "
          f"MDD {m['full']['mdd']*100:.1f}%")


if __name__ == "__main__":
    main()
