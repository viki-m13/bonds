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


def build_weights(live_extend: bool = False) -> pd.DataFrame:
    """Decision-dated daily target weights ({VEHICLE} column).

    Caveat on the live edge: the last month in the index is still in
    progress, so 'last N days of the month' is only known for it once the
    next month begins — EXCEPT that the NYSE calendar is deterministic, so
    the live layer appends the next business day and the month window is
    evaluated on the known exchange calendar.
    """
    dates = load_etf("SPY")["Open"].dropna().index
    if live_extend and len(dates) > 0:
        dates = dates.append(pd.DatetimeIndex([dates[-1] + pd.tseries.offsets.BDay()]))
    ym = dates.to_period("M")
    pos = pd.Series(0.0, index=dates)
    for p in ym.unique():
        m = dates[ym == p]
        pos.loc[m[-N_BEFORE:]] = 1.0
        pos.loc[m[:N_AFTER]] = 1.0
    W = pd.DataFrame({VEHICLE: pos})
    return W.loc[START_DATE:]


def main():
    from sleeve_engine import backtest_weights, perf_block

    W = build_weights()
    opens = load_etf(VEHICLE)[["Open"]].rename(columns={"Open": VEHICLE}).reindex(W.index)
    bt = backtest_weights(W, opens, COST_BPS)
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
