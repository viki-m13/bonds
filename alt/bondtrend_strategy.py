"""BONDTREND — long/flat duration trend, expressed via TMF; cash in BIL.

Rule (designed on IS 2010-2018 only; see PHOENIX_V3.md):
  - TLT 63-day momentum (lagged 1 day) > 0  ->  100% TMF, else 100% BIL.
  - Weekly Wednesday rebalance (frozen in between).
  - A short leg (TMV/TBT on negative momentum) was tested on IS and
    REJECTED (it degraded IS Sharpe in a falling-rate decade); long/flat
    was the best IS variant.
  - 10 bps/side TMF, 2 bps BIL.

Booking: unified realization-dated convention (alt/sleeve_engine.py).

Outputs:
  data/results/bondtrend_returns.csv  (Date, ret)
  data/results/bondtrend_metrics.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF = ROOT / "data/etfs"
R = ROOT / "data/results"

SIGNAL_TICKER = "TLT"
LONG_VEHICLE = "TMF"
CASH = "BIL"
MOM_LB = 63
REBAL_DOW = 2  # Wednesday
COST_BPS = {"TMF": 10.0, "BIL": 2.0}
START_DATE = pd.Timestamp("2010-03-11")
IS_END = "2018-12-31"


def load_etf(t):
    df = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).set_index("Date")
    return df[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")


def build_weights(live_extend: bool = False) -> pd.DataFrame:
    """Decision-dated daily target weights over {TMF, BIL}."""
    dates = load_etf("SPY")["Open"].dropna().index
    if live_extend and len(dates) > 0:
        dates = dates.append(pd.DatetimeIndex([dates[-1] + pd.tseries.offsets.BDay()]))
    tlt = load_etf(SIGNAL_TICKER)["Close"].reindex(dates).ffill(limit=3)
    mom = tlt.pct_change(MOM_LB).shift(1)
    W = pd.DataFrame(0.0, index=dates, columns=[LONG_VEHICLE, CASH])
    W.loc[mom > 0, LONG_VEHICLE] = 1.0
    W.loc[~(mom > 0), CASH] = 1.0
    # weekly freeze
    reb = (pd.Series(dates.dayofweek, index=dates) == REBAL_DOW)
    reb.iloc[0] = True
    mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], W.shape),
                        index=W.index, columns=W.columns)
    W = W.where(mask, np.nan).ffill().fillna(0.0)
    return W.loc[START_DATE:]


def main():
    from sleeve_engine import backtest_weights, perf_block

    W = build_weights()
    opens = pd.DataFrame({t: load_etf(t)["Open"] for t in [LONG_VEHICLE, CASH]}).reindex(W.index)
    bt = backtest_weights(W, opens, COST_BPS)
    ret = bt["ret"]

    m = {"full": perf_block(ret), "is": perf_block(ret.loc[:IS_END]),
         "oos": perf_block(ret.loc["2019-01-02":]),
         "params": {"signal": SIGNAL_TICKER, "vehicle": LONG_VEHICLE, "mom_lb": MOM_LB,
                    "rebal_dow": REBAL_DOW, "cost_bps": COST_BPS}}
    out = ret.to_frame("ret")
    out.index.name = "Date"
    out.to_csv(R / "bondtrend_returns.csv")
    (R / "bondtrend_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print(f"BONDTREND: full SR {m['full']['sharpe']:.2f}  CAGR {m['full']['cagr']*100:.1f}%  "
          f"MDD {m['full']['mdd']*100:.1f}%")


if __name__ == "__main__":
    main()
