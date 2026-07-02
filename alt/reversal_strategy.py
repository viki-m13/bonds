"""REVERSAL — short-horizon dip-buying inside uptrends, expressed via LETFs.

Rules (designed on IS 2010-2018 only; see PHOENIX_V3.md):
  - Underlyings {QQQ, SPY, SMH}; expressions {TQQQ, UPRO, SOXL}.
  - Signal at decision date t (uses close[t-1] only):
      z = 5-day return z-score vs its trailing 60d distribution
      trigger when z < -1.0 AND close > 200d SMA (dip inside an uptrend)
  - A trigger keeps the name active for the next 5 sessions.
  - Active names are equal-weighted; total capped at 1.0; cash otherwise
    implicit at 0% (conservative).
  - 10 bps/side.

Booking: unified realization-dated convention (alt/sleeve_engine.py) —
W[t] decided from close[t-1], held from open[t]; return booked at date t is
the P&L over open[t-1] -> open[t].

Outputs:
  data/results/reversal_returns.csv  (Date, ret)
  data/results/reversal_metrics.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF = ROOT / "data/etfs"
R = ROOT / "data/results"

PAIRS = {"QQQ": "TQQQ", "SPY": "UPRO", "SMH": "SOXL"}
Z_TH = -1.0
HOLD = 5
Z_WIN = 60
SMA_LB = 200
COST_BPS = 10.0
START_DATE = pd.Timestamp("2010-03-11")
IS_END = "2018-12-31"


def load_etf(t):
    df = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).set_index("Date")
    return df[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")


def _dates(live_extend=False):
    spy = load_etf("SPY")["Open"].dropna().index
    if live_extend and len(spy) > 0:
        spy = spy.append(pd.DatetimeIndex([spy[-1] + pd.tseries.offsets.BDay()]))
    return spy


def build_weights(live_extend: bool = False) -> pd.DataFrame:
    """Decision-dated daily target weights over the LETF expressions.

    live_extend: if True, append one BDay (ffilled closes) so the last row
    is W[t+1] computed from close[t] info.
    """
    dates = _dates(live_extend)
    closes = pd.DataFrame({u: load_etf(u)["Close"].reindex(dates).ffill(limit=3)
                           for u in PAIRS})
    r5 = closes.pct_change(5)
    z = (r5 - r5.rolling(Z_WIN).mean()) / r5.rolling(Z_WIN).std()
    above = closes > closes.rolling(SMA_LB).mean()
    trigger = ((z < Z_TH) & above).shift(1).fillna(False)
    held = trigger.astype(float).rolling(HOLD, min_periods=1).max()
    tot = held.sum(axis=1)
    W = held.div(tot.where(tot > 0), axis=0).mul(tot.clip(upper=1.0), axis=0).fillna(0.0)
    W = W.rename(columns=PAIRS)
    return W.loc[START_DATE:]


def main():
    from sleeve_engine import backtest_weights, perf_block

    W = build_weights()
    opens = pd.DataFrame({l: load_etf(l)["Open"] for l in PAIRS.values()})
    opens = opens.reindex(W.index)
    bt = backtest_weights(W, opens, COST_BPS)
    ret = bt["ret"]

    m = {"full": perf_block(ret), "is": perf_block(ret.loc[:IS_END]),
         "oos": perf_block(ret.loc["2019-01-02":]),
         "params": {"pairs": PAIRS, "z_th": Z_TH, "hold": HOLD, "z_win": Z_WIN,
                    "sma_lb": SMA_LB, "cost_bps": COST_BPS}}
    out = ret.to_frame("ret")
    out.index.name = "Date"
    out.to_csv(R / "reversal_returns.csv")
    (R / "reversal_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print(f"REVERSAL: full SR {m['full']['sharpe']:.2f}  CAGR {m['full']['cagr']*100:.1f}%  "
          f"MDD {m['full']['mdd']*100:.1f}%")


if __name__ == "__main__":
    main()
