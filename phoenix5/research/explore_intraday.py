"""Exploratory: session-level edges in the 5-min intraday data (2016-2026).

Tests (all signals lagged correctly, evaluated IS 2016-2018 / OOS 2019+):
  1. Intraday momentum: first 30-min return sign -> hold last hour (Gao-Han-Li-Zhou).
  2. Overnight session: long close->open, flat intraday (using daily etfs data, 2005+).
  3. Last-hour momentum: rest-of-day return -> last 30 min.
TC assumption: 1 bp per side for SPY/QQQ-class liquidity (round trip 2 bp / day traded).
"""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
I5 = ROOT / "data/intraday_5min"
ETF = ROOT / "data/etfs"
TC = 0.0001  # 1 bp per side

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLF"]


def sr(r, ann=252):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0


def load_5min(t):
    df = pd.read_csv(I5 / f"{t}.csv", parse_dates=["ts"])
    df["date"] = df["ts"].dt.date
    df["time"] = df["ts"].dt.time
    return df


def session_table(t):
    """Per-day: open, close at key times."""
    df = load_5min(t)
    piv = df.pivot_table(index="date", columns="time", values="close")
    op = df.pivot_table(index="date", columns="time", values="open")
    out = pd.DataFrame(index=piv.index)
    import datetime as dt
    out["o0930"] = op.get(dt.time(9, 30))
    out["c1000"] = piv.get(dt.time(10, 0))   # close of 10:00 bar = 10:05... use 9:55 bar close for first 30min
    out["c0955"] = piv.get(dt.time(9, 55))   # first 30 min: 9:30->10:00 = close of 9:55 bar
    out["c1430"] = piv.get(dt.time(14, 25))  # 14:30 level
    out["c1500"] = piv.get(dt.time(14, 55))  # 15:00 level
    out["c1525"] = piv.get(dt.time(15, 25))  # 15:30 level
    out["close"] = piv.get(dt.time(15, 55))  # last bar close ~ 16:00
    out.index = pd.to_datetime(out.index)
    return out.dropna()


def test_intraday_momentum():
    print("=" * 70)
    print("1) INTRADAY MOMENTUM: sign(first 30 min) -> hold 15:30->close")
    print("=" * 70)
    streams = {}
    for t in TICKERS:
        s = session_table(t)
        r_first = s["c0955"] / s["o0930"] - 1
        r_last = s["close"] / s["c1525"] - 1
        sig = np.sign(r_first)
        ret = sig * r_last - 2 * TC  # in and out every day
        streams[t] = ret
        m = ret.loc[:"2018"]
        o = ret.loc["2019":]
        print(f"  {t:5s} IS SR={sr(m):5.2f}  OOS SR={sr(o):5.2f}  full={sr(ret):5.2f}  n={len(ret)}")
    df = pd.concat(streams, axis=1)
    eq = df.mean(axis=1)
    print(f"  EQW   IS SR={sr(eq.loc[:'2018']):5.2f}  OOS SR={sr(eq.loc['2019':]):5.2f}  full={sr(eq):5.2f}")
    return df


def test_intraday_momentum_v2():
    print("=" * 70)
    print("1b) INTRADAY MOM v2: sign(open->14:30) -> hold 14:30->close")
    print("=" * 70)
    streams = {}
    for t in TICKERS:
        s = session_table(t)
        r_sofar = s["c1430"] / s["o0930"] - 1
        r_last = s["close"] / s["c1430"] - 1
        sig = np.sign(r_sofar)
        ret = sig * r_last - 2 * TC
        streams[t] = ret
        print(f"  {t:5s} IS SR={sr(ret.loc[:'2018']):5.2f}  OOS SR={sr(ret.loc['2019':]):5.2f}  full={sr(ret):5.2f}")
    df = pd.concat(streams, axis=1)
    eq = df.mean(axis=1)
    print(f"  EQW   IS SR={sr(eq.loc[:'2018']):5.2f}  OOS SR={sr(eq.loc['2019':]):5.2f}  full={sr(eq):5.2f}")
    return df


def test_overnight():
    print("=" * 70)
    print("2) OVERNIGHT: long close->open every night (daily etfs data, 2005+)")
    print("=" * 70)
    streams = {}
    for t in TICKERS:
        d = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")
        on = d["Open"] / d["Close"].shift(1) - 1
        intra = d["Close"] / d["Open"] - 1
        ret = on - 2 * TC
        streams[t] = ret
        print(f"  {t:5s} overnight SR full={sr(ret):5.2f} IS(:2018)={sr(ret.loc[:'2018']):5.2f} OOS={sr(ret.loc['2019':]):5.2f} | intraday SR={sr(intra):5.2f}")
    return pd.concat(streams, axis=1)


def main():
    df1 = test_intraday_momentum()
    df2 = test_intraday_momentum_v2()
    df3 = test_overnight()


if __name__ == "__main__":
    main()
