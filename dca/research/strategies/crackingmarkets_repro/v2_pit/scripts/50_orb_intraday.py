"""Intraday ORB (Opening Range Breakout) on the committed 5-min ETF bars —
honest reproduction of the Zarattini/Barbon/Aziz setup, with costs.

Data: data/intraday_5min/{DIA,GLD,IWM,QQQ,SPY,TLT,XLF}.csv, 2016-01..2026-04,
RTH 09:30-15:55 (78 bars/day; the 15:55 bar close is the session close).

Baseline spec (paper: "Can Day Trading Really Be Profitable?", 2023):
  - opening range = first 5-min bar (09:30-09:35)
  - direction = sign of the first bar (close vs open); doji -> no trade
  - enter at the 09:35 bar OPEN in that direction
  - stop = the other extreme of the opening range
  - if stopped: exit at the stop (plus slippage); else exit at session close
  - sizing: risk R% of equity per trade = distance to stop, subject to a
    max intraday leverage cap (4x day-trading margin; no overnight, so no
    financing and no gap risk)
Variants: 15/30-min opening range (entry on break of range high/low),
long-only vs long-short, all 7 ETFs.

Cost model per side: half-spread (QQQ/SPY 0.5bp, others 1bp) + $0.005/sh
commission (~0.15bp at these prices) -> 1bp/side QQQ/SPY, 1.5bp others.
Stop fills get +2bp adverse slippage; if a bar gaps through the stop, fill
at the bar open (worse), not at the stop.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import stats, fmt, riskfree_daily, OUT

ID = "/home/user/bonds/data/intraday_5min"
COST = {"QQQ": 1.0, "SPY": 1.0, "IWM": 1.5, "DIA": 1.5, "GLD": 1.5,
        "TLT": 1.5, "XLF": 1.5}          # bps per side
STOP_SLIP = 2.0                          # extra bps when a stop triggers


def load(tk):
    df = pd.read_csv(os.path.join(ID, f"{tk}.csv"), parse_dates=["ts"])
    df["date"] = df["ts"].dt.normalize()
    df["t"] = df["ts"].dt.time
    return df


def orb(tk, or_bars=1, risk=0.01, maxlev=4.0, side="both", entry_mode="first",
        costs=True):
    """One-ETF ORB. Returns daily strategy returns (fraction of equity).

    entry_mode 'first': direction = sign of opening range candle(s), enter at
    the next bar's open (paper baseline, or_bars=1).
    entry_mode 'break': enter long the moment a later bar closes above the
    OR high (short below OR low), at the next bar's open.
    """
    df = load(tk)
    cost = COST[tk] / 1e4 if costs else 0.0
    slip = STOP_SLIP / 1e4 if costs else 0.0
    out = {}
    for d, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) < or_bars + 3:
            continue
        orh, orl = g["high"][:or_bars].max(), g["low"][:or_bars].min()
        oro, orc = g["open"][0], g["close"][or_bars - 1]
        sess_close = g["close"].iloc[-1]
        dirn, e_i = 0, None
        if entry_mode == "first":
            if orc > oro:
                dirn = 1
            elif orc < oro:
                dirn = -1
            e_i = or_bars
        else:
            for i in range(or_bars, len(g) - 1):
                if g["close"][i] > orh:
                    dirn, e_i = 1, i + 1
                    break
                if g["close"][i] < orl:
                    dirn, e_i = -1, i + 1
                    break
        if dirn == 0 or e_i is None or e_i >= len(g):
            out[d] = 0.0
            continue
        if side == "long" and dirn < 0:
            out[d] = 0.0
            continue
        entry = g["open"][e_i]
        stop = orl if dirn > 0 else orh
        stop_frac = abs(entry - stop) / entry
        if stop_frac <= 0.0005:            # degenerate range
            out[d] = 0.0
            continue
        lev = min(risk / stop_frac, maxlev)
        # walk forward to stop or close
        exit_px, stopped = sess_close, False
        seg = g.iloc[e_i:]
        if dirn > 0:
            hit = seg[seg["low"] <= stop]
            if len(hit):
                b = hit.iloc[0]
                exit_px = min(stop, b["open"])
                stopped = True
        else:
            hit = seg[seg["high"] >= stop]
            if len(hit):
                b = hit.iloc[0]
                exit_px = max(stop, b["open"])
                stopped = True
        gross = dirn * (exit_px / entry - 1)
        fees = 2 * cost + (slip if stopped else 0)
        out[d] = lev * (gross - fees)
    return pd.Series(out).sort_index()


if __name__ == "__main__":
    t0 = time.time()
    print("=== ORB baseline: first-5-min-bar direction, risk 1%, cap 4x, "
          "long-short, costed ===")
    res = {}
    for tk in ["QQQ", "SPY", "IWM", "DIA", "XLF", "GLD", "TLT"]:
        r = orb(tk)
        rf = riskfree_daily(r.index)
        res[tk] = r
        print(fmt(stats(r, rf, f"ORB5 {tk}")) +
              f"  trades {(r != 0).mean()*100:.0f}% of days")
    print()
    print("=== QQQ variants ===")
    qs = {}
    for label, kw in [
        ("ORB5 QQQ long-only",          dict(side="long")),
        ("ORB5 QQQ risk 2%",            dict(risk=0.02)),
        ("ORB5 QQQ FREE (no costs)",    dict(costs=False)),
        ("ORB15 break QQQ",             dict(or_bars=3, entry_mode="break")),
        ("ORB30 break QQQ",             dict(or_bars=6, entry_mode="break")),
        ("ORB15 break QQQ long-only",   dict(or_bars=3, entry_mode="break",
                                             side="long")),
    ]:
        r = orb("QQQ", **kw)
        rf = riskfree_daily(r.index)
        qs[label] = r
        print(fmt(stats(r, rf, label)) +
              f"  trades {(r != 0).mean()*100:.0f}% of days")
    keep = pd.DataFrame({"orb_qqq": res["QQQ"], "orb_spy": res["SPY"],
                         "orb_iwm": res["IWM"]})
    keep.to_parquet(os.path.join(OUT, "sleeveG_orb.parquet"))
    print(f"\nsaved -> out/sleeveG_orb.parquet  t={time.time()-t0:.0f}s")
