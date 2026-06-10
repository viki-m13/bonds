"""Refine the intraday-reversal edge + test other candidate sleeves.

A) Intraday reversal variants (which window to fade, sizing).
B) Gap fade: fade overnight gap during the day.
C) Cross-sectional short-term reversal on ~100 large-cap stocks.
D) Treasury duration carry+momentum timing.
All: IS <= 2018-12-31, OOS >= 2019-01-01, costs included.
"""
import pandas as pd, numpy as np
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
I5 = ROOT / "data/intraday_5min"
ETF = ROOT / "data/etfs"
STK = ROOT / "data/stocks"
FRED = ROOT / "data/fred"
TC = 0.0001

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLF"]


def sr(r, ann=252):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if len(r) > 5 and r.std() > 0 else 0.0


def isoos(ret, label):
    print(f"  {label:34s} IS={sr(ret.loc[:'2018']):5.2f}  OOS={sr(ret.loc['2019':]):5.2f}  full={sr(ret):5.2f}")


def session_levels(t):
    df = pd.read_csv(I5 / f"{t}.csv", parse_dates=["ts"])
    df["date"] = df["ts"].dt.date
    df["time"] = df["ts"].dt.time
    piv = df.pivot_table(index="date", columns="time", values="close")
    op = df.pivot_table(index="date", columns="time", values="open")
    out = pd.DataFrame(index=pd.to_datetime(piv.index))
    piv.index = out.index; op.index = out.index
    out["o0930"] = op.get(dt.time(9, 30)).values
    for h, m in [(9,55),(10,25),(10,55),(11,55),(13,55),(14,25),(14,55),(15,25),(15,55)]:
        out[f"c{h:02d}{m:02d}"] = piv.get(dt.time(h, m)).values
    return out.dropna(subset=["o0930", "c1555"])


def part_a():
    print("=" * 78)
    print("A) INTRADAY REVERSAL VARIANTS (fade signal window, trade exec window)")
    print("=" * 78)
    levels = {t: session_levels(t) for t in TICKERS}
    variants = {
        "fade 930-1000 in 1530-1600": ("o0930", "c0955", "c1525", "c1555"),
        "fade 930-1030 in 1530-1600": ("o0930", "c1025", "c1525", "c1555"),
        "fade 930-1100 in 1500-1600": ("o0930", "c1055", "c1455", "c1555"),
        "fade 930-1500 in 1500-1600": ("o0930", "c1455", "c1455", "c1555"),
        "fade 1430-1530 in 1530-1600": ("c1425", "c1525", "c1525", "c1555"),
        "fade 930-1000 in 1000-1100": ("o0930", "c0955", "c0955", "c1055"),
        "fade 930-1000 in 1000-1600": ("o0930", "c0955", "c0955", "c1555"),
    }
    best = {}
    for name, (s0, s1, e0, e1) in variants.items():
        rs = {}
        for t in TICKERS:
            L = levels[t]
            sig = -np.sign(L[s1] / L[s0] - 1)
            ret = sig * (L[e1] / L[e0] - 1) - 2 * TC
            rs[t] = ret
        eq = pd.concat(rs, axis=1, sort=True).mean(axis=1)
        isoos(eq, name)
        best[name] = eq

    print("\n  -- sizing variants on 'fade 930-1000 in 1530-1600' --")
    rs, rs_z = {}, {}
    for t in TICKERS:
        L = levels[t]
        r1 = L["c0955"] / L["o0930"] - 1
        rlast = L["c1555"] / L["c1525"] - 1
        # z-scored signal capped at 2: bigger move -> bigger fade
        z = (r1 / r1.rolling(60).std()).clip(-2, 2)
        rs_z[t] = (-z) * rlast - 2 * TC * (z.abs() / 2)
        # threshold: only trade when |move| > 0.5 * 60d std
        thr = r1.rolling(60).std() * 0.5
        sig = pd.Series(0.0, index=L.index)
        sig[r1 > thr] = -1.0
        sig[r1 < -thr] = 1.0
        rs[t] = sig * rlast - 2 * TC * sig.abs()
    isoos(pd.concat(rs, axis=1, sort=True).mean(axis=1), "threshold |z|>0.5")
    isoos(pd.concat(rs_z, axis=1, sort=True).mean(axis=1), "linear z-sized (cap 2)")
    return best


def part_b():
    print("=" * 78)
    print("B) GAP FADE: fade overnight gap during next day (daily data, 2005+)")
    print("=" * 78)
    for t in TICKERS:
        d = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")
        gap = d["Open"] / d["Close"].shift(1) - 1
        intra = d["Close"] / d["Open"] - 1
        z = (gap / gap.rolling(60).std()).clip(-2, 2)
        ret = (-z) * intra - 2 * TC * z.abs() / 2
        isoos(ret, f"{t} gap-fade z-sized")


def part_c():
    print("=" * 78)
    print("C) CROSS-SECTIONAL 5d REVERSAL, ~100 large caps, weekly, 5bp/side")
    print("=" * 78)
    closes = {}
    for f in STK.glob("*.csv"):
        t = f.stem
        if t == "BIL":
            continue
        d = pd.read_csv(f, parse_dates=["Date"], index_col="Date")["Close"]
        closes[t] = d
    px = pd.concat(closes, axis=1, sort=True)
    px = px.loc["2009":]
    print(f"  universe: {px.shape[1]} stocks, {px.index[0].date()} -> {px.index[-1].date()}")
    r1 = px.pct_change()
    r5 = px.pct_change(5)
    vol = r1.rolling(60).std()
    # market-neutral: rank of -(5d ret / vol), long bottom decile losers, short top decile winners
    score = (r5 / (vol * np.sqrt(5))).rank(axis=1, pct=True)
    n_valid = score.notna().sum(axis=1)
    long_w = (score <= 0.2).div(n_valid * 0.2, axis=0)
    short_w = (score >= 0.8).div(n_valid * 0.2, axis=0)
    w = (long_w - short_w)
    # hold 5 days (rebalance weekly on Mondays), execute next open... use close-to-close t+1 signal lag
    monday = pd.Series(w.index.dayofweek == 0, index=w.index)
    w = w.where(monday, np.nan).ffill(limit=5).shift(1).fillna(0)
    gross = (w * r1).sum(axis=1)
    to = w.diff().abs().sum(axis=1)
    net = gross - to * 0.0005
    isoos(net, "5d vol-adj reversal L/S weekly")
    # momentum 12-1 monthly long-only top decile vs SPY hedge skipped; test 21d industry-free reversal
    score2 = (r1.rolling(21).sum() / (vol * np.sqrt(21))).rank(axis=1, pct=True)
    w2 = ((score2 <= 0.2).div(n_valid * 0.2, axis=0) - (score2 >= 0.8).div(n_valid * 0.2, axis=0))
    w2 = w2.where(monday, np.nan).ffill(limit=5).shift(1).fillna(0)
    net2 = (w2 * r1).sum(axis=1) - w2.diff().abs().sum(axis=1) * 0.0005
    isoos(net2, "21d vol-adj reversal L/S weekly")


def part_d():
    print("=" * 78)
    print("D) TREASURY DURATION TIMING: carry + momentum on TLT/IEF vs cash")
    print("=" * 78)
    dgs10 = pd.read_csv(FRED / "DGS10.csv", parse_dates=["Date"], index_col="Date")["DGS10"]
    dgs2 = pd.read_csv(FRED / "DGS2.csv", parse_dates=["Date"], index_col="Date")["DGS2"]
    ff = pd.read_csv(FRED / "DGS3MO.csv", parse_dates=["Date"], index_col="Date")["DGS3MO"]
    tlt = pd.read_csv(ETF / "TLT.csv", parse_dates=["Date"], index_col="Date")["Close"]
    ief = pd.read_csv(ETF / "IEF.csv", parse_dates=["Date"], index_col="Date")["Close"]
    rt = tlt.pct_change()
    ri = ief.pct_change()
    carry = (dgs10 - ff).reindex(rt.index).ffill()      # term carry
    mom = tlt.pct_change(63)                              # 3m momentum
    sig = (0.5 * np.sign(carry) + 0.5 * np.sign(mom)).shift(1)
    ret = sig * rt - (sig.diff().abs().fillna(0)) * 0.0002
    isoos(ret, "TLT carry+mom sign blend")
    sig2 = np.sign(carry).shift(1)
    isoos(sig2 * rt, "TLT pure carry sign")
    sig3 = np.sign(mom).shift(1)
    isoos(sig3 * rt, "TLT pure 3m mom sign")
    # 10s2s steepener proxy: long IEF short duration-matched TLT when curve inverted rising... keep simple
    st = (dgs10 - dgs2).reindex(rt.index).ffill()
    sig4 = np.sign(st.diff(21)).shift(1)
    isoos(sig4 * (ri - 0.45 * rt), "curve steepener 21d trend")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
