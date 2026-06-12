"""Shared data layer for the DCA stock-selection research.

Builds aligned wide panels (Open/High/Low/Close/Volume) for the point-in-time
S&P 500 universe, plus a daily boolean membership mask so signals can only
select names that were index members on the signal date.

All prices are Yahoo auto-adjusted (splits + dividends), so Close-to-Close
returns are total returns and Open prices are consistently adjusted.
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIT_DIR = os.path.join(ROOT, "data", "pit")
PRICE_DIR = os.path.join(PIT_DIR, "prices")
CACHE = os.path.join(PIT_DIR, "panel.parquet")


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Repair corrupted Yahoo records on delisted tickers.

    Two defenses (data repair, not signal logic — bad ticks never existed in
    the real market):
      1. Bad-tick removal: a price >4x or <0.25x the median of its +/-5-day
         neighborhood is a recording error -> drop the row.
      2. Truncation: if the cleaned close series still jumps >75% in a day
         (essentially impossible for an S&P large cap), the tail is a
         different/garbage listing -> truncate before the first jump.
    """
    med = df["Close"].rolling(11, center=True, min_periods=3).median()
    ratio = df["Close"] / med
    bad = (ratio > 4) | (ratio < 0.25)
    if bad.any():
        df = df[~bad]
    r = df["Close"].pct_change()
    crazy = r.abs() > 0.75
    if crazy.any():
        first = crazy.idxmax()
        df = df.loc[:first].iloc[:-1]
    return df


def _load_membership() -> pd.DataFrame:
    mem = pd.read_csv(os.path.join(PIT_DIR, "sp500_pit_membership.csv"))
    mem["date"] = pd.to_datetime(mem["date"])
    return mem.set_index("date").sort_index()


def build_panel(force: bool = False) -> dict:
    """Return dict of wide DataFrames: open, high, low, close, volume, member."""
    fields = ["open", "high", "low", "close", "volume"]
    paths = {f: os.path.join(PIT_DIR, f"panel_{f}.parquet") for f in fields + ["member"]}
    if not force and all(os.path.exists(p) for p in paths.values()):
        return {f: pd.read_parquet(p) for f, p in paths.items()}

    frames = {f: {} for f in fields}
    for fn in sorted(os.listdir(PRICE_DIR)):
        if not fn.endswith(".csv"):
            continue
        t = fn[:-4]
        df = pd.read_csv(os.path.join(PRICE_DIR, fn), index_col=0, parse_dates=True)
        df = df[~df.index.duplicated()]
        # Ticker-recycling guard: Yahoo serves some delisted tickers with a
        # long gap followed by a *different* company's prices (e.g. CFC, MEE).
        # Keep only the first continuous listing segment.
        gaps = df.index.to_series().diff().dt.days
        brk = gaps[gaps > 30]
        if len(brk):
            df = df.loc[:brk.index[0] - pd.Timedelta(days=1)]
        df = _clean_prices(df)
        if len(df) < 50:
            continue
        for f in fields:
            col = f.capitalize()
            if col in df.columns:
                frames[f][t] = df[col]
    panels = {f: pd.DataFrame(frames[f]).sort_index() for f in fields}

    # Daily membership mask aligned to trading days, forward-filled between
    # membership snapshot dates. Tickers use '-' like the price files.
    mem = _load_membership()
    idx = panels["close"].index
    cols = panels["close"].columns
    snap_dates = mem.index
    # for each trading day, use the most recent snapshot on/before it
    locs = snap_dates.searchsorted(idx, side="right") - 1
    cache = {}
    col_pos = {c: i for i, c in enumerate(cols)}
    arr = np.zeros((len(idx), len(cols)), dtype=bool)
    for i, li in enumerate(locs):
        if li < 0:
            continue
        if li not in cache:
            ticks = [t.replace(".", "-") for t in mem["tickers"].iloc[li].split(",")]
            cache[li] = [col_pos[t] for t in ticks if t in col_pos]
        arr[i, cache[li]] = True
    member = pd.DataFrame(arr, index=idx, columns=cols)
    panels["member"] = member

    for f, p in paths.items():
        panels[f].to_parquet(p)
    return panels


def build_panel_n100(force: bool = False) -> dict:
    """Secondary universe: NASDAQ-100 point-in-time (2015+), from the
    jmccarrell/n100tickers dataset. Coverage is partial (delisted names that
    Yahoo lacks) — use only as a transfer/robustness check with its own
    random-pick control."""
    fields = ["open", "high", "low", "close", "volume"]
    paths = {f: os.path.join(PIT_DIR, f"n100_panel_{f}.parquet")
             for f in fields + ["member"]}
    if not force and all(os.path.exists(p) for p in paths.values()):
        return {f: pd.read_parquet(p) for f, p in paths.items()}

    mem = pd.read_csv(os.path.join(PIT_DIR, "n100_pit_membership.csv"))
    mem["date"] = pd.to_datetime(mem["date"])
    mem = mem.set_index("date").sort_index()
    allt = sorted({t.replace(".", "-") for row in mem["tickers"]
                   for t in row.split(",")})

    n100_dir = os.path.join(PIT_DIR, "prices_n100")
    frames = {f: {} for f in fields}
    for t in allt:
        for d in (n100_dir, PRICE_DIR):
            p = os.path.join(d, f"{t}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                df = df[~df.index.duplicated()]
                gaps = df.index.to_series().diff().dt.days
                brk = gaps[gaps > 30]
                if len(brk):
                    df = df.loc[:brk.index[0] - pd.Timedelta(days=1)]
                df = _clean_prices(df)
                if len(df) < 50:
                    break
                for f in fields:
                    col = f.capitalize()
                    if col in df.columns:
                        frames[f][t] = df[col]
                break
    panels = {f: pd.DataFrame(frames[f]).sort_index() for f in fields}
    idx, cols = panels["close"].index, panels["close"].columns
    locs = mem.index.searchsorted(idx, side="right") - 1
    arr = np.zeros((len(idx), len(cols)), dtype=bool)
    col_pos = {c: i for i, c in enumerate(cols)}
    cache = {}
    for i, li in enumerate(locs):
        if li < 0:
            continue
        if li not in cache:
            ticks = [t.replace(".", "-") for t in
                     mem["tickers"].iloc[li].split(",")]
            cache[li] = [col_pos[t] for t in ticks if t in col_pos]
        arr[i, cache[li]] = True
    panels["member"] = pd.DataFrame(arr, index=idx, columns=cols)
    for f, p in paths.items():
        panels[f].to_parquet(p)
    return panels


def load_benchmark(ticker: str) -> pd.DataFrame:
    """Adjusted OHLCV for a benchmark ETF from data/etfs_extended (Adj-scaled)."""
    path = os.path.join(ROOT, "data", "etfs_extended", f"{ticker}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # OHLC in these files is already split+dividend adjusted (auto_adjust
    # downloads); the partial "Adj Close" column only exists on recently
    # appended rows and duplicates Close.
    df = df[~df.index.duplicated()]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(
        subset=["Open", "Close"])


def coverage_report() -> pd.DataFrame:
    """Per-year share of PIT members for which we actually have prices."""
    panels = build_panel()
    close, member = panels["close"], panels["member"]
    have = close.notna()
    mem = _load_membership()
    rows = []
    for dt in pd.date_range("2005-01-01", close.index[-1], freq="YS"):
        li = mem.index.searchsorted(dt, side="right") - 1
        ticks = {t.replace(".", "-") for t in mem["tickers"].iloc[li].split(",")}
        day = close.index[close.index.searchsorted(dt)]
        covered = sum(1 for t in ticks if t in have.columns and have.loc[day, t])
        rows.append({"year": dt.year, "members": len(ticks), "covered": covered,
                     "pct": covered / len(ticks)})
    return pd.DataFrame(rows)
