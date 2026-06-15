"""Data layer for the HF underwater-avoidance stock picker.

Reads the point-in-time S&P 500 panel (`data/pit/summit_panel.parquet`) and
exposes aligned wide frames: open, close, volume, and a boolean daily
membership mask. All prices are Yahoo auto-adjusted (splits + dividends), so
close-to-close and open-to-close moves are total returns.

The objective for this project is NOT maximum return — it is *entry quality*:
buy names that, after purchase, spend little or no time below the purchase
price. Everything downstream is built to measure and optimize that.
"""
import os
import functools

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, "data", "pit", "summit_panel.parquet")


@functools.lru_cache(maxsize=1)
def load_panel() -> dict:
    """Return dict of wide DataFrames: open, close, volume, member.

    Cached so the (large) parquet read happens once per process.
    """
    d = pd.read_parquet(PANEL)
    fields = sorted({c[0] for c in d.columns})
    out = {f: d[f].sort_index() for f in fields}
    # member as bool
    out["member"] = out["member"].fillna(0).astype(bool)
    return out


def eligibility(min_history: int = 252) -> pd.DataFrame:
    """Boolean (day x ticker): index member, priced today, and >= min_history
    of trailing closes (so trailing features are defined)."""
    p = load_panel()
    close = p["close"]
    enough = close.notna().rolling(min_history).count() >= min_history
    return p["member"] & enough & close.notna()


def load_benchmark(ticker: str) -> pd.DataFrame:
    """Adjusted OHLCV for a benchmark ETF (SPY, QQQ, ...) from etfs_extended."""
    path = os.path.join(ROOT, "data", "etfs_extended", f"{ticker}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(
        subset=["Open", "Close"])


def trading_days() -> pd.DatetimeIndex:
    return load_panel()["close"].index


if __name__ == "__main__":
    p = load_panel()
    c = p["close"]
    print("panel:", c.shape, c.index.min().date(), "->", c.index.max().date())
    print("fields:", sorted(p))
    elig = eligibility()
    print("median eligible/day (2015+):",
          int(elig.loc["2015":].sum(axis=1).median()))
