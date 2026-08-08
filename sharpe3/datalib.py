"""Data layer for the sharpe3 stock-picking research.

Point-in-time universes available:
  - S&P 500 PIT panel ("summit"): 2004-2026, ~410 members/day, open/close/volume/member.
  - NASDAQ-100 PIT panel ("n100"): 2004-2026, OHLCV + member.
  - Tiingo full US universe (delisting-inclusive adjClose+volume, 1990-2026).

Causality contract (same as dca/RESEARCH_PROTOCOL.md): a signal row dated d may
use information through the close of day d only; execution is next open (or
next close), handled by bt.py.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIT = os.path.join(ROOT, "data", "pit")
TIINGO = os.path.join(ROOT, "dca", "research", "data", "tiingo")


def load_summit():
    """S&P 500 PIT panel: dict of wide frames open/close/volume/member."""
    p = pd.read_parquet(os.path.join(PIT, "summit_panel.parquet"))
    out = {f: p[f] for f in ("open", "close", "volume", "member")}
    out["member"] = out["member"].fillna(False).astype(bool)
    return out


def load_n100():
    """NASDAQ-100 PIT panel: dict open/high/low/close/volume/member."""
    out = {}
    for f in ("open", "high", "low", "close", "volume", "member"):
        out[f] = pd.read_parquet(os.path.join(PIT, f"n100_panel_{f}.parquet"))
    out["member"] = out["member"].fillna(False).astype(bool)
    return out


def load_tiingo_universe():
    """PIT listing key: ticker, exchange, assetType, startDate, endDate."""
    return pd.read_parquet(os.path.join(TIINGO, "tiingo_universe_pit.parquet"))


def load_tiingo_prices(tickers=None, field="ac"):
    """Concat tiingo chunked wide parquets. field: 'ac' (adjClose) or 'vol'."""
    pdir = os.path.join(TIINGO, "prices")
    frames = []
    for fn in sorted(os.listdir(pdir)):
        if fn.startswith(field + "_") and fn.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(pdir, fn))
            if tickers is not None:
                keep = [t for t in df.columns if t in tickers]
                if not keep:
                    continue
                df = df[keep]
            frames.append(df)
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def dollar_volume(panel):
    """Trailing 20d median dollar volume, in $."""
    dv = (panel["close"] * panel["volume"]).rolling(20, min_periods=5).median()
    return dv
