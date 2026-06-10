"""Price data download and caching (self-contained; uses its own cache dir)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import CACHE_DIR, DATA_START
from universe import UNIVERSE, MARKET_TICKER, VIX_TICKER

CLOSE_FILE = CACHE_DIR / "close.parquet"


def _download(tickers: list[str]) -> pd.DataFrame:
    """Download adjusted daily closes for tickers, retrying once on failure."""
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers, start=DATA_START, auto_adjust=True,
                progress=False, threads=True, group_by="column",
            )
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            return close
        except Exception as exc:  # network hiccups
            if attempt == 2:
                raise
            print(f"download failed ({exc}), retrying...")
            time.sleep(5 * (attempt + 1))


def load_prices(refresh: bool = False, max_age_hours: float = 20.0) -> pd.DataFrame:
    """Return a dates x tickers DataFrame of adjusted closes.

    Includes the tradeable universe plus SPY and ^VIX context columns.
    Cached to parquet; re-downloaded when stale or refresh=True.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    if CLOSE_FILE.exists() and not refresh:
        age = datetime.now() - datetime.fromtimestamp(CLOSE_FILE.stat().st_mtime)
        if age < timedelta(hours=max_age_hours):
            return pd.read_parquet(CLOSE_FILE)

    tickers = UNIVERSE + [MARKET_TICKER, VIX_TICKER]
    print(f"Downloading {len(tickers)} tickers from {DATA_START}...")
    close = _download(tickers)
    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # Drop tickers with essentially no data
    good = close.columns[close.notna().sum() > 252]
    dropped = sorted(set(tickers) - set(good))
    if dropped:
        print(f"Dropping {len(dropped)} tickers with insufficient data: {dropped}")
    close = close[good]
    close.to_parquet(CLOSE_FILE)
    print(f"Cached {close.shape[0]} days x {close.shape[1]} tickers "
          f"({close.index[0].date()} .. {close.index[-1].date()})")
    return close
