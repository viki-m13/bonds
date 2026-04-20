"""Alpaca intraday/daily loader. Keys passed via env vars — NEVER committed.

Usage:
  ALPACA_KEY=... ALPACA_SECRET=... python3 alpaca_loader.py daily
  ALPACA_KEY=... ALPACA_SECRET=... python3 alpaca_loader.py intraday
"""
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


KEY = os.environ.get("ALPACA_KEY")
SECRET = os.environ.get("ALPACA_SECRET")
if not KEY or not SECRET:
    sys.exit("Set ALPACA_KEY and ALPACA_SECRET env vars")

HEADERS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}
BASE = "https://data.alpaca.markets/v2/stocks/bars"

ROOT = Path("/home/user/bonds")
STOCKS_DIR = ROOT / "data/stocks"
ETF_DIR = ROOT / "data/etfs"
OUT_DAILY = ROOT / "data/intraday_daily"
OUT_5MIN = ROOT / "data/intraday_5min"
OUT_DAILY.mkdir(parents=True, exist_ok=True)
OUT_5MIN.mkdir(parents=True, exist_ok=True)


def fetch_bars(symbol, timeframe, start, end, feed="sip", adjustment="all"):
    """Paginated pull. Returns list of bar dicts."""
    params = {
        "symbols": symbol, "timeframe": timeframe,
        "start": start, "end": end, "limit": 10000,
        "adjustment": adjustment, "feed": feed,
    }
    all_bars = []
    while True:
        for attempt in range(4):
            r = requests.get(BASE, headers=HEADERS, params=params, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            print(f"  HTTP {r.status_code} {symbol}: {r.text[:200]}", flush=True)
            return all_bars
        data = r.json()
        bars = (data.get("bars") or {}).get(symbol, [])
        all_bars.extend(bars)
        tok = data.get("next_page_token")
        if not tok:
            break
        params["page_token"] = tok
    return all_bars


def bars_to_df(bars, tz_convert=True):
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df = df.rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume", "n": "trades", "vw": "vwap"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if tz_convert:
        df["ts"] = df["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return df[["ts", "open", "high", "low", "close", "volume", "vwap"]]


def load_tickers():
    stocks = sorted([p.stem for p in STOCKS_DIR.glob("*.csv")])
    etfs = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EFA", "EEM", "VNQ", "XLK", "XLF",
            "XLE", "XLY", "XLP", "XLI", "XLV", "XLU", "XLB", "XLRE", "HYG", "IEF", "SHY", "BIL"]
    return sorted(set(stocks + etfs))


def pull_daily():
    tickers = load_tickers()
    start = "2016-01-02T00:00:00Z"
    end = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
    print(f"DAILY: {len(tickers)} tickers from 2016-01-02 to {end[:10]}", flush=True)

    for i, t in enumerate(tickers):
        out = OUT_DAILY / f"{t}.csv"
        if out.exists():
            continue
        bars = fetch_bars(t, "1Day", start, end)
        df = bars_to_df(bars, tz_convert=False)
        if df.empty:
            print(f"  [{i+1}/{len(tickers)}] {t}: EMPTY", flush=True)
            continue
        df.to_csv(out, index=False)
        print(f"  [{i+1}/{len(tickers)}] {t}: {len(df)} rows", flush=True)


def pull_5min():
    # limit to key ETFs to keep data manageable
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "DIA", "XLF", "XLK", "VIX"]
    start = "2016-01-04T09:30:00-05:00"
    end = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
    print(f"5MIN: {len(tickers)} tickers from 2016-01-04 to {end[:10]}", flush=True)
    for i, t in enumerate(tickers):
        out = OUT_5MIN / f"{t}.csv"
        if out.exists():
            continue
        bars = fetch_bars(t, "5Min", start, end)
        df = bars_to_df(bars, tz_convert=True)
        if df.empty:
            print(f"  [{i+1}/{len(tickers)}] {t}: EMPTY", flush=True)
            continue
        # Filter to market hours 9:30–16:00
        df = df[(df["ts"].dt.time >= pd.Timestamp("09:30").time()) &
                (df["ts"].dt.time < pd.Timestamp("16:00").time())]
        df.to_csv(out, index=False)
        print(f"  [{i+1}/{len(tickers)}] {t}: {len(df)} rows", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if mode == "daily":
        pull_daily()
    elif mode == "intraday":
        pull_5min()
    else:
        sys.exit(f"Unknown mode {mode!r}")
