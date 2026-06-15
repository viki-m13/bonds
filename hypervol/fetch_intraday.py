"""HYPERVOL — intraday (1h) data layer.

Hyperliquid only retains ~5000 1h candles (~208 days), so the intraday study
covers ~2025-11 -> present. That is exactly the *current, crowded, low-funding*
regime — the right window to (a) measure the carry's true intraday basis/de-peg
risk that the daily-close model cannot see, and (b) assess what the trade earns
NOW rather than in its 2023-24 heyday.

Pulls, all on an aligned UTC hourly grid:
  * HL 1h perp candles                      (the short leg)
  * Binance.US 1h spot klines               (the long leg; binance.com is geo-blocked)
  * HL raw hourly funding                   (credited to the short each hour)
"""
from __future__ import annotations

import json
import time as _time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/user/bonds/data/hypervol")
OUT.mkdir(parents=True, exist_ok=True)

HL_INFO = "https://api.hyperliquid.xyz/info"
BINANCE_US = "https://api.binance.us/api/v3/klines"
HOUR_MS = 3_600_000
# Liquid HL perps that also have a Binance.US USDT spot pair (for the long leg).
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC",
         "BCH", "DOT", "UNI", "AAVE", "ATOM", "NEAR", "APT", "ARB", "INJ"]
SPOT_SYM = {c: f"{c}USDT" for c in COINS}


def _post(payload: dict, retries: int = 5):
    body = json.dumps(payload).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(HL_INFO, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == retries - 1:
                raise
            _time.sleep(2 ** i)


def _get(url: str, retries: int = 5):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == retries - 1:
                raise
            _time.sleep(2 ** i)


def now_ms() -> int:
    return int(_time.time() * 1000)


def fetch_perp_1h(coin: str) -> pd.DataFrame:
    start = now_ms() - 5000 * HOUR_MS
    d = _post({"type": "candleSnapshot", "req": {
        "coin": coin, "interval": "1h", "startTime": start, "endTime": now_ms()}})
    df = pd.DataFrame(d)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None).dt.floor("h")
    df["perp"] = df["c"].astype(float)
    return df.drop_duplicates("ts").set_index("ts")[["perp"]].sort_index()


def fetch_spot_1h(coin: str, start_ms: int) -> pd.DataFrame:
    rows, start = [], start_ms
    end = now_ms()
    while start < end:
        url = (f"{BINANCE_US}?symbol={SPOT_SYM[coin]}&interval=1h"
               f"&startTime={start}&endTime={end}&limit=1000")
        d = _get(url)
        if not d:
            break
        rows.extend(d)
        last = d[-1][0]
        if last <= start:
            break
        start = last + HOUR_MS
        if len(d) < 1000:
            break
        _time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "ct",
                                     "qv", "n", "tb", "tq", "ig"])
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None).dt.floor("h")
    df["spot"] = df["c"].astype(float)
    return df.drop_duplicates("ts").set_index("ts")[["spot"]].sort_index()


def fetch_funding_1h(coin: str, start_ms: int) -> pd.DataFrame:
    rows, start = [], start_ms
    end = now_ms()
    while start < end:
        chunk = _post({"type": "fundingHistory", "coin": coin,
                       "startTime": start, "endTime": end})
        if not chunk:
            break
        rows.extend(chunk)
        last = chunk[-1]["time"]
        if last <= start:
            break
        start = last + 1
        if len(chunk) < 500:
            break
        _time.sleep(0.05)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(None).dt.floor("h")
    df["funding"] = df["fundingRate"].astype(float)
    df["premium"] = df["premium"].astype(float)
    return df.drop_duplicates("ts").set_index("ts")[["funding", "premium"]].sort_index()


def build() -> None:
    ok = []
    for c in COINS:
        try:
            perp = fetch_perp_1h(c)
            if perp.empty:
                print(f"  {c}: no HL perp candles, skip"); continue
            start_ms = int(perp.index[0].timestamp() * 1000)
            spot = fetch_spot_1h(c, start_ms)
            fund = fetch_funding_1h(c, start_ms)
            if spot.empty:
                print(f"  {c}: no Binance.US spot, skip"); continue
            df = perp.join(spot, how="inner").join(fund, how="left")
            if len(df) < 1000:
                print(f"  {c}: only {len(df)} aligned hrs, skip"); continue
            df.index.name = "ts"
            df.to_parquet(OUT / f"intraday_{c}.parquet")
            ok.append(c)
            print(f"  {c}: {len(df)} hrs  {df.index[0].date()} -> {df.index[-1].date()}  "
                  f"basis std={(df['perp']/df['spot']-1).std():.3%}  "
                  f"ann funding={df['funding'].mean()*HOUR_MS/HOUR_MS*24*365:+.1%}")
        except Exception as e:
            print(f"  {c}: ERROR {e}, skip")
    print(f"Done -> {OUT}   ({len(ok)} coins: {ok})")


if __name__ == "__main__":
    build()
