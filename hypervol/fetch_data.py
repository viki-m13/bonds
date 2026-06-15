"""HYPERVOL — data layer.

Pulls the three ingredients needed to port the Concretum VIX-ETN volatility
strategy ("The Volatility Edge") onto Hyperliquid perpetuals:

  1. Hyperliquid PERP prices   (1d candles)        -> the tradeable instrument
  2. Hyperliquid FUNDING rates (hourly -> daily)   -> the perp-native analog of
                                                       the VIX term-structure roll
  3. Deribit DVOL index        (30d implied vol)   -> the crypto "VIX"

Everything is cached under data/hypervol/ so the backtest/validation are
reproducible offline. Network calls retry with backoff.

Honesty notes baked in here:
  * Hyperliquid funding is charged HOURLY (rate is already per-hour). We sum the
    24 hourly prints into a realised daily funding so the backtest can debit/
    credit it on the actual schedule a position would have paid.
  * Perp prices (not Yahoo spot) are used because that is what actually fills on
    HL; the small perp/spot basis is exactly the funding we model separately.
  * DVOL only exists for BTC & ETH on Deribit. Alts therefore run a funding-only
    variant of the strategy (no eVRP gate) — flagged explicitly downstream.
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
DERIBIT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

# HL mainnet funding begins 2023-06-01. DVOL (Deribit) begins 2021-03.
HL_START_MS = 1685577600000          # 2023-06-01 UTC
DVOL_START_MS = 1614556800000        # 2021-03-01 UTC
HOUR_MS = 3_600_000
DAY_MS = 86_400_000

# Core tradeable set. BTC/ETH have DVOL (full strategy); the rest run the
# funding-only variant for breadth/robustness.
DVOL_COINS = ["BTC", "ETH"]
PERP_COINS = ["BTC", "ETH", "SOL"]


def _post(url: str, payload: dict, retries: int = 5) -> object:
    body = json.dumps(payload).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if i == retries - 1:
                raise
            _time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def _get(url: str, retries: int = 5) -> object:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if i == retries - 1:
                raise
            _time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def now_ms() -> int:
    return int(_time.time() * 1000)


# --------------------------------------------------------------------------- #
# Hyperliquid funding (hourly) -> daily realised funding                       #
# --------------------------------------------------------------------------- #
def fetch_funding(coin: str) -> pd.DataFrame:
    """Hourly funding paginated to present, returned as a daily frame.

    Columns:
      funding_day   sum of the 24 hourly funding rates that day (the rate a
                    short would *receive* / a long would *pay* over the day)
      funding_ann   annualised funding (funding_day * 365)
      n_hours       hourly prints that day (sanity / partial-day guard)
    """
    rows = []
    start = HL_START_MS
    end = now_ms()
    while start < end:
        chunk = _post(HL_INFO, {"type": "fundingHistory", "coin": coin,
                                "startTime": start, "endTime": end})
        if not chunk:
            break
        rows.extend(chunk)
        last_t = chunk[-1]["time"]
        if last_t <= start:
            break
        start = last_t + 1
        if len(chunk) < 500:      # last page
            break
        _time.sleep(0.05)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.drop_duplicates(subset="time").set_index("time").sort_index()

    daily = df["fundingRate"].resample("1D").agg(["sum", "count"])
    daily.columns = ["funding_day", "n_hours"]
    daily["funding_ann"] = daily["funding_day"] * 365
    daily.index = daily.index.tz_convert(None).normalize()
    return daily


# --------------------------------------------------------------------------- #
# Hyperliquid perp 1d candles                                                  #
# --------------------------------------------------------------------------- #
def fetch_candles(coin: str) -> pd.DataFrame:
    """Daily perp OHLCV, paginated (API caps ~5000 candles/response)."""
    rows = []
    start = HL_START_MS
    end = now_ms()
    while start < end:
        chunk = _post(HL_INFO, {"type": "candleSnapshot", "req": {
            "coin": coin, "interval": "1d", "startTime": start, "endTime": end}})
        if not chunk:
            break
        rows.extend(chunk)
        last_t = chunk[-1]["t"]
        if last_t <= start:
            break
        start = last_t + DAY_MS
        if len(chunk) < 5000:
            break
        _time.sleep(0.05)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    for c in ("o", "h", "l", "c", "v"):
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset="date").set_index("date").sort_index()
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


# --------------------------------------------------------------------------- #
# Deribit DVOL (the crypto VIX)                                                #
# --------------------------------------------------------------------------- #
def fetch_dvol(currency: str) -> pd.DataFrame:
    """Daily DVOL (30d implied vol, points). Deribit returns OHLC of the index;
    we keep the close as 'iv'. Paginated in <=10000-candle windows."""
    rows = []
    start = DVOL_START_MS
    end = now_ms()
    # Deribit caps at ~1000 bars/response and returns the most-recent 1000 in the
    # window, so we must page forward in sub-1000-day windows.
    while start < end:
        win_end = min(start + 900 * DAY_MS, end)
        url = (f"{DERIBIT}?currency={currency}&start_timestamp={start}"
               f"&end_timestamp={win_end}&resolution=86400")
        res = _get(url)
        data = res.get("result", {}).get("data", [])
        if not data:
            start = win_end + DAY_MS
            continue
        rows.extend(data)
        last_t = data[-1][0]
        start = max(last_t + DAY_MS, win_end + DAY_MS)
        _time.sleep(0.05)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"])
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    df = df.drop_duplicates(subset="date").set_index("date").sort_index()
    df["iv"] = df["c"].astype(float)          # 30d implied vol, in vol points
    return df[["iv"]]


def build_all() -> None:
    print("Fetching Deribit DVOL ...")
    for cur in DVOL_COINS:
        d = fetch_dvol(cur)
        d.to_parquet(OUT / f"dvol_{cur}.parquet")
        print(f"  DVOL {cur}: {len(d)} days  {d.index.min().date()} -> {d.index.max().date()}")

    print("Fetching Hyperliquid perp candles ...")
    for c in PERP_COINS:
        k = fetch_candles(c)
        k.to_parquet(OUT / f"perp_{c}.parquet")
        print(f"  PERP {c}: {len(k)} days  {k.index.min().date()} -> {k.index.max().date()}")

    print("Fetching Hyperliquid funding ...")
    for c in PERP_COINS:
        f = fetch_funding(c)
        f.to_parquet(OUT / f"funding_{c}.parquet")
        ann = f["funding_ann"].mean()
        print(f"  FUND {c}: {len(f)} days  mean ann funding {ann:+.1%}  "
              f"{f.index.min().date()} -> {f.index.max().date()}")

    print("Done. Cached under", OUT)


if __name__ == "__main__":
    build_all()
