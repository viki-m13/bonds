#!/usr/bin/env python3
"""Download the Federal Reserve's Securities Lending operation results,
per-CUSIP daily, from the NY Fed markets API.

The Fed lends specific Treasury CUSIPs from the SOMA portfolio to primary
dealers each afternoon. How much of a given CUSIP dealers try to borrow
(parAmtSubmitted) and how much is on loan (outstandingLoans) is a direct,
free, daily measure of that security's SCARCITY / repo specialness — a bond
that is "on special" is expensive to short and tends to be richly priced in
cash. This information is NOT in end-of-day prices, so it is a clean
(non-bid-ask-bounce) candidate signal.

Endpoint (per-CUSIP detail, date-range search):
  https://markets.newyorkfed.org/api/seclending/all/results/details/search.json
Fetched in quarterly chunks and flattened to one row per (date, cusip).

Output: data/raw/seclending.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "seclending.parquet"
URL = "https://markets.newyorkfed.org/api/seclending/all/results/details/search.json"


def fetch(start: str, end: str) -> list[dict]:
    for attempt in range(5):
        try:
            r = requests.get(URL, params={"startDate": start, "endDate": end}, timeout=120)
            r.raise_for_status()
            return r.json()["seclending"]["operations"]
        except Exception as e:
            if attempt == 4:
                print(f"  FAIL {start}..{end}: {e}", file=sys.stderr)
                return []
            time.sleep(2 ** (attempt + 1))
    return []


def main() -> int:
    rows = []
    for year in range(2010, 2027):
        for q in range(4):
            start = f"{year}-{q*3+1:02d}-01"
            end_month = q * 3 + 3
            end = f"{year}-{end_month:02d}-{'31' if end_month in (3,12) else '30'}"
            ops = fetch(start, end)
            n0 = len(rows)
            for op in ops:
                d = op["operationDate"]
                for det in op.get("details", []):
                    rate = det.get("weightedAverageRate")
                    try:
                        rate = float(rate)
                    except (TypeError, ValueError):
                        rate = None
                    rows.append({
                        "date": d,
                        "cusip": det.get("cusip"),
                        "par_submitted": det.get("parAmtSubmitted") or 0,
                        "par_accepted": det.get("parAmtAccepted") or 0,
                        "wavg_rate": rate,
                        "soma_holdings": det.get("somaHoldings") or 0,
                        "avail_to_borrow": det.get("actualAvailToBorrow") or 0,
                        "outstanding_loans": det.get("outstandingLoans") or 0,
                    })
            print(f"{start}..{end}: {len(ops)} ops, +{len(rows)-n0} rows (total {len(rows)})",
                  flush=True)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["cusip"]).drop_duplicates(["date", "cusip"], keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df):,} rows, {df['cusip'].nunique():,} cusips, "
          f"{df['date'].min().date()}..{df['date'].max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
