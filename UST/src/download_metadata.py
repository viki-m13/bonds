#!/usr/bin/env python3
"""Download security-level metadata for all marketable Treasuries.

Source: TreasuryDirect securities API (no key required):
    https://www.treasurydirect.gov/TA_WS/securities/search?format=json

Each record is one auction (reopenings appear as separate records for the
same CUSIP). We keep the fields needed to build accrued-interest schedules
and to classify securities: cusip, security type/term, issue date, dated
date, maturity date, coupon rate, TIPS/FRN flags.

Output: data/raw/securities_meta.json (raw API response, list of dicts)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "securities_meta.json"

URL = "https://www.treasurydirect.gov/TA_WS/securities/search"


TYPES = ["Bill", "Note", "Bond", "TIPS", "FRN", "CMB"]


def main() -> int:
    session = requests.Session()
    records = []
    for sec_type in TYPES:
        for attempt in range(5):
            try:
                r = session.get(
                    URL, params={"format": "json", "type": sec_type}, timeout=300
                )
                r.raise_for_status()
                page = r.json()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"retry {sec_type}: {e}", file=sys.stderr)
                time.sleep(2 ** (attempt + 1))
        records.extend(page)
        print(f"{sec_type}: {len(page)} records (total {len(records)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records))
    print(f"wrote {len(records)} auction records -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
