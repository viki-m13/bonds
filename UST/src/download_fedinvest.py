#!/usr/bin/env python3
"""Download daily per-CUSIP end-of-day Treasury prices from FedInvest.

Source: https://www.treasurydirect.gov/GA-FI/FedInvest/securityPriceDetail
The Bureau of the Fiscal Service publishes, for every outstanding marketable
Treasury security (bills, notes, bonds, TIPS, FRNs), a daily table of
CUSIP-level prices: BUY, SELL and END OF DAY price per $100 par.

One POST per business day returns a headerless CSV with columns:
    CUSIP, SECURITY TYPE, RATE, MATURITY DATE, CALL DATE, BUY, SELL, END OF DAY

Files are cached under data/raw/fedinvest/YYYY/YYYY-MM-DD.csv.
Empty responses (holidays) are recorded in data/raw/fedinvest/empty_dates.txt
so they are not re-requested on subsequent runs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import sys
import threading
import time
from pathlib import Path

import requests

URL = "https://www.treasurydirect.gov/GA-FI/FedInvest/securityPriceDetail"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "fedinvest"

_lock = threading.Lock()
_counts = {"ok": 0, "empty": 0, "fail": 0, "cached": 0}


def fetch_date(session: requests.Session, d: dt.date, retries: int = 4) -> str:
    payload = {
        "priceDateDay": str(d.day),
        "priceDateMonth": str(d.month),
        "priceDateYear": str(d.year),
        "fileType": "csv",
        "csv": "CSV FORMAT",
    }
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            r = session.post(URL, data=payload, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def load_empty_dates() -> set[str]:
    f = RAW / "empty_dates.txt"
    if f.exists():
        return set(f.read_text().split())
    return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="default: today")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    RAW.mkdir(parents=True, exist_ok=True)
    empty_dates = load_empty_dates()

    dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # weekdays only
            iso = d.isoformat()
            out = RAW / str(d.year) / f"{iso}.csv"
            if out.exists():
                _counts["cached"] += 1
            elif iso in empty_dates:
                _counts["cached"] += 1
            else:
                dates.append(d)
        d += dt.timedelta(days=1)

    print(f"{len(dates)} dates to fetch ({_counts['cached']} already cached)")
    new_empty: list[str] = []

    def work(d: dt.date) -> None:
        session = _tls.session if hasattr(_tls, "session") else None
        if session is None:
            session = requests.Session()
            _tls.session = session
        try:
            text = fetch_date(session, d)
        except Exception as e:
            with _lock:
                _counts["fail"] += 1
                print(f"FAIL {d}: {e}", file=sys.stderr)
            return
        text = text.strip()
        with _lock:
            if not text:
                _counts["empty"] += 1
                new_empty.append(d.isoformat())
            else:
                out = RAW / str(d.year) / f"{d.isoformat()}.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text + "\n")
                _counts["ok"] += 1
            done = _counts["ok"] + _counts["empty"] + _counts["fail"]
            if done % 200 == 0:
                print(f"  progress: {done}/{len(dates)} {_counts}", flush=True)

    _tls = threading.local()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, dates))

    if new_empty:
        f = RAW / "empty_dates.txt"
        all_empty = sorted(load_empty_dates() | set(new_empty))
        f.write_text("\n".join(all_empty) + "\n")

    print(f"done: {_counts}")
    return 1 if _counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
