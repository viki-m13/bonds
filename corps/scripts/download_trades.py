"""Download corporate-bond trade histories from FINRA TRACE.

Two modes:
  build-universe : scan a recent window of traceCorporateBond, aggregate by
                   CUSIP, and write the liquid universe.
  download       : pull the full per-CUSIP trade history for the top-N liquid
                   bonds into corps/data/trades/{cusip}.csv.gz.

Requires FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET (free registration).
Without them the script explains what's needed and exits.

Usage:
  python corps/scripts/download_trades.py build-universe [YYYY-MM-DD YYYY-MM-DD]
  python corps/scripts/download_trades.py download [TOP_N]

Field mapping follows FINRA's documented TRACE schema; the normalizer
`_normalize()` is the single place to adjust if the live schema differs.
"""

from __future__ import annotations

import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finra_client import FinraClient, FinraError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUP = "fixedIncomeMarket"
DATASET = "traceCorporateBond"
UNIVERSE = ROOT / "data" / "universe" / "universe.csv.gz"
TRADES = ROOT / "data" / "trades"

# TRACE column candidates (FINRA API camelCase); first present wins.
COLS = {
    "cusip": ["cusip", "cusipIdentifier", "cusip9"],
    "date": ["tradeExecutionDate", "tradeReportDate", "executionDate"],
    "price": ["lastSalePrice", "reportedPrice", "price", "dollarPrice"],
    "yield": ["yieldPercent", "lastSaleYield", "yield"],
    "par": ["reportedTradeVolume", "entrantVolume", "quantity",
            "reportedVolume"],
    "rpt_side": ["reportingPartySide", "buySellIndicator", "reportSideCode",
                 "sideCode"],
    "contra": ["contraPartyType", "contraPartyIndicator", "contraPartyId"],
    "rpt_party": ["reportingPartyType", "reportingPartyId"],
    "symbol": ["issueSymbolIdentifier", "symbol", "ficoSymbol"],
}


def _pick(row: dict, key: str):
    for c in COLS[key]:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return None


def _side(row: dict) -> str:
    """Map TRACE reporting side + contra to EMMA-style S/P/D.

    Convention (matches the muni pipeline so the shared engine works):
      S = customer BUY  (dealer sold to a customer)
      P = customer SELL (dealer bought from a customer)
      D = inter-dealer
    TRACE reports from the dealer's perspective: reporting-side 'S' means the
    reporting dealer sold; if the contra is a customer that is a customer buy.
    """
    contra = (_pick(row, "contra") or "").upper()
    rpt = (_pick(row, "rpt_side") or "").upper()
    if contra.startswith("D"):
        return "D"
    if contra.startswith("C"):
        # dealer sold to customer -> customer buy (S); dealer bought -> P
        if rpt.startswith("S"):
            return "S"
        if rpt.startswith("B"):
            return "P"
    return "D"


def _normalize(row: dict) -> dict | None:
    cusip = _pick(row, "cusip")
    date = _pick(row, "date")
    price = _pick(row, "price")
    if not (cusip and date and price):
        return None
    try:
        px = float(price)
    except ValueError:
        return None
    par = _pick(row, "par")
    try:
        par = float(str(par).replace(",", "").replace("MM+", "000000")) if par else None
    except ValueError:
        par = None
    y = _pick(row, "yield")
    try:
        y = float(y) if y else None
    except ValueError:
        y = None
    return {"cusip": cusip, "date": date[:10], "price": px, "ytw": y,
            "par": par, "side": _side(row)}


def build_universe(client: FinraClient, lo: str, hi: str) -> None:
    print(f"scanning TRACE {lo}..{hi} for the corporate universe", flush=True)
    counts: Counter = Counter()
    vol: Counter = Counter()
    desc: dict[str, str] = {}
    filt = {"dateRangeFilters": [
        {"fieldName": COLS["date"][0], "startDate": lo, "endDate": hi}]}
    n = 0
    for row in client.iter_dataset(GROUP, DATASET, filters=filt, page=5000):
        nr = _normalize(row)
        if not nr:
            continue
        counts[nr["cusip"]] += 1
        if nr["par"]:
            vol[nr["cusip"]] += nr["par"]
        desc.setdefault(nr["cusip"], _pick(row, "symbol") or "")
        n += 1
        if n % 100000 == 0:
            print(f"  {n:,} trades, {len(counts):,} cusips", flush=True)
    UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(UNIVERSE, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cusip", "symbol", "trades_win", "volume_win"])
        for cusip, c in counts.most_common():
            w.writerow([cusip, desc.get(cusip, ""), c, int(vol.get(cusip, 0))])
    print(f"universe: {len(counts):,} cusips -> {UNIVERSE}", flush=True)


def download(client: FinraClient, top_n: int) -> None:
    with gzip.open(UNIVERSE, "rt") as fh:
        uni = list(csv.DictReader(fh))
    targets = uni[:top_n]
    TRADES.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(targets)} of {len(uni)} cusips", flush=True)
    for i, u in enumerate(targets):
        cusip = u["cusip"]
        out = TRADES / f"{cusip}.csv.gz"
        if out.exists():
            continue
        filt = {"compareFilters": [
            {"fieldName": COLS["cusip"][0], "compareType": "EQUAL",
             "fieldValue": cusip}]}
        rows = []
        for row in client.iter_dataset(GROUP, DATASET, filters=filt, page=5000):
            nr = _normalize(row)
            if nr:
                rows.append(nr)
        rows.sort(key=lambda r: r["date"])
        with gzip.open(out, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "price", "ytw", "par", "side"])
            for r in rows:
                w.writerow([r["date"], r["price"], r["ytw"], r["par"], r["side"]])
        if i % 25 == 0:
            print(f"[{i}/{len(targets)}] {cusip}: {len(rows)} trades", flush=True)
    print("done", flush=True)


def main() -> None:
    client = FinraClient()
    if not client.authenticated:
        print(__doc__)
        print("\nSelf-test against the public FINRA endpoint (no creds needed):")
        try:
            print(client.selftest())
        except FinraError as e:
            print("  selftest failed:", e)
        sys.exit("\nSet FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET to run.")
    mode = sys.argv[1] if len(sys.argv) > 1 else "build-universe"
    if mode == "build-universe":
        lo = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
        hi = sys.argv[3] if len(sys.argv) > 3 else "2025-12-31"
        build_universe(client, lo, hi)
    elif mode == "download":
        download(client, int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
