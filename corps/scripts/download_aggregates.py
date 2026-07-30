"""Download the FINRA fixed-income AGGREGATE datasets accessible to a basic
API credential (trade-level TRACE requires an upgraded credential).

Accessible with a basic credential:
  corporateMarketBreadth  — daily advances/declines/unchanged, 52wk high/low,
                            trades, volume — by product category
                            (all / investment grade / high yield / convertibles)
  corporateMarketSentiment— daily trades/volume by grade x flow direction
                            (customer buy / customer sell / inter-dealer / ...)
  corporatesAndAgenciesCappedVolume — daily volume by grade/144A

Writes corps/data/aggregates/{dataset}.csv.gz.

Usage:  FINRA_API_CLIENT_ID=... FINRA_API_CLIENT_SECRET=... \
        python corps/scripts/download_aggregates.py
"""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finra_client import FinraClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "aggregates"
DATASETS = ["corporateMarketBreadth", "corporateMarketSentiment",
            "corporatesAndAgenciesCappedVolume"]


def main() -> None:
    c = FinraClient()
    if not c.authenticated:
        sys.exit("set FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET")
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        rows = list(c.iter_dataset("fixedIncomeMarket", ds, page=5000))
        if not rows:
            print(f"{ds}: no rows"); continue
        cols = list(rows[0].keys())
        with gzip.open(OUT / f"{ds}.csv.gz", "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        dcol = next(k for k in cols if "Date" in k or "date" in k)
        ds_dates = sorted(r[dcol] for r in rows)
        print(f"{ds}: {len(rows)} rows, {ds_dates[0]}..{ds_dates[-1]} "
              f"-> {OUT / (ds + '.csv.gz')}")


if __name__ == "__main__":
    main()
