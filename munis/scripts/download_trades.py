"""Download full trade-by-trade history for the most liquid securities in
the universe scan.

Writes munis/data/trades/{six}.csv.gz, one file per security, columns:
  ts (ISO-8601, ET), price, ytw, par, side   (side: D inter-dealer,
  S customer buy / dealer sell, P customer sell / dealer purchase)

Selection: top N securities by trades_1y (default 1200). The strategy layer
applies its own trailing-liquidity gate at each backtest date, so this cut
only bounds download size, it is not the tradable-universe definition.

Usage:  python munis/scripts/download_trades.py [N]
Resumable: existing per-security files are skipped.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emma_client import EmmaClient, EmmaError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "data" / "universe" / "universe.csv.gz"
TRADES_DIR = ROOT / "data" / "trades"
META = ROOT / "data" / "universe" / "download_meta.csv"


def main(top_n: int = 1200) -> None:
    with gzip.open(UNIVERSE, "rt") as fh:
        universe = list(csv.DictReader(fh))
    universe.sort(key=lambda r: -int(r["trades_1y"] or 0))
    targets = universe[:top_n]
    print(f"universe {len(universe)} securities, downloading top {len(targets)}")

    client = EmmaClient(ROOT / "data" / "universe" / "_cookies.txt", delay=0.8)
    client.ensure_session()

    meta_exists = META.exists()
    done = set()
    if meta_exists:
        with open(META) as fh:
            done = {row["six"] for row in csv.DictReader(fh)}

    with open(META, "a", newline="") as mfh:
        mwriter = csv.DictWriter(mfh, fieldnames=["six", "desc", "n_trades",
                                                  "first_trade", "last_trade"])
        if not meta_exists:
            mwriter.writeheader()

        for i, row in enumerate(targets):
            six = row["six"]
            out = TRADES_DIR / f"{six}.csv.gz"
            if out.exists() and six in done:
                continue
            try:
                info = client.security_trade_info(six)
            except EmmaError as exc:
                print(f"[{i}] {six}: FAILED ({exc})", flush=True)
                continue
            trades = info.get("data") or []
            trades.sort(key=lambda t: t["TDT"])
            with gzip.open(out, "wt", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["ts", "price", "ytw", "par", "side"])
                for t in trades:
                    ts = dt.datetime.fromtimestamp(t["TDT"] / 1000,
                                                   dt.timezone.utc)
                    w.writerow([ts.strftime("%Y-%m-%dT%H:%M:%S"),
                                t.get("PX"), t.get("YX"), t.get("TA"),
                                t.get("TT")])
            first = (dt.datetime.fromtimestamp(trades[0]["TDT"] / 1000,
                     dt.timezone.utc).date().isoformat() if trades else "")
            last = (dt.datetime.fromtimestamp(trades[-1]["TDT"] / 1000,
                    dt.timezone.utc).date().isoformat() if trades else "")
            mwriter.writerow({"six": six,
                              "desc": info.get("securityDesc", "")[:120],
                              "n_trades": len(trades),
                              "first_trade": first, "last_trade": last})
            mfh.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(targets)}] {six}: {len(trades)} trades "
                      f"{first}..{last}", flush=True)

    print("done")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    main(n)
