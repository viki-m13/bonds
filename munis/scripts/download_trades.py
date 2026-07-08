"""Download full trade-by-trade history for the most liquid securities in
the universe scan.

Writes munis/data/trades/{six}.csv.gz, one file per security, columns:
  ts (ISO-8601, ET), price, ytw, par, side   (side: D inter-dealer,
  S customer buy / dealer sell, P customer sell / dealer purchase)

Selection (bounds download size only — the strategy layer applies its own
trailing-liquidity gate at each backtest date):
  * Tier A: top TOP_N securities by trades_1y (today's most tradable names,
    skewed to recent issues because new issues trade heavily), plus
  * Tier B: SAMPLE_N drawn at random (seeded) from the rest with
    trades_1y >= SAMPLE_MIN — this reaches seasoned bonds with multi-year
    histories that a pure top-N cut would miss.

Usage:  python munis/scripts/download_trades.py [TOP_N] [SAMPLE_N]
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


SAMPLE_MIN = 40   # min trailing-year trades for the random seasoned sample


def select_targets(top_n: int, sample_n: int) -> list[dict]:
    import random
    with gzip.open(UNIVERSE, "rt") as fh:
        universe = list(csv.DictReader(fh))
    universe.sort(key=lambda r: -int(r["trades_1y"] or 0))
    tier_a = universe[:top_n]
    a_ids = {r["six"] for r in tier_a}
    pool = [r for r in universe[top_n:]
            if int(r["trades_1y"] or 0) >= SAMPLE_MIN and r["six"] not in a_ids]
    random.Random(17).shuffle(pool)
    tier_b = pool[:sample_n]
    return tier_a + tier_b


def main(top_n: int = 1000, sample_n: int = 800,
         shard_idx: int = 0, shard_count: int = 1) -> None:
    targets = select_targets(top_n, sample_n)
    if shard_count > 1:
        targets = [t for i, t in enumerate(targets)
                   if i % shard_count == shard_idx]
    meta_path = (META if shard_count == 1
                 else META.with_name(f"download_meta_{shard_idx}.csv"))
    print(f"shard {shard_idx}/{shard_count}: {len(targets)} targets")

    client = EmmaClient(ROOT / "data" / "universe" / "_cookies.txt", delay=0.5)
    client.ensure_session()

    meta_exists = meta_path.exists()
    done = set()
    if meta_exists:
        with open(meta_path) as fh:
            done = {row["six"] for row in csv.DictReader(fh)}

    with open(meta_path, "a", newline="") as mfh:
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
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    samp = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    sidx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    scnt = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    main(top, samp, sidx, scnt)
