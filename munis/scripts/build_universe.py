"""Build the tradable-muni universe: every security that traded in the last
year, state by state, with per-security liquidity stats.

Writes munis/data/universe/universe.csv.gz with one row per security:
  six, desc, state, maturity, coupon, trades_1y, volume_1y,
  min_px, max_px, min_ytw, max_ytw, scan_date_begin, scan_date_end

Usage:  python munis/scripts/build_universe.py
Resumable: states already present in the partial file are skipped.
"""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emma_client import EmmaClient, EmmaError, STATES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "universe" / "universe.csv.gz"
PARTIAL = ROOT / "data" / "universe" / "_universe_partial.csv"

FIELDS = ["six", "desc", "state", "maturity", "coupon", "trades_1y",
          "volume_1y", "min_px", "max_px", "min_ytw", "max_ytw",
          "scan_begin", "scan_end"]


def _write_rows(writer, res: dict, state: str, seen: set[str]) -> int:
    n = 0
    for r in res["data"]:
        if r["six"] in seen:
            continue
        seen.add(r["six"])
        writer.writerow({
            "six": r["six"],
            "desc": (r.get("sd") or "").replace("\n", " ").strip(),
            "state": r.get("st") or state,
            "maturity": (r.get("md") or "")[:10],
            "coupon": r.get("ir"),
            "trades_1y": r.get("tc"),
            "volume_1y": r.get("tta"),
            "min_px": r.get("mnp"),
            "max_px": r.get("mxp"),
            "min_ytw": r.get("mny"),
            "max_ytw": r.get("mxy"),
            "scan_begin": (res.get("tradeDateBegin") or "")[:10],
            "scan_end": (res.get("tradeDateEnd") or "")[:10],
        })
        n += 1
    return n


def _maturity_seeds() -> list[tuple[str, str]]:
    """(maturity, six) seeds spaced across the maturity spectrum, drawn from
    rows already scanned. Seeds are used only for their maturity center;
    their own state does not constrain the query."""
    with open(PARTIAL) as fh:
        rows = [(r["maturity"], r["six"]) for r in csv.DictReader(fh)
                if r["maturity"]]
    rows.sort()
    if not rows:
        raise RuntimeError("no scanned rows to draw maturity seeds from")
    lo_year = int(rows[0][0][:4])
    hi_year = int(rows[-1][0][:4])
    seeds = []
    # 3-year steps with a +-2y ("2years") band => 1y overlap; fine enough
    # that even the densest states stay under EMMA's result cap.
    y = lo_year
    while y <= hi_year + 2:
        best = min(rows, key=lambda r: abs(
            (int(r[0][:4]) - y) * 12 + int(r[0][5:7]) - 6))
        seeds.append((best[0], best[1]))
        y += 3
    seeds.append((rows[-1][0], rows[-1][1]))
    dedup = {}
    for m, s in seeds:
        dedup[s] = m
    return [(m, s) for s, m in dedup.items()]


def main() -> None:
    client = EmmaClient(ROOT / "data" / "universe" / "_cookies.txt", delay=1.0)
    client.ensure_session()

    # seed: any security works because only the state criterion is selected
    actives = client.most_actively_traded()
    seed = actives[0]["six"]

    done_states: set[str] = set()
    seen: set[str] = set()
    if PARTIAL.exists():
        with open(PARTIAL) as fh:
            for row in csv.DictReader(fh):
                done_states.add(row["state"])
                seen.add(row["six"])
        print(f"resuming, {len(done_states)} states already scanned")

    mode = "a" if PARTIAL.exists() else "w"
    with open(PARTIAL, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for state in STATES:
            if state in done_states:
                continue
            try:
                res = client.find_similar(seed, state)
            except EmmaError as exc:
                print(f"{state}: FAILED ({exc})", flush=True)
                continue
            status = res.get("status")
            if status == "data":
                n = _write_rows(writer, res, state, seen)
                fh.flush()
                print(f"{state}: {n} securities", flush=True)
                continue
            if status != "too-many":
                print(f"{state}: status={status} rows=0", flush=True)
                continue
            # partition the state by maturity tiles (+-5y around each seed)
            print(f"{state}: too-many, partitioning by maturity", flush=True)
            total = 0
            for maturity, mseed in _maturity_seeds():
                try:
                    res = client.find_similar(mseed, state,
                                              maturity_band="2years")
                except EmmaError as exc:
                    print(f"  {state} ~{maturity}: FAILED ({exc})", flush=True)
                    continue
                if res.get("status") != "data":
                    print(f"  {state} ~{maturity}: status={res.get('status')}",
                          flush=True)
                    continue
                n = _write_rows(writer, res, state, seen)
                total += n
                fh.flush()
                print(f"  {state} ~{maturity}: +{n}", flush=True)
            print(f"{state}: {total} securities (partitioned)", flush=True)

    # compress the finished scan
    with open(PARTIAL, "rb") as src, gzip.open(OUT, "wb") as dst:
        dst.write(src.read())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
