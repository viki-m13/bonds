"""Partition the strategy-ready OSBAP panel into per-year parquet files under
corps/data/panel/ so every file is < 100 MB and the full dataset lives in the
repo (GitHub's per-file limit). Rebuild with build_osbap_panel / osbap raw.

  python corps/scripts/partition_panel.py [SRC_PANEL]
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "panel"


def main(src):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(src)
    p["year"] = pd.to_datetime(p["date"]).dt.year
    for yr, g in p.groupby("year"):
        out = OUTDIR / f"osbap_{yr}.parquet"
        g.drop(columns=["year"]).to_parquet(out, compression="zstd")
        print(f"{yr}: {len(g):,} rows -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "panel_osbap_full.parquet")
