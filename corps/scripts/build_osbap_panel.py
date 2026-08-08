"""Build a KEYSTONE-compatible daily panel from the Open Source Bond Asset
Pricing daily TRACE panel (free, ~22y, 30M bond-days, 74k corporate bonds).

Source (free direct download, no WRDS):
  https://openbondassetpricing.com/  ->  stage1_osbap_0k_volume_2025.parquet

Maps the OSBAP schema to the same convention the muni engine uses so the
proven backtest runs unchanged:
  six   = cusip_id
  mid   = pr        (daily volume-weighted CLEAN price — signal reference)
  s_px  = prc_ask   (customer buys at the ask)
  p_px  = prc_bid   (customer sells at the bid)
  ytw   = ytm*100
  coupon proxy = ytm*100 (carry; nets out of the matched-control excess)

Liquidity: keep bonds with >= MIN_DAYS bond-days, capped to TOP_BONDS most
active to keep the pure-Python engine tractable (muni run was ~3k bonds).

Usage:  python corps/scripts/build_osbap_panel.py [SRC_PARQUET]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "panel_osbap.parquet"
MIN_DAYS = 500
TOP_BONDS = 8000
COLS = ["cusip_id", "trd_exctn_dt", "pr", "prc_bid", "prc_ask", "ytm",
        "credit_spread", "qvolume", "bond_maturity"]


def main(src: str) -> None:
    pf = pq.ParquetFile(src)
    dc_path = ROOT.parent / "corps" / "data" / "_daycounts.pkl"
    scratch = Path("/tmp/claude-0/-home-user-bonds/"
                   "a180e7e5-cce4-5262-b48d-9e4e7ffdb6d1/scratchpad/"
                   "osbap_daycounts.pkl")
    counts = pickle.load(open(scratch, "rb")) if scratch.exists() else None
    if counts is None:
        import collections
        counts = collections.Counter()
        for i in range(pf.num_row_groups):
            counts.update(pf.read_row_group(i, columns=["cusip_id"])
                          .to_pandas()["cusip_id"].value_counts().to_dict())
    ser = pd.Series(counts)
    liquid = ser[ser >= MIN_DAYS].sort_values(ascending=False)
    keep = set(liquid.head(TOP_BONDS).index)
    print(f"{len(ser)} bonds; {len(liquid)} >= {MIN_DAYS} days; "
          f"keeping top {len(keep)}", flush=True)

    parts = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=COLS).to_pandas()
        df = df[df["cusip_id"].isin(keep)]
        if len(df):
            parts.append(df)
        if i % 10 == 0:
            print(f"  row group {i}/{pf.num_row_groups}", flush=True)
    raw = pd.concat(parts, ignore_index=True)
    del parts

    panel = pd.DataFrame({
        "six": raw["cusip_id"].astype(str),
        "date": pd.to_datetime(raw["trd_exctn_dt"]),
        "mid": raw["pr"].astype(float),
        "s_px": raw["prc_ask"].astype(float),
        "p_px": raw["prc_bid"].astype(float),
        "ytw": (raw["ytm"].astype(float) * 100),
        "cs": raw["credit_spread"].astype(float),
        "qvolume": raw["qvolume"].astype(float),
    })
    # sanity: prices in a plausible band
    panel = panel[(panel["mid"] > 1) & (panel["mid"] < 400)]
    panel = panel.sort_values(["six", "date"]).reset_index(drop=True)
    panel.to_parquet(OUT)
    print(f"panel: {len(panel):,} bond-days, {panel['six'].nunique()} bonds, "
          f"{panel['date'].min().date()}..{panel['date'].max().date()} -> {OUT}",
          flush=True)


if __name__ == "__main__":
    src = (sys.argv[1] if len(sys.argv) > 1 else
           "/tmp/claude-0/-home-user-bonds/"
           "a180e7e5-cce4-5262-b48d-9e4e7ffdb6d1/scratchpad/"
           "stage1_osbap_0k_volume_2025.parquet")
    main(src)
