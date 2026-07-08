"""Build a daily per-bond panel from the raw EMMA trade prints.

For each (security, date):
  s_px / p_px / d_px : par-weighted mean price of customer-buy (S),
                       customer-sell (P) and inter-dealer (D) prints
  s_n, p_n, d_n      : print counts per side
  s_par, p_par, d_par: total par per side
  mid                : d_px if present, else (s_px+p_px)/2, else whichever
                       side exists (used for signals only, never for fills)
  ytw                : par-weighted mean yield-to-worst across all prints

Fills in the backtest only ever use the actual S (buy) and P (sell) prints —
`mid` is a signal input, not an executable price.

EMMA's chart endpoint timestamps are date-granular, so nothing intraday is
assumed anywhere downstream.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES_DIR = ROOT / "data" / "trades"
PANEL_PATH = ROOT / "data" / "panel_daily.parquet"

# retail-representative print size band (par $)
RETAIL_MIN = 5_000
RETAIL_MAX = 250_000


def _one(path: str) -> pd.DataFrame | None:
    six = Path(path).stem.replace(".csv", "")
    df = pd.read_csv(path, parse_dates=["ts"])
    if df.empty:
        return None
    df["date"] = df["ts"].dt.normalize()
    df = df[df["price"].between(1, 300)]

    out = {}
    for side in ("S", "P", "D"):
        s = df[df["side"] == side]
        if side in ("S", "P"):
            s = s[s["par"].between(RETAIL_MIN, RETAIL_MAX)]
        if s.empty:
            continue
        g = s.groupby("date")
        wpx = g.apply(
            lambda x: np.average(x["price"], weights=x["par"]),
            include_groups=False)
        out[f"{side.lower()}_px"] = wpx
        out[f"{side.lower()}_n"] = g.size()
        out[f"{side.lower()}_par"] = g["par"].sum()

    g_all = df.groupby("date")
    y = df.dropna(subset=["ytw"])
    if len(y):
        out["ytw"] = y.groupby("date").apply(
            lambda x: np.average(x["ytw"], weights=x["par"]),
            include_groups=False)
    out["n_all"] = g_all.size()

    p = pd.DataFrame(out)
    p["mid"] = p.get("d_px")
    if "s_px" in p and "p_px" in p:
        both = p["s_px"].notna() & p["p_px"].notna()
        p.loc[p["mid"].isna() & both, "mid"] = (
            (p["s_px"] + p["p_px"]) / 2)[p["mid"].isna() & both]
    for c in ("s_px", "p_px"):
        if c in p:
            p["mid"] = p["mid"].fillna(p[c])
    p["six"] = six
    return p.reset_index().rename(columns={"index": "date"})


def build() -> pd.DataFrame:
    files = sorted(glob.glob(str(TRADES_DIR / "*.csv.gz")))
    parts = []
    for i, f in enumerate(files):
        p = _one(f)
        if p is not None:
            parts.append(p)
        if i % 200 == 0:
            print(f"panel {i}/{len(files)}", flush=True)
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["six", "date"]).reset_index(drop=True)
    panel.to_parquet(PANEL_PATH)
    print(f"panel: {len(panel):,} bond-days, {panel['six'].nunique()} bonds "
          f"-> {PANEL_PATH}")
    return panel


def load() -> pd.DataFrame:
    return pd.read_parquet(PANEL_PATH)


if __name__ == "__main__":
    build()
