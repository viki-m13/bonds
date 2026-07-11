"""Generate today's actionable dislocation-reversion candidates.

A bond is a live candidate on the last data date if:
  * it is liquidity-eligible (>=8 active trading days in the trailing 90),
  * its most recent customer-buy print is >= DISCOUNT points below its
    trailing 60-day median mid (the entry signal), and
  * it printed recently (within RECENT_DAYS) so the level is actionable.

Output: munis/research/current_picks.csv, ranked by discount depth.
These are buy-and-hold-~1yr candidates, not day trades.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DISCOUNT = 3.0
RECENT_DAYS = 20


def main() -> None:
    panel = pd.read_parquet(ROOT / "data" / "panel_daily.parquet")
    uni = (pd.read_csv(ROOT / "data" / "universe" / "universe.csv.gz")
           .drop_duplicates("six").set_index("six"))
    bonds = bt.prepare(panel, uni["coupon"])
    asof = panel["date"].max()
    rows = []
    for six, g in bonds.items():
        gi = g.set_index("date")
        med = gi["mid"].rolling("60D", min_periods=5).median().shift(1)
        last = g.iloc[-1]
        if (asof - last["date"]).days > RECENT_DAYS:
            continue
        if not bool(last["eligible"]):
            continue
        s_px = last["s_px"]
        m = med.iloc[-1]
        if pd.isna(s_px) or pd.isna(m):
            continue
        disc = float(s_px - m)
        if disc > -DISCOUNT:
            continue
        u = uni.loc[six] if six in uni.index else None
        rows.append({
            "six": six,
            "desc": (u["desc"][:70] if u is not None else ""),
            "state": u["state"] if u is not None else "",
            "maturity": u["maturity"] if u is not None else "",
            "coupon": u["coupon"] if u is not None else np.nan,
            "last_date": last["date"].date(),
            "last_buy_px": round(float(s_px), 3),
            "trailing_med": round(float(m), 3),
            "discount_pts": round(disc, 2),
            "last_ytw": round(float(last["ytw"]), 3)
            if not pd.isna(last["ytw"]) else np.nan,
        })
    cols = ["six", "desc", "state", "maturity", "coupon", "last_date",
            "last_buy_px", "trailing_med", "discount_pts", "last_ytw"]
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out = out.sort_values("discount_pts")
    out.to_csv(ROOT / "research" / "current_picks.csv", index=False)
    print(f"as of {asof.date()}: {len(out)} live candidates "
          f"(>= {DISCOUNT}pt below trailing median, printed in last "
          f"{RECENT_DAYS}d, liquidity-eligible)")
    if len(out):
        print(out.head(25).to_string(index=False))
    else:
        print("No bonds are currently trading at a >=3pt idiosyncratic "
              "discount — expected outside a selloff. The screen is "
              "live-runnable daily; it lights up when forced selling hits.")


if __name__ == "__main__":
    main()
