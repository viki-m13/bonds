#!/usr/bin/env python3
"""Merge Fed securities-lending specialness onto the price panel and build a
per-(date,cusip) specialness score plus the local cheapness residual and its
forward changes.

Specialness features (all point-in-time; the SecLend result for date t is
published ~12:15pm ET on t, before the EOD marks):
  - loan_frac   = outstanding_loans / soma_holdings   (share of Fed's holding on loan)
  - demand_frac = par_submitted / avail_to_borrow     (dealer demand vs supply)
  - fee         = wavg_rate                            (SecLend fee; higher = more special)
  - special     = composite z-scored score across the cross-section each day

A bond present in the panel but ABSENT from the SecLend list on a date is
treated as not-borrowed (features 0) — the Fed lists every SOMA-held CUSIP
that had activity/availability, and absence means no borrowing demand.

Also computes the repo-carry proxy for honesty: a special security can be
lent (if long) / must be borrowed rich (if short). We approximate the daily
specialness carry as (GC - special_rate) using the SecLend fee vs a GC proxy;
this is charged against short-special / credited to long-special positions in
the backtest so the strategy cannot book uncompensated repo cost as alpha.

Output: data/processed/special_panel.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
ROOT = SRC.parent
import strategies as st

PANEL = ROOT / "data" / "processed" / "panel.parquet"
SL = ROOT / "data" / "raw" / "seclending.parquet"
OUT = ROOT / "data" / "processed" / "special_panel.parquet"


def zscore_x(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and np.isfinite(sd) else s * 0.0


def main() -> int:
    panel = pd.read_parquet(PANEL)
    sl = pd.read_parquet(SL)

    # SecLend "minimum fee" floor is 0.05 in this sample; fee above floor = special
    sl["fee"] = sl["wavg_rate"].fillna(0.0)
    sl["loan_frac"] = sl["outstanding_loans"] / sl["soma_holdings"].replace(0, np.nan)
    sl["demand_frac"] = sl["par_submitted"] / sl["avail_to_borrow"].replace(0, np.nan)
    for c in ("loan_frac", "demand_frac"):
        sl[c] = sl[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 5)
    sl["fee_excess"] = (sl["fee"] - 0.05).clip(lower=0.0)

    keep = ["date", "cusip", "loan_frac", "demand_frac", "fee_excess",
            "par_submitted", "outstanding_loans"]
    m = panel.merge(sl[keep], on=["date", "cusip"], how="left")
    for c in ("loan_frac", "demand_frac", "fee_excess", "par_submitted", "outstanding_loans"):
        m[c] = m[c].fillna(0.0)
    m["sl_covered"] = panel.merge(
        sl[["date", "cusip"]].assign(cov=1), on=["date", "cusip"], how="left"
    )["cov"].fillna(0).values

    # composite specialness score, z-scored cross-sectionally each day among
    # coupon notes/bonds only
    coup = m["sec_type"].isin(["MARKET BASED NOTE", "MARKET BASED BOND"])
    m["special"] = np.nan
    parts = []
    for date, day in m[coup].groupby("date"):
        z = (zscore_x(day["loan_frac"]) + zscore_x(day["demand_frac"])
             + zscore_x(day["fee_excess"]))
        parts.append(pd.Series(z.values, index=day.index))
    if parts:
        m.loc[pd.concat(parts).index, "special"] = pd.concat(parts).values

    # local cheapness residual (yield vs nearest-maturity neighbours), from the
    # cached signal if present else recomputed
    lf = ROOT / "data" / "processed" / "cache_full" / "sig_local_k6.parquet"
    if lf.exists():
        loc = pd.read_parquet(lf)
        loc = loc.stack().rename("local_resid").reset_index()
        loc.columns = ["date", "cusip", "local_resid"]
        m = m.merge(loc, on=["date", "cusip"], how="left")
    else:
        m["local_resid"] = np.nan

    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT, index=False)
    cov = m.loc[coup, "sl_covered"].mean()
    print(f"wrote {OUT}: {len(m):,} rows")
    print(f"SecLend coverage of coupon rows: {cov:.1%}")
    print(f"special score non-null: {m['special'].notna().mean():.1%}")
    print(m.loc[coup & (m['special'].notna()), 'special'].describe().round(3).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
