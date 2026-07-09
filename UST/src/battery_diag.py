#!/usr/bin/env python3
"""IS-only DIAGNOSTICS for non-price structural effects (no bounce risk).

Before building any tradeable strategy, check whether the effect exists at all
in the training window. Each is measured in duration-neutral idiosyncratic
return (daily return residualized on modified duration cross-sectionally), so a
signal is a genuine rich/cheap drift, not a rates move.

1. Month-end index extension: mean idio return of the long-duration tercile by
   trading-day-of-month. If duration demand concentrates at month-end, the long
   end should richen (positive idio) in the last few sessions.
2. Auction roll: when a new on-the-run is issued, the previous on-the-run
   ("1st off-the-run") is said to cheapen. Track idio return of bonds by age
   since becoming off-the-run.
3. Curve curvature (butterfly): does the fitted-curve curvature beta mean-revert
   (autocorrelation of its weekly change)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
ROOT = SRC.parent
from explore_special import duration_neutral_returns

IS_END = pd.Timestamp("2019-12-31")


def main() -> int:
    p = pd.read_parquet(ROOT / "data" / "processed" / "special_panel.parquet")
    p = p[p["sec_type"].isin(["MARKET BASED NOTE", "MARKET BASED BOND"])]
    p = p[(p["tsy_years"] >= 1.0) & (p["date"] <= IS_END)].copy()
    p = p.sort_values(["cusip", "date"]).reset_index(drop=True)
    p = duration_neutral_returns(p)

    # ---- 1. month-end extension ----
    dates = pd.Series(sorted(p["date"].unique()))
    # trading-day-of-month index (1 = first session of month) and days-to-month-end
    dom = {}
    dte = {}
    for _, grp in dates.groupby(dates.dt.to_period("M")):
        gl = list(grp)
        for i, d in enumerate(gl):
            dom[d] = i + 1
            dte[d] = len(gl) - i  # sessions remaining incl. today
    p["dte"] = p["date"].map(dte)
    p["dur_tercile"] = p.groupby("date")["mod_dur"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if s.nunique() >= 3 else np.nan)
    longend = p[p["dur_tercile"] == 2]
    print("[1] month-end extension: long-duration tercile idio return (bp/day) "
          "by sessions-to-month-end")
    g = longend.groupby("dte")["r_idio"].mean() * 1e4
    print(g.reindex(range(1, 8)).round(3).to_string())
    # naive edge: mean idio return in last 3 sessions vs rest
    last3 = longend[longend["dte"] <= 3]["r_idio"].mean() * 1e4
    rest = longend[longend["dte"] > 3]["r_idio"].mean() * 1e4
    print(f"  last<=3 sessions: {last3:.3f} bp/day vs rest: {rest:.3f} bp/day")

    # ---- 2. auction roll ----
    # define on-the-run: youngest bond per original-term bucket each day
    p["term_bucket"] = p["orig_term_years"].round(0)
    def tag_age(day):
        day = day.copy()
        day["is_otr"] = False
        for tb, g2 in day.groupby("term_bucket"):
            if g2["age_years"].notna().any():
                youngest = g2["age_years"].idxmin()
                day.loc[youngest, "is_otr"] = True
        return day
    # rank age within (date, term_bucket): 0 = OTR, 1 = 1st off-the-run, ...
    p["otr_rank"] = p.groupby(["date", "term_bucket"])["age_years"].rank(method="first") - 1
    print("\n[2] auction roll: idio return (bp/day) by on-the-run rank "
          "(0=OTR,1=1st off,...), 2/3/5/7/10y buckets")
    sub = p[p["term_bucket"].isin([2, 3, 5, 7, 10])]
    g2 = sub.groupby("otr_rank")["r_idio"].mean() * 1e4
    print(g2.reindex(range(0, 6)).round(3).to_string())

    # ---- 3. curvature mean reversion ----
    params = pd.read_parquet(ROOT / "data" / "processed" / "cache_full" / "curve_params.parquet")
    params = params[params.index <= IS_END]
    fly = params["b2"]  # NSS first curvature loading
    wk = fly.resample("W-WED").last().dropna()
    dwk = wk.diff().dropna()
    ac1 = dwk.autocorr(1)
    print(f"\n[3] curvature (b2) weekly change autocorr(1) = {ac1:.3f} "
          f"(negative => mean-reverting fly)")
    lvl_ac = wk.diff().dropna().autocorr(1)
    print(f"    curvature level weekly autocorr of level changes = {lvl_ac:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
