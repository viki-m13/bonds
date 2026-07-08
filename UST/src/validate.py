#!/usr/bin/env python3
"""Data-quality validation of the FedInvest panel.

Checks:
 1. Coverage: securities per date by type; no large holes in the calendar.
 2. Price sanity: clean price bounds, spread distribution.
 3. Return sanity: daily total returns vs duration -> flag outliers; the
    duration-implied yield move for a big return day should be plausible.
 4. External cross-check: our computed YTMs for ~2y/~10y/~30y securities
    vs FRED constant-maturity series (DGS2/DGS10/DGS30). These are
    independent sources; median absolute gap should be a few bps.
 5. Coupon capture: average annual coupon income per bond ~= coupon rate.

Writes results/validation_report.txt and returns nonzero on hard failures.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "processed" / "panel.parquet"
RESULTS = ROOT / "results"

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fred_series(sid: str) -> pd.Series:
    r = requests.get(FRED.format(sid=sid), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df[sid], errors="coerce")
    return pd.Series(s.values, index=df["date"], name=sid).dropna()


def main() -> int:
    out = []
    hard_fail = False

    def log(msg: str) -> None:
        print(msg)
        out.append(msg)

    p = pd.read_parquet(PANEL)
    log(f"panel: {len(p):,} rows, {p['cusip'].nunique():,} cusips, "
        f"{p['date'].nunique():,} dates ({p['date'].min().date()}..{p['date'].max().date()})")

    # 1. coverage
    cov = p.groupby(["date", "sec_type"]).size().unstack(fill_value=0)
    log("\n[1] securities per date by type (median):")
    log(str(cov.median().to_dict()))
    dates = pd.Series(sorted(p["date"].unique()))
    gaps = dates.diff().dt.days
    big_gaps = dates[gaps > 5]
    log(f"calendar gaps >5 days: {len(big_gaps)}" +
        (f" e.g. {list(big_gaps.head(5).dt.date)}" if len(big_gaps) else ""))
    if cov.sum(axis=1).min() < 100:
        log("WARN: some dates have <100 securities")

    # 2. price sanity
    bad_px = p[(p["eod"] < 40) | (p["eod"] > 250)]
    log(f"\n[2] clean price outside [40,250]: {len(bad_px)} rows")
    if len(bad_px) > 50:
        hard_fail = True
    sp = p["spread_pct"].dropna()
    log(f"buy-sell spread %% of mid: median={sp.median():.4f} p90={sp.quantile(.9):.4f} "
        f"p99={sp.quantile(.99):.4f} (coverage {p['spread_pct'].notna().mean():.1%})")

    # 3. return sanity: |ret| should be < dur * 50bp + 1% almost always
    r = p.dropna(subset=["ret"])
    lim = r["mod_dur"] * 0.005 + 0.01
    outliers = r[np.abs(r["ret"]) > lim]
    log(f"\n[3] daily |ret| > dur*50bp+1%: {len(outliers)} of {len(r):,} "
        f"({len(outliers)/len(r):.2e})")
    if len(outliers) / len(r) > 0.001:
        hard_fail = True
    log("largest daily returns:")
    top = r.reindex(np.abs(r["ret"]).nlargest(5).index)
    log(str(top[["date", "cusip", "sec_type", "tsy_years", "ret"]].to_string(index=False)))

    # 4. FRED cross-check
    log("\n[4] cross-check computed YTM vs FRED constant-maturity yields:")
    for sid, lo, hi in [("DGS2", 1.9, 2.1), ("DGS10", 9.5, 10.5), ("DGS30", 29.0, 30.0)]:
        try:
            f = fred_series(sid)
        except Exception as e:
            log(f"  {sid}: FETCH FAILED ({e}) - skipped")
            continue
        sub = p[(p["tsy_years"] >= lo) & (p["tsy_years"] <= hi) & (p["rate"] > 0)]
        ours = sub.groupby("date")["ytm"].median() * 100
        both = pd.concat([ours.rename("ours"), f.rename("fred")], axis=1).dropna()
        gap_bp = ((both["ours"] - both["fred"]) * 100).abs()
        log(f"  {sid}: n={len(both):,} median|gap|={gap_bp.median():.1f}bp "
            f"p95={gap_bp.quantile(.95):.1f}bp")
        if gap_bp.median() > 15:
            hard_fail = True

    # 5. coupon capture: total coupons per cusip-year vs rate
    cp = p[p["rate"] > 0].copy()
    cp["year"] = cp["date"].dt.year
    ann = cp.groupby(["cusip", "year"]).agg(
        coup=("coupon_paid", "sum"), rate=("rate", "first"), n=("date", "size"))
    full_years = ann[ann["n"] >= 240]
    ratio = (full_years["coup"] / full_years["rate"]).dropna()
    log(f"\n[5] coupons received per full cusip-year / rate: median={ratio.median():.3f} "
        f"(expect ~1.0), p10={ratio.quantile(.1):.3f}, p90={ratio.quantile(.9):.3f}")
    if not 0.9 < ratio.median() < 1.1:
        hard_fail = True

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation_report.txt").write_text("\n".join(out) + "\n")
    log(f"\n{'HARD FAIL' if hard_fail else 'ALL CHECKS PASSED (see warnings above)'}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
