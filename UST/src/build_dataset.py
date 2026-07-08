#!/usr/bin/env python3
"""Build the analysis panel from raw FedInvest files + TreasuryDirect metadata.

Output: data/processed/panel.parquet with one row per (date, cusip):
    date, cusip, sec_type, rate, maturity, buy, sell, eod (clean prices per 100),
    accrued, dirty, coupon_paid, ret (daily total return), ytm, mod_dur,
    tsy_years (time to maturity in years), orig_term_years, issue_date, age_years,
    spread_pct (buy-sell as % of mid, a realistic round-trip transaction cost)

Universe: nominal fixed-coupon securities only (MARKET BASED BILL / NOTE /
BOND). TIPS (need CPI index ratios) and FRNs (floating coupons) are excluded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bondmath

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "fedinvest"
META = ROOT / "data" / "raw" / "securities_meta.json"
OUT = ROOT / "data" / "processed" / "panel.parquet"

COLS = ["cusip", "sec_type", "rate", "maturity", "call_date", "buy", "sell", "eod"]
KEEP_TYPES = {"MARKET BASED BILL", "MARKET BASED NOTE", "MARKET BASED BOND"}


def load_raw() -> pd.DataFrame:
    frames = []
    files = sorted(RAW.glob("*/*.csv"))
    if not files:
        raise SystemExit("no raw files; run download_fedinvest.py first")
    for f in files:
        df = pd.read_csv(f, header=None, names=COLS, dtype={"cusip": str})
        df["date"] = f.stem
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["sec_type"] = panel["sec_type"].str.strip()
    panel = panel[panel["sec_type"].isin(KEEP_TYPES)].copy()
    panel["maturity"] = pd.to_datetime(panel["maturity"], format="%m/%d/%Y")
    for c in ("buy", "sell", "eod", "rate"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    # zero prices mean "not available"
    for c in ("buy", "sell", "eod"):
        panel.loc[panel[c] <= 1.0, c] = np.nan
    panel = panel.dropna(subset=["eod"])
    panel = panel.drop_duplicates(subset=["date", "cusip"], keep="first")
    return panel.drop(columns=["call_date"])


def parse_term_years(term: str) -> float:
    """'9-Year 10-Month' -> 9.83; '26-Week' -> 0.5; '30-Year' -> 30."""
    if not isinstance(term, str) or not term:
        return np.nan
    years = 0.0
    for part in term.split():
        try:
            num, unit = part.split("-", 1)
            num = float(num)
        except ValueError:
            continue
        unit = unit.lower()
        if unit.startswith("year"):
            years += num
        elif unit.startswith("month"):
            years += num / 12.0
        elif unit.startswith("week"):
            years += num * 7 / 365.25
        elif unit.startswith("day"):
            years += num / 365.25
    return years if years > 0 else np.nan


def load_meta() -> pd.DataFrame:
    recs = json.loads(META.read_text())
    m = pd.DataFrame.from_records(
        recs,
        columns=["cusip", "issueDate", "maturityDate", "originalSecurityTerm",
                 "securityTerm", "securityType"],
    )
    m["issueDate"] = pd.to_datetime(m["issueDate"])
    term = m["originalSecurityTerm"].where(
        m["originalSecurityTerm"].astype(bool), m["securityTerm"]
    )
    m["orig_term_years"] = term.map(parse_term_years)
    # one row per cusip: original issue date, original term
    g = m.sort_values("issueDate").groupby("cusip", as_index=False).agg(
        issue_date=("issueDate", "min"),
        orig_term_years=("orig_term_years", "max"),
    )
    return g


def add_bond_math(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["cusip", "date"]).reset_index(drop=True)
    dates = panel["date"].values.astype("datetime64[D]")
    mats = panel["maturity"].values.astype("datetime64[D]")
    rates = panel["rate"].values.astype(float)

    accrued = np.empty(len(panel))
    ytm = np.empty(len(panel))
    mod_dur = np.empty(len(panel))
    dirty = panel["eod"].values.astype(float).copy()

    chunk = 200_000
    for lo in range(0, len(panel), chunk):
        hi = min(lo + chunk, len(panel))
        acc, times, cfs = bondmath.accrued_and_times(dates[lo:hi], mats[lo:hi], rates[lo:hi])
        accrued[lo:hi] = acc
        d = panel["eod"].values[lo:hi] + acc
        y, md = bondmath.solve_ytm(d, times, cfs)
        ytm[lo:hi] = y
        mod_dur[lo:hi] = md
        dirty[lo:hi] = d
        print(f"  bond math {hi}/{len(panel)}", flush=True)

    panel["accrued"] = accrued
    panel["dirty"] = dirty
    panel["ytm"] = ytm
    panel["mod_dur"] = mod_dur
    panel["tsy_years"] = (mats - dates).astype(int) / 365.25
    return panel


def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily total return per cusip, coupon-adjusted."""
    panel = panel.sort_values(["cusip", "date"]).reset_index(drop=True)
    g = panel.groupby("cusip", sort=False)
    prev_date = g["date"].shift(1)
    prev_dirty = g["dirty"].shift(1)
    prev_accr = g["accrued"].shift(1)

    # coupon received in (prev_date, date] <=> accrued dropped and it's a coupon bond
    accr_drop = (panel["accrued"] < prev_accr - 1e-9) & (panel["rate"] > 0)
    coupon_paid = np.where(accr_drop, panel["rate"] / 2.0, 0.0)
    panel["coupon_paid"] = coupon_paid

    gap_days = (panel["date"] - prev_date).dt.days
    ret = (panel["dirty"] + coupon_paid) / prev_dirty - 1.0
    # invalidate returns across gaps > 10 calendar days (stale bridge)
    ret[gap_days > 10] = np.nan
    panel["ret"] = ret
    panel["gap_days"] = gap_days

    mid = (panel["buy"] + panel["sell"]) / 2.0
    panel["spread_pct"] = (panel["buy"] - panel["sell"]) / mid * 100.0
    return panel


def main() -> int:
    print("loading raw files...")
    panel = load_raw()
    print(f"  {len(panel):,} rows, {panel['cusip'].nunique():,} cusips, "
          f"{panel['date'].nunique():,} dates "
          f"({panel['date'].min().date()} .. {panel['date'].max().date()})")

    meta = load_meta()
    panel = panel.merge(meta, on="cusip", how="left")
    print(f"  metadata matched for {panel['issue_date'].notna().mean():.1%} of rows")

    print("computing accrued/ytm/duration...")
    panel = add_bond_math(panel)
    print("computing returns...")
    panel = add_returns(panel)
    panel["age_years"] = (panel["date"] - panel["issue_date"]).dt.days / 365.25

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
