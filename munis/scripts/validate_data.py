"""Validate the downloaded EMMA trade data before any backtesting.

Checks
  1. Structural: parseable files, sane ranges (price, yield, par, side),
     sorted timestamps.
  2. Economic: on days with both customer-buy (S) and customer-sell (P)
     prints in the same bond, median S price should exceed median P price
     (customers buy at the ask, sell at the bid). This validates our
     reading of the side codes and measures the effective retail spread.
  3. Price-yield inversion: within a bond, daily price changes and
     yield-to-worst changes must be negatively correlated.
  4. Cross-endpoint: per-security min/max price and yield over the scan
     window recomputed from the trade files must match the independent
     FindSimilarSecurities summary (universe.csv.gz) within tolerance.
  5. Coverage: trades/bond, active days, calendar span, par-size shape.

Writes munis/data/VALIDATION.md with the results.

Usage:  python munis/scripts/validate_data.py
"""

from __future__ import annotations

import glob
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES_DIR = ROOT / "data" / "trades"
UNIVERSE = ROOT / "data" / "universe" / "universe.csv.gz"
REPORT = ROOT / "data" / "VALIDATION.md"


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df["date"] = df["ts"].dt.date
    return df


def main() -> None:
    files = sorted(glob.glob(str(TRADES_DIR / "*.csv.gz")))
    if not files:
        sys.exit("no trade files found — run download_trades.py first")
    uni = pd.read_csv(UNIVERSE)
    uni = uni.set_index("six")
    scan_begin = pd.Timestamp(uni["scan_begin"].iloc[0])
    scan_end = pd.Timestamp(uni["scan_end"].iloc[0])

    structural_fails: list[str] = []
    side_ok_days = 0
    side_bad_days = 0
    spreads: list[float] = []
    py_corrs: list[float] = []
    xcheck_rows: list[dict] = []
    n_trades_total = 0
    n_bonds = 0
    spans: list[int] = []
    active_days: list[int] = []
    par_values: list[np.ndarray] = []
    first_years: list[int] = []

    rng = np.random.default_rng(7)
    xcheck_sample = set(rng.choice(len(files), size=min(150, len(files)),
                                   replace=False).tolist())

    for i, path in enumerate(files):
        six = Path(path).stem.replace(".csv", "")
        try:
            df = load_trades(path)
        except Exception as exc:  # noqa: BLE001
            structural_fails.append(f"{six}: unreadable ({exc})")
            continue
        if df.empty:
            structural_fails.append(f"{six}: empty")
            continue
        n_bonds += 1
        n_trades_total += len(df)

        # -- structural
        if not df["ts"].is_monotonic_increasing:
            structural_fails.append(f"{six}: timestamps not sorted")
        bad_px = (~df["price"].between(1, 300)).sum()
        bad_y = df["ytw"].notna() & (~df["ytw"].between(-10, 50))
        bad_par = (df["par"] < 1000).sum()
        bad_side = (~df["side"].isin(["D", "S", "P"])).sum()
        if bad_px or bad_y.sum() or bad_par or bad_side:
            structural_fails.append(
                f"{six}: bad px={bad_px} ytw={bad_y.sum()} "
                f"par={bad_par} side={bad_side}")

        # -- economic: S vs P by day
        day_side = (df.pivot_table(index="date", columns="side",
                                   values="price", aggfunc="median"))
        if "S" in day_side and "P" in day_side:
            both = day_side.dropna(subset=["S", "P"])
            if len(both):
                ok = (both["S"] >= both["P"])
                side_ok_days += int(ok.sum())
                side_bad_days += int((~ok).sum())
                spreads.extend((both["S"] - both["P"])
                               .clip(-5, 15).tolist())

        # -- price-yield inversion (daily median, bonds w/ enough days)
        daily = df.groupby("date")[["price", "ytw"]].median()
        if len(daily) >= 30 and daily["ytw"].notna().sum() >= 30:
            ch = daily.diff().dropna()
            ch = ch[(ch["price"] != 0) & (ch["ytw"] != 0)]
            if len(ch) >= 20:
                py_corrs.append(float(ch["price"].corr(ch["ytw"])))

        # -- coverage
        spans.append((df["ts"].iloc[-1] - df["ts"].iloc[0]).days)
        active_days.append(df["date"].nunique())
        first_years.append(df["ts"].iloc[0].year)
        par_values.append(df["par"].to_numpy())

        # -- cross-endpoint check
        if i in xcheck_sample and six in uni.index:
            u = uni.loc[six]
            win = df[(df["ts"] >= scan_begin)
                     & (df["ts"] < scan_end + pd.Timedelta(days=1))]
            if len(win):
                xcheck_rows.append({
                    "six": six,
                    "min_px_file": win["price"].min(),
                    "min_px_scan": float(u["min_px"]),
                    "max_px_file": win["price"].max(),
                    "max_px_scan": float(u["max_px"]),
                    "n_file": len(win),
                    "n_scan": int(u["trades_1y"]),
                })

    # summarize
    xdf = pd.DataFrame(xcheck_rows)
    if len(xdf):
        px_tol = ((xdf["min_px_file"] - xdf["min_px_scan"]).abs()
                  .le(0.011) &
                  (xdf["max_px_file"] - xdf["max_px_scan"]).abs()
                  .le(0.011))
        cnt_match = (xdf["n_file"] - xdf["n_scan"]).abs() <= np.maximum(
            2, 0.02 * xdf["n_scan"])
    else:
        px_tol = pd.Series(dtype=bool)
        cnt_match = pd.Series(dtype=bool)

    par_all = np.concatenate(par_values) if par_values else np.array([])
    pct_5k = float((par_all % 5000 == 0).mean()) if len(par_all) else np.nan

    side_total = side_ok_days + side_bad_days
    lines = [
        "# Data validation — EMMA individual muni trades",
        "",
        f"Files: **{n_bonds}** securities, **{n_trades_total:,}** trades.",
        "",
        "## 1. Structural checks",
        f"- Failures: **{len(structural_fails)}**",
        *[f"  - {s}" for s in structural_fails[:20]],
        "",
        "## 2. Side semantics (customer buy S ≥ customer sell P, same day)",
        f"- Bond-days with both sides: **{side_total:,}**",
        f"- median(S) ≥ median(P): **{side_ok_days / side_total:.1%}**"
        if side_total else "- no overlapping days",
        f"- Median same-day retail spread (S−P): "
        f"**{np.median(spreads):.3f}** points; mean {np.mean(spreads):.3f}"
        if spreads else "",
        "",
        "## 3. Price–yield inversion (daily changes, per bond)",
        f"- Bonds tested: **{len(py_corrs)}**",
        f"- Median corr(Δprice, Δytw): **{np.median(py_corrs):.3f}**; "
        f"share < 0: {np.mean(np.array(py_corrs) < 0):.1%}"
        if py_corrs else "",
        "",
        "## 4. Cross-endpoint check vs FindSimilarSecurities summary",
        f"- Sampled securities: **{len(xdf)}**",
        f"- Min/max price match (±0.011): **{px_tol.mean():.1%}**"
        if len(xdf) else "",
        f"- Trade-count match (±max(2,2%)): **{cnt_match.mean():.1%}**"
        if len(xdf) else "",
        "",
        "## 5. Coverage",
        f"- Median span per bond: **{np.median(spans):.0f}** days; "
        f"median active days: **{np.median(active_days):.0f}**",
        f"- First-trade year distribution: "
        f"{pd.Series(first_years).value_counts().sort_index().to_dict()}",
        f"- Par sizes multiple of 5k: **{pct_5k:.1%}**",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
