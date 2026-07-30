"""KEYSTONE-XL equity curve for the muni page: XL vs base vs MUB, monthly NAV
(linear intra-trade attribution — the page's stated convention for munis,
which do not print daily). Writes docs/keystone_xl_curve.json.

  python munis/research/xl_equity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
from strategies import FACTORIES  # noqa: E402
from limit_transfer import load_bonds, limit_filter  # noqa: E402
from keystone_xl import issuer_cap, recovery_exit, IS_LO, OOS_HI  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parents[2] / "docs"


def nav_series(fills):
    d0 = min(f.entry_date for f in fills); d1 = max(f.exit_date for f in fills)
    days = pd.date_range(d0, d1, freq="D")
    di = {d: i for i, d in enumerate(days)}
    sret = np.zeros(len(days)); cnt = np.zeros(len(days))
    for f in fills:
        if f.hold_days <= 0:
            continue
        dr = (1 + f.ret) ** (1 / f.hold_days) - 1
        sret[di[f.entry_date]:di[f.exit_date]] += dr
        cnt[di[f.entry_date]:di[f.exit_date]] += 1
    daily = np.where(cnt > 0, sret / np.where(cnt > 0, cnt, 1), 0)
    nav = np.cumprod(1 + daily)
    return pd.Series(nav, index=days)


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    peak = nav.cummax()
    return {"total": float(nav.iloc[-1] - 1),
            "cagr": float(nav.iloc[-1] ** (1 / yrs) - 1),
            "maxdd": float((nav / peak - 1).min())}


def main():
    bonds = load_bonds()
    print(f"{len(bonds)} muni bonds", flush=True)
    fn = FACTORIES["price_discount"](discount=3.0)
    base = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                         date_lo=IS_LO, date_hi=OOS_HI)
    xl = recovery_exit(bonds, issuer_cap(limit_filter(bonds, base)))
    print(f"base fills {len(base)}, XL fills {len(xl)}", flush=True)
    nav_b = nav_series(base)
    nav_x = nav_series(xl)
    idx = nav_b.index.union(nav_x.index)
    nav_b = nav_b.reindex(idx).ffill().fillna(1.0)
    nav_x = nav_x.reindex(idx).ffill().fillna(1.0)
    mub = (pd.read_csv(ROOT / "data" / "mub_daily.csv.gz", parse_dates=["date"])
           .set_index("date").iloc[:, 0].reindex(idx).ffill().bfill())
    mub_eq = mub / mub.iloc[0]
    mon = pd.Series(range(len(idx)), index=idx).resample("MS").first().dropna().astype(int)
    series = [{"date": idx[i].strftime("%Y-%m-%d"),
               "xl": round(float(nav_x.iloc[i]), 4),
               "base": round(float(nav_b.iloc[i]), 4),
               "mub": round(float(mub_eq.iloc[i]), 4)} for i in mon]
    if series[-1]["date"] != idx[-1].strftime("%Y-%m-%d"):
        series.append({"date": idx[-1].strftime("%Y-%m-%d"),
                       "xl": round(float(nav_x.iloc[-1]), 4),
                       "base": round(float(nav_b.iloc[-1]), 4),
                       "mub": round(float(mub_eq.iloc[-1]), 4)})
    out = {"series": series, "xl": stats(nav_x), "base": stats(nav_b),
           "mub": stats(mub_eq), "n_xl": len(xl), "n_base": len(base),
           "window": f"{idx[0].date()} .. {idx[-1].date()}"}
    (DOCS / "keystone_xl_curve.json").write_text(json.dumps(out, default=float))
    print("XL:", out["xl"], "\nbase:", out["base"], "\nMUB:", out["mub"], flush=True)
    print("wrote docs/keystone_xl_curve.json", flush=True)


if __name__ == "__main__":
    main()
