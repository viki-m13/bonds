"""Assemble the corporate white-paper data (docs/corps_data.json) from the
bias-free full-universe results + equity curves. Publishes only the bias-free
numbers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "munis" / "research"))
import backtest as bt          # noqa: E402
sys.path.insert(0, str(ROOT / "research"))
from panel_io import load_full  # noqa: E402
from strategies import FACTORIES  # noqa: E402

DOCS = ROOT.parent / "docs"
MAXH = 455
DATA_END = pd.Timestamp("2025-03-31")


def equity(bonds, disc):
    fn = FACTORIES["price_discount"](discount=disc)
    fills = bt.run_signal(bonds, fn, min_hold=365, max_hold=MAXH,
                          date_lo=pd.Timestamp("2002-01-01"),
                          date_hi=DATA_END - pd.Timedelta(days=MAXH))
    start = min(f.entry_date for f in fills)
    end = max(f.exit_date for f in fills)
    days = pd.date_range(start, end, freq="D")
    di = {d: i for i, d in enumerate(days)}
    sret = np.zeros(len(days)); cnt = np.zeros(len(days))
    for f in fills:
        if f.hold_days <= 0:
            continue
        dr = (1 + f.ret) ** (1.0 / f.hold_days) - 1
        sret[di[f.entry_date]:di[f.exit_date]] += dr
        cnt[di[f.entry_date]:di[f.exit_date]] += 1
    eq = np.cumprod(1 + np.where(cnt > 0, sret / np.where(cnt > 0, cnt, 1), 0))

    def st(e, dd):
        yrs = (dd[-1] - dd[0]).days / 365.25
        peak = np.maximum.accumulate(e)
        return {"total": float(e[-1] - 1), "cagr": float(e[-1] ** (1 / yrs) - 1),
                "maxdd": float((e / peak - 1).min())}
    idx = pd.Series(range(len(days)), index=days)
    mon = idx.resample("MS").first().dropna().astype(int).tolist()
    if len(days) - 1 not in mon:
        mon.append(len(days) - 1)
    def bench(fname):
        """Rebase a benchmark to $1 at its FIRST AVAILABLE date. No bfill:
        AGG launched 2003-09, after this book starts, and backfilling would
        credit it a flat stretch it never had. NaN before inception."""
        s = (pd.read_csv(ROOT / "data" / "etf" / fname, parse_dates=["date"])
             .set_index("date")["adjclose"])
        first = s.index.min()
        r = s.reindex(days.union(s.index)).ffill().reindex(days)
        r[days < first] = np.nan
        e = (r / r.dropna().iloc[0]).to_numpy()
        return e, ~np.isnan(e), first

    lqd_eq, lqd_mask, lqd_first = bench("LQD.csv.gz")
    agg_eq, agg_mask, agg_first = bench("AGG.csv.gz")

    def jn(v):
        return round(float(v), 4) if np.isfinite(v) else None

    # common window (from the later of the two benchmark inceptions) so the
    # strategy and both benchmarks are measured over identical dates
    c0 = max(lqd_first, agg_first)
    cm = days >= c0
    def rebase(e):
        sub = e[cm]
        return sub / sub[0]
    common = {"start": c0.strftime("%Y-%m-%d"),
              "strat": st(rebase(eq), days[cm]),
              "lqd": st(rebase(lqd_eq), days[cm]),
              "agg": st(rebase(agg_eq), days[cm])}
    return {
        "series": [{"date": days[i].strftime("%Y-%m-%d"),
                    "strat": round(float(eq[i]), 4),
                    "lqd": jn(lqd_eq[i]), "agg": jn(agg_eq[i])}
                   for i in mon],
        "strat": st(eq, days),
        "lqd": st(lqd_eq[lqd_mask], days[lqd_mask]) if lqd_mask.any() else None,
        "agg": (dict(st(agg_eq[agg_mask], days[agg_mask]),
                     start=agg_first.strftime("%Y-%m-%d")) if agg_mask.any() else None),
        "common": common,
        "n_trades": len(fills), "years": round((days[-1] - days[0]).days / 365.25, 1),
        "avg_positions": int(cnt[cnt > 0].mean()),
    }


def main():
    p = load_full(columns=["six", "date", "mid", "s_px", "p_px", "ytw"])
    coup = p.groupby("six")["ytw"].median().clip(1, 12)
    n_bonds = p["six"].nunique(); n_days = len(p)
    print(f"{n_bonds} bonds; preparing ...", flush=True)
    bonds = bt.prepare(p, coup)

    res = json.loads((ROOT / "research" / "osbap_results.json").read_text())
    imp1 = json.loads((ROOT / "research" / "osbap_improve_results.json").read_text())
    imp2 = json.loads((ROOT / "research" / "osbap_improve2_results.json").read_text())

    print("equity >=3pt ...", flush=True); eq3 = equity(bonds, 3.0)
    print("equity >=4pt ...", flush=True); eq4 = equity(bonds, 4.0)

    data = {
        "meta": {"n_bonds": int(n_bonds), "n_days": int(n_days),
                 "raw_bonds": 73835, "raw_days": 29776137,
                 "start": "2002-07-01", "end": "2025-03-31",
                 "source": "Open Source Bond Asset Pricing (openbondassetpricing.com)"},
        "threshold": res["threshold"], "is": res["is"], "oos": res["oos"],
        "era": res["era"], "equity3": eq3, "equity4": eq4,
        "improve1": imp1, "improve2": imp2,
    }
    DOCS.mkdir(exist_ok=True)
    # Preserve keys this script does not own (e.g. "selective", written by
    # selective_equity.py). Rewriting the file wholesale silently dropped them
    # and broke the page — merge instead of overwrite.
    out_p = DOCS / "corps_data.json"
    if out_p.exists():
        prior = json.loads(out_p.read_text())
        kept = [k for k in prior if k not in data]
        if kept:
            print(f"preserving keys owned elsewhere: {kept}", flush=True)
            data = {**{k: prior[k] for k in kept}, **data}
    out_p.write_text(json.dumps(data, default=float))
    print("wrote docs/corps_data.json", flush=True)
    print("eq3:", eq3["strat"], "| lqd:", eq3["lqd"])


if __name__ == "__main__":
    main()
