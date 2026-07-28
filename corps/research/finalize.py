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
    lqd = (pd.read_csv(ROOT / "data" / "etf" / "LQD.csv.gz", parse_dates=["date"])
           .set_index("date")["adjclose"].reindex(days).ffill())
    lqd_eq = (lqd / lqd.iloc[0]).to_numpy()
    return {
        "series": [{"date": days[i].strftime("%Y-%m-%d"),
                    "strat": round(float(eq[i]), 4),
                    "lqd": round(float(lqd_eq[i]), 4) if not np.isnan(lqd_eq[i]) else None}
                   for i in mon],
        "strat": st(eq, days), "lqd": st(lqd_eq[~np.isnan(lqd_eq)],
                                         days[~np.isnan(lqd_eq)]),
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
    (DOCS / "corps_data.json").write_text(json.dumps(data, default=float))
    print("wrote docs/corps_data.json", flush=True)
    print("eq3:", eq3["strat"], "| lqd:", eq3["lqd"])


if __name__ == "__main__":
    main()
