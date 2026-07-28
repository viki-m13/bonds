"""Equity curves + per-era breakdown for the selective finalists identified by
selective.py: DURATION is the concentrating axis (short <=5y), optionally
trimming deep distress (cs<=5%). Confirms the higher per-trade edge is not
hiding worse drawdown. Writes corps/research/selective_equity.json.

  python corps/research/selective_equity.py
"""

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

MAXH = 455
DATA_END = pd.Timestamp("2025-03-31")
DISC = 3.0
ERAS = [("2004-2007", "2004-01-01", "2007-12-31"),
        ("2008-2009 GFC", "2008-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2021-2023", "2021-01-01", "2023-12-31")]


def load():
    p = load_full(columns=["six", "date", "mid", "s_px", "p_px", "ytw", "cs", "mat"])
    p["mat_yr"] = p["mat"].astype("int64")
    coup = p.groupby("six")["ytw"].median().clip(1, 12)
    print(f"{p['six'].nunique()} bonds; preparing ...", flush=True)
    bonds = bt.prepare(p, coup)
    for g in bonds.values():
        g["_base_elig"] = g["eligible"].to_numpy()
    return bonds


def set_gate(bonds, cond):
    for g in bonds.values():
        g["eligible"] = g["_base_elig"] & cond(g)
    bt._ARR_CACHE.clear()


def _fills(bonds, lo, hi):
    fn = FACTORIES["price_discount"](discount=DISC)
    return bt.run_signal(bonds, fn, min_hold=365, max_hold=MAXH,
                         date_lo=lo, date_hi=hi)


def equity(bonds):
    fills = _fills(bonds, pd.Timestamp("2002-01-01"), DATA_END - pd.Timedelta(days=MAXH))
    start = min(f.entry_date for f in fills); end = max(f.exit_date for f in fills)
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
           .set_index("date")["adjclose"].reindex(days).ffill().bfill())
    lqd_eq = (lqd / lqd.iloc[0]).to_numpy()
    mask = ~np.isnan(lqd_eq)
    return {
        "series": [{"date": days[i].strftime("%Y-%m-%d"), "strat": round(float(eq[i]), 4),
                    "lqd": round(float(lqd_eq[i]), 4) if not np.isnan(lqd_eq[i]) else None}
                   for i in mon],
        "strat": st(eq, days), "lqd": st(lqd_eq[mask], days[mask]) if mask.any() else None,
        "n_trades": len(fills), "years": round((days[-1] - days[0]).days / 365.25, 1),
        "avg_positions": int(cnt[cnt > 0].mean()),
    }


def era_row(bonds, lo, hi, label):
    fills = _fills(bonds, pd.Timestamp(lo), pd.Timestamp(hi))
    ctl = bt.matched_random_control(bonds, fills, min_hold=365, max_hold=MAXH, n_draws=15)
    s = bt.summarize(fills, label, control=ctl)
    return {"label": label, "n": s.get("n", 0), "win_rate": s.get("win_rate", 0),
            "mean_ret": s.get("mean_ret", 0), "excess_vs_control": s.get("excess_vs_control", 0),
            "excess_p_boot": s.get("excess_p_boot", 1)}


CONDS = {
    "short5": ("Short-dated ≤5y", lambda g: g["mat_yr"].to_numpy() <= 5),
    "short5_xdist": ("Short ≤5y, exclude deep distress (cs≤5%)",
                     lambda g: (g["mat_yr"].to_numpy() <= 5) & (g["cs"].to_numpy() <= 0.05)),
}


def main():
    bonds = load()
    out = {}
    for key, (label, cond) in CONDS.items():
        print(f"\n[{key}] {label}", flush=True)
        set_gate(bonds, cond)
        eq = equity(bonds)
        print(f"  equity: total={eq['strat']['total']*100:+.1f}% "
              f"cagr={eq['strat']['cagr']*100:+.2f}% maxdd={eq['strat']['maxdd']*100:.1f}% "
              f"n={eq['n_trades']} avg_pos={eq['avg_positions']}", flush=True)
        eras = [era_row(bonds, lo, hi, lab) for lab, lo, hi in ERAS]
        for e in eras:
            print(f"    {e['label']:16} n={e['n']:5} win={e['win_rate']*100:3.0f}% "
                  f"excess={e['excess_vs_control']*100:+5.2f}% p={e['excess_p_boot']:.3f}", flush=True)
        out[key] = {"label": label, "equity": eq, "era": eras}
    (ROOT / "research" / "selective_equity.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/selective_equity.json", flush=True)


if __name__ == "__main__":
    main()
