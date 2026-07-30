"""Execution & slippage model for KEYSTONE-XL (munis). Same haircut design as
corps; capacity uses the muni panel's per-side par volumes (s_par = customer-
buy par printed on the entry day — the flow we would compete for).
Writes docs/exec_muni.json.

  python munis/research/exec_model.py
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
from xl_equity import nav_series, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parents[2] / "docs"


def haircut(fills, h):
    out = []
    for f in fills:
        out.append(bt.Fill(f.six, f.entry_date, f.entry_px + h,
                           f.exit_date, max(f.exit_px - h, 1.0),
                           f.coupon, f.stale_exit))
    return out


def main():
    bonds = load_bonds()
    print(f"{len(bonds)} muni bonds", flush=True)
    fn = FACTORIES["price_discount"](discount=3.0)
    base = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                         date_lo=IS_LO, date_hi=OOS_HI)
    fills = recovery_exit(bonds, issuer_cap(limit_filter(bonds, base)))
    print(f"XL fills {len(fills)}", flush=True)

    out = {"slippage": []}
    for h in (0.0, 0.125, 0.25, 0.5):
        fh = haircut(fills, h)
        nav = nav_series(fh)
        st = stats(nav)
        rr = np.array([f.ret for f in fh])
        row = {"h": h, "mean_ret": float(rr.mean()), "win": float((rr > 0).mean()),
               "cagr": st["cagr"], "maxdd": st["maxdd"]}
        out["slippage"].append(row)
        print(f"  h={h:5.3f}  mean={rr.mean()*100:+6.2f}% cagr={st['cagr']*100:+6.2f}% "
              f"maxdd={st['maxdd']*100:.1f}%", flush=True)

    entry_par = []
    for f in fills:
        g = bonds[f.six]
        row = g[g["date"] == f.entry_date]
        if len(row) and np.isfinite(row["s_par"].iloc[0]):
            entry_par.append(float(row["s_par"].iloc[0]))
    ep = np.array(entry_par) / 1e6      # $MM customer-buy par on entry day
    part = 0.25
    pos = part * ep
    span = (max(f.exit_date for f in fills) - min(f.entry_date for f in fills)).days
    concurrency = sum(f.hold_days for f in fills) / span
    out["capacity"] = {
        "participation": part,
        "median_pos_mm": float(np.median(pos)),
        "p75_pos_mm": float(np.percentile(pos, 75)),
        "avg_concurrency": float(concurrency),
        "annual_deployable_mm": float(pos.sum() / (span / 365.25)),
        "implied_aum_mm": float(np.median(pos) * concurrency),
        "n_with_volume": int(len(ep)),
    }
    print(f"  capacity: median pos ${np.median(pos):.3f}M @25% part., "
          f"concurrency {concurrency:.0f}, implied AUM ${np.median(pos)*concurrency:.1f}M",
          flush=True)
    ent_years = pd.Series([f.entry_date.year for f in fills])
    out["ops"] = {"fills_per_year": float(len(fills) / (span / 365.25)),
                  "max_fills_year": int(ent_years.value_counts().max())}
    (DOCS / "exec_muni.json").write_text(json.dumps(out, default=float))
    print("wrote docs/exec_muni.json", flush=True)


if __name__ == "__main__":
    main()
