"""Extract representative KEYSTONE-XL muni trades with surrounding tape for
the muni page's trade-anatomy section. Same selection protocol as corps:
percentile picks (p90/p50/p10) by realized return among tape-dense trades.

  python munis/research/trade_examples.py
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

DOCS = Path(__file__).resolve().parents[2] / "docs"


def frame(bonds, six):
    g = bonds[six]
    med = g.set_index("date")["mid"].rolling("60D", min_periods=5).median().shift(1)
    return g, med


def tape(bonds, f, label, note):
    g, med = frame(bonds, f.six)
    lo = f.entry_date - pd.Timedelta(days=90)
    hi = f.exit_date + pd.Timedelta(days=60)
    seg = g[(g["date"] >= lo) & (g["date"] <= hi)]
    rows = []
    for _, r in seg.iterrows():
        def fv(x):
            return round(float(x), 3) if np.isfinite(x) else None
        rows.append({"d": r["date"].strftime("%Y-%m-%d"), "m": fv(r["mid"]),
                     "a": fv(r.get("s_px", np.nan)), "b": fv(r.get("p_px", np.nan)),
                     "md": fv(med.asof(r["date"]))})
    return {"label": label, "note": note, "cusip": f.six,
            "entry_date": f.entry_date.strftime("%Y-%m-%d"),
            "entry_px": round(float(f.entry_px), 3),
            "exit_date": f.exit_date.strftime("%Y-%m-%d"),
            "exit_px": round(float(f.exit_px), 3),
            "ret": round(float(f.ret), 4), "hold": int(f.hold_days),
            "coupon": round(float(f.coupon), 2), "stale": bool(f.stale_exit),
            "tape": rows}


def main():
    bonds = load_bonds()
    print(f"{len(bonds)} muni bonds", flush=True)
    fn = FACTORIES["price_discount"](discount=3.0)
    base = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                         date_lo=IS_LO, date_hi=OOS_HI)
    fills = recovery_exit(bonds, issuer_cap(limit_filter(bonds, base)))
    dense = []
    for f in fills:
        g = bonds[f.six]
        seg = g[(g["date"] >= f.entry_date - pd.Timedelta(days=90))
                & (g["date"] <= f.exit_date + pd.Timedelta(days=60))]
        if len(seg) >= 40:
            dense.append(f)
    dense.sort(key=lambda f: f.ret)
    n = len(dense)
    print(f"KEYSTONE-XL fills {len(fills)}, tape-dense {n}", flush=True)
    picks = [
        tape(bonds, dense[int(0.90 * n)], "90th-percentile outcome",
             "A deep muni dislocation — typically a retail liquidation absorbed by a dealer — reverting to its median."),
        tape(bonds, dense[int(0.50 * n)], "Median outcome",
             "The typical KEYSTONE-XL trade: buy the forced seller's print, sell into the recovery."),
        tape(bonds, dense[int(0.10 * n)], "10th-percentile outcome",
             "A loser, shown deliberately: the discount persisted or widened and the exit realized a loss."),
    ]
    for p in picks:
        print(f"  {p['label']:24} {p['cusip']} {p['entry_date']}→{p['exit_date']} "
              f"ret={p['ret']*100:+.1f}% hold={p['hold']}d tape={len(p['tape'])}", flush=True)
    sel = {"protocol": "Percentile picks by realized return among tape-dense "
                       "KEYSTONE-XL trades (>=40 prints in window). Not cherry-picked.",
           "trades": picks}
    (DOCS / "trade_examples_muni.json").write_text(json.dumps(sel, default=float))
    print("wrote docs/trade_examples_muni.json", flush=True)


if __name__ == "__main__":
    main()
