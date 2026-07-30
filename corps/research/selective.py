"""Selectivity experiments: does filtering the >=3pt dislocation signal on
CREDIT QUALITY (entry-day credit spread) and/or DURATION (years to maturity)
concentrate the alpha into far fewer, higher-conviction trades that still hold
out-of-sample?

Every filter is point-in-time: `cs` (credit spread) and `mat` (years to
maturity) are contemporaneous daily fields known on the signal day t, so gating
on them uses no future information. The filter is baked into the per-bond
`eligible` mask, which gates BOTH the strategy and its matched random-entry
control — so the reported excess is pure timing alpha *within* the selected
subset, and the trade count is the honest live-book breadth.

  python corps/research/selective.py
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
IS = (pd.Timestamp("2002-01-01"), pd.Timestamp("2015-12-31"))
OOS = (pd.Timestamp("2016-01-01"), DATA_END - pd.Timedelta(days=MAXH))
DISC = 3.0


def load():
    p = load_full(columns=["six", "date", "mid", "s_px", "p_px", "ytw", "cs", "mat"])
    p["mat_yr"] = p["mat"].astype("int64")
    coup = p.groupby("six")["ytw"].median().clip(1, 12)
    print(f"{p['six'].nunique()} bonds; preparing ...", flush=True)
    bonds = bt.prepare(p, coup)   # keeps cs / mat_yr columns per bond
    for g in bonds.values():      # snapshot base liquidity gate
        g["_base_elig"] = g["eligible"].to_numpy()
    return bonds


def set_gate(bonds, cond):
    """cond(g)->bool array; eligible = base liquidity gate & cond. Clears the
    _Arr cache so the new mask takes effect."""
    for g in bonds.values():
        g["eligible"] = g["_base_elig"] & cond(g)
    bt._ARR_CACHE.clear()


def ev(bonds, lo, hi, label, min_hold=365, max_hold=MAXH):
    fn = FACTORIES["price_discount"](discount=DISC)
    fills = bt.run_signal(bonds, fn, min_hold=min_hold, max_hold=max_hold,
                          date_lo=lo, date_hi=hi)
    ctl = bt.matched_random_control(bonds, fills, min_hold=min_hold,
                                    max_hold=max_hold, n_draws=15)
    s = bt.summarize(fills, label, control=ctl)
    print(f"  {label:30} n={s.get('n',0):6} bonds={s.get('n_bonds',0):5} "
          f"win={s.get('win_rate',0)*100:3.0f}% mean={s.get('mean_ret',0)*100:+6.2f}% "
          f"excess={s.get('excess_vs_control',0)*100:+5.2f}% p={s.get('excess_p_boot',1):.3f}",
          flush=True)
    return s


def both(bonds, cond, tag):
    set_gate(bonds, cond)
    r = {"IS": ev(bonds, *IS, f"{tag} IS"),
         "OOS": ev(bonds, *OOS, f"{tag} OOS")}
    return r


def main():
    bonds = load()
    out = {}

    print("\n[BASELINE] no credit/duration filter (full universe):", flush=True)
    out["base"] = both(bonds, lambda g: np.ones(len(g), bool), "base")

    # ---- CREDIT QUALITY (entry-day credit spread, cs is decimal) ----
    print("\n[A] Credit-quality band at entry (cs = credit spread):", flush=True)
    credit = {
        "cs<=1%  (top IG)":        lambda g: g["cs"].to_numpy() <= 0.01,
        "cs<=1.5% (solid IG)":     lambda g: g["cs"].to_numpy() <= 0.015,
        "cs 1-3% (mid IG)":        lambda g: (g["cs"].to_numpy() > 0.01) & (g["cs"].to_numpy() <= 0.03),
        "cs<=3%  (excl HY/dist)":  lambda g: g["cs"].to_numpy() <= 0.03,
        "cs<=5%  (excl distress)": lambda g: g["cs"].to_numpy() <= 0.05,
        "cs>3%   (crossover/HY)":  lambda g: g["cs"].to_numpy() > 0.03,
        "cs>5%   (distressed)":    lambda g: g["cs"].to_numpy() > 0.05,
    }
    out["credit"] = {}
    for tag, cond in credit.items():
        out["credit"][tag] = both(bonds, cond, tag)

    # ---- DURATION (years to maturity) ----
    print("\n[B] Duration band (mat = years to maturity):", flush=True)
    dur = {
        "mat<=3y  (short)":     lambda g: g["mat_yr"].to_numpy() <= 3,
        "mat<=5y  (short-int)": lambda g: g["mat_yr"].to_numpy() <= 5,
        "mat 5-12y (belly)":    lambda g: (g["mat_yr"].to_numpy() > 5) & (g["mat_yr"].to_numpy() <= 12),
        "mat>12y  (long)":      lambda g: g["mat_yr"].to_numpy() > 12,
    }
    out["duration"] = {}
    for tag, cond in dur.items():
        out["duration"][tag] = both(bonds, cond, tag)

    # ---- COMBINED: high-conviction selective operating points ----
    print("\n[C] Combined credit x duration (selective operating points):", flush=True)
    combo = {
        "IG(cs<=3%) & short(<=5y)":
            lambda g: (g["cs"].to_numpy() <= 0.03) & (g["mat_yr"].to_numpy() <= 5),
        "IG(cs<=3%) & belly(5-12y)":
            lambda g: (g["cs"].to_numpy() <= 0.03) & (g["mat_yr"].to_numpy() > 5) & (g["mat_yr"].to_numpy() <= 12),
        "solidIG(cs<=1.5%) & short(<=5y)":
            lambda g: (g["cs"].to_numpy() <= 0.015) & (g["mat_yr"].to_numpy() <= 5),
        "excl-dist(cs<=5%) & short(<=5y)":
            lambda g: (g["cs"].to_numpy() <= 0.05) & (g["mat_yr"].to_numpy() <= 5),
    }
    out["combo"] = {}
    for tag, cond in combo.items():
        out["combo"][tag] = both(bonds, cond, tag)

    (ROOT / "research" / "selective_results.json").write_text(
        json.dumps(out, default=float))
    print("\nwrote corps/research/selective_results.json", flush=True)


if __name__ == "__main__":
    main()
