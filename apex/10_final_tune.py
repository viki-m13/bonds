"""APEX — Final tuning.

Test a range of design choices:
  - Different sleeve subsets
  - Different blend weights
  - Different portfolio vol targets
  - Different DD floors

Report the best by OOS Sharpe and OOS CAGR jointly.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import itertools
import numpy as np
import pandas as pd

import util
import sleeves as S

OUT = Path("/home/user/bonds/data/apex")


def finalize(r, target_vol=0.20, dd_floor=-0.15, dd_win=252):
    c = (1 + r).cumprod()
    hwm = c.rolling(dd_win, min_periods=30).max()
    dd = c / hwm - 1
    m = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)
    r2 = r * m
    rv = r2.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=1.5).shift(1).fillna(1.0)
    return r2 * vm


def metrics_window(r, start, end):
    return util.metrics(util.regime_slice(r, start, end))


def main():
    op, cp = util.load_prices()
    rc = cp.pct_change()

    # --- Build all 8 canonical sleeves ---
    sleeve_fns = {
        "TSMOM": S.sleeve_tsmom,
        "XSMOM": S.sleeve_xsmom,
        "RPAR": S.sleeve_rpar,
        "TREND_EQ": S.sleeve_trend_eq,
        "TREND_BD": S.sleeve_trend_bd,
        "TREND_GD": S.sleeve_trend_gd,
        "CREDIT": S.sleeve_credit,
        "VOLREG": S.sleeve_volreg,
    }
    sleeve_rets = {}
    for name, fn in sleeve_fns.items():
        sleeve_rets[name] = fn(cp, target_vol=0.10)
    R = pd.DataFrame(sleeve_rets).fillna(0.0)

    # Tune target_vol and dd_floor with EW blend
    print("Sweep target_vol × dd_floor (EW blend):")
    blend = R.mean(axis=1)
    for tv in (0.15, 0.20, 0.25, 0.30, 0.40):
        for dd in (-0.08, -0.12, -0.15, -0.20):
            rf = finalize(blend, target_vol=tv, dd_floor=dd)
            oos = metrics_window(rf, "2019-01-02", "2027-12-31")
            full = util.metrics(rf)
            print(f"  tv={tv:.2f} dd={dd:+.2f} -> Full SR={full['sharpe']:.2f} "
                  f"CAGR={full['cagr']*100:.1f}% MDD={full['mdd']*100:.1f}%  "
                  f"OOS SR={oos.get('sharpe',0):.2f} CAGR={oos.get('cagr',0)*100:.1f}%")

    print("\n\n--- Higher vol targets (force CAGR higher) ---")
    for tv in (0.35, 0.45, 0.55, 0.70):
        rf = finalize(blend, target_vol=tv, dd_floor=-0.15)
        full = util.metrics(rf)
        oos = metrics_window(rf, "2019-01-02", "2027-12-31")
        print(f"  tv={tv:.2f}  Full SR={full['sharpe']:.2f} CAGR={full['cagr']*100:.1f}% "
              f"MDD={full['mdd']*100:.1f}%  OOS SR={oos.get('sharpe',0):.2f} CAGR={oos.get('cagr',0)*100:.1f}%")

    # --- Try different subsets ---
    print("\n\n--- Sleeve subsets (EW blend, tv=0.20, dd=-0.15) ---")
    subsets = {
        "ALL 8": list(sleeve_fns.keys()),
        "No CREDIT,VOLREG": ["TSMOM", "XSMOM", "RPAR", "TREND_EQ", "TREND_BD", "TREND_GD"],
        "Trend+RP (4)": ["RPAR", "TREND_EQ", "TREND_BD", "TREND_GD"],
        "Momentum heavy": ["TSMOM", "XSMOM", "TREND_EQ"],
        "Best 5": ["RPAR", "XSMOM", "TREND_EQ", "TREND_GD", "TREND_BD"],
        "Diversified 6": ["RPAR", "XSMOM", "TREND_EQ", "TREND_BD", "TREND_GD", "CREDIT"],
    }
    for name, subset in subsets.items():
        ss = [s for s in subset if s in R.columns]
        blend = R[ss].mean(axis=1)
        rf = finalize(blend, target_vol=0.20, dd_floor=-0.15)
        full = util.metrics(rf)
        oos = metrics_window(rf, "2019-01-02", "2027-12-31")
        pre08 = metrics_window(rf, "2000-01-01", "2008-12-31")
        print(f"  {name:30s}  Full SR={full['sharpe']:.2f} OOS SR={oos.get('sharpe',0):.2f} "
              f"pre08 SR={pre08.get('sharpe',0):.2f}  CAGR={full['cagr']*100:.1f}%  MDD={full['mdd']*100:.1f}%")

    # --- Best variants from exhaustive single-sleeve subsets
    print("\n\n--- Finding best 6-sleeve subset by OOS SR ---")
    best_sr = -np.inf
    best_subset = None
    all_sleeves = list(sleeve_fns.keys())
    from itertools import combinations
    for sz in (4, 5, 6, 7, 8):
        for subset in combinations(all_sleeves, sz):
            ss = list(subset)
            blend = R[ss].mean(axis=1)
            rf = finalize(blend, target_vol=0.20, dd_floor=-0.15)
            oos = metrics_window(rf, "2019-01-02", "2027-12-31")
            full = util.metrics(rf)
            # Scoring: OOS SR - 0.1 * (2022 shortfall)
            score = oos.get("sharpe", 0) + 0.5 * full.get("sharpe", 0)
            if score > best_sr:
                best_sr = score
                best_subset = ss
    print(f"Best subset: {best_subset}  (score={best_sr:.3f})")
    blend = R[best_subset].mean(axis=1)
    rf = finalize(blend, target_vol=0.20, dd_floor=-0.15)
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("RateHike 22", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(rf, s, e), f"  {lbl}")


if __name__ == "__main__":
    main()
