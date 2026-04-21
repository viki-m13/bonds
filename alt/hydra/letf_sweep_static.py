"""Step 3 — static-weight recipe sweep.

Covers systematic mix-grids across:
  (a) single-LETF concentration (100% each major LETF)
  (b) HFEA variants: UPRO/TMF and TQQQ/TMF at 40/60 → 80/20
  (c) 3-sleeve UPRO/TMF/UGL (stock/bond/gold) over a 10% grid
  (d) 4-sleeve UPRO/TQQQ/TMF/UGL over a coarser grid
  (e) Equal-weight-N variants for several size-N baskets
  (f) Theme-balanced baskets (equity / tech / bond / gold)

Rebalance cadence tested: {3, 5, 10, 21} business days.
Execution lag: 1 day (next-day open).
Window: 2011-01-04 .. latest (all 17 long LETFs active).
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from letf_engine import (load_universe, common_window_returns,
                         run_backtest, summarise, w_fixed)
from letf_universe import LETF_LONG_2011


REBAL_DAYS = [3, 5, 10, 21]
START = "2011-01-01"
OUT_DIR = Path("/home/user/bonds/data/results")


def mix_grid_2(a, b, step=0.10):
    """All (a, b) pairs on step grid summing to 1, inclusive 0.10..0.90."""
    out = []
    for i in range(1, 10):
        wa = round(i * step, 2)
        wb = round(1.0 - wa, 2)
        out.append({a: wa, b: wb})
    return out


def mix_grid_3(a, b, c, step=0.10):
    out = []
    for i in range(0, 11):
        for j in range(0, 11 - i):
            k = 10 - i - j
            out.append({a: i * step, b: j * step, c: k * step})
    # keep those with every weight in [0.1, 0.8] to avoid degenerate ones
    out = [w for w in out if max(w.values()) <= 0.8 and min(w.values()) >= 0.1]
    return out


def mix_grid_4(a, b, c, d, step=0.20):
    out = []
    # step=0.20 => weights in {0.2, 0.4, 0.6, 0.8} summing to 1
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                l = 5 - i - j - k
                if 1 <= l <= 4:
                    out.append({a: i * step, b: j * step,
                                c: k * step, d: l * step})
    return out


def build_recipes():
    recipes = {}

    # (a) single-LETF
    for t in LETF_LONG_2011:
        recipes[f"100% {t}"] = {t: 1.0}

    # (b) HFEA UPRO/TMF grid
    for w in mix_grid_2("UPRO", "TMF"):
        recipes[f"HFEA {int(w['UPRO']*100)}/{int(w['TMF']*100)} UPRO/TMF"] = w

    # HFEA tech TQQQ/TMF grid
    for w in mix_grid_2("TQQQ", "TMF"):
        recipes[f"HFEA-Tech {int(w['TQQQ']*100)}/{int(w['TMF']*100)} TQQQ/TMF"] = w

    # UPRO/UGL and TQQQ/UGL (stock vs gold)
    for w in mix_grid_2("UPRO", "UGL"):
        recipes[f"UPRO/UGL {int(w['UPRO']*100)}/{int(w['UGL']*100)}"] = w

    # (c) 3-sleeve UPRO/TMF/UGL
    for w in mix_grid_3("UPRO", "TMF", "UGL"):
        k = f"3sleeve {int(w['UPRO']*100)}/{int(w['TMF']*100)}/{int(w['UGL']*100)} UPRO/TMF/UGL"
        recipes[k] = w

    # TQQQ/TMF/UGL 3-sleeve
    for w in mix_grid_3("TQQQ", "TMF", "UGL"):
        k = f"3sleeve {int(w['TQQQ']*100)}/{int(w['TMF']*100)}/{int(w['UGL']*100)} TQQQ/TMF/UGL"
        recipes[k] = w

    # (d) 4-sleeve UPRO/TQQQ/TMF/UGL
    for w in mix_grid_4("UPRO", "TQQQ", "TMF", "UGL"):
        k = (f"4sleeve {int(w['UPRO']*100)}/{int(w['TQQQ']*100)}/"
             f"{int(w['TMF']*100)}/{int(w['UGL']*100)} UPRO/TQQQ/TMF/UGL")
        recipes[k] = w

    # (e) Equal-weight N-baskets
    baskets = {
        "EW5 UPRO/TQQQ/SOXL/TMF/UGL": ["UPRO","TQQQ","SOXL","TMF","UGL"],
        "EW5 UPRO/TQQQ/TECL/TMF/UGL": ["UPRO","TQQQ","TECL","TMF","UGL"],
        "EW4 UPRO/TQQQ/TMF/UGL":      ["UPRO","TQQQ","TMF","UGL"],
        "EW4 UPRO/TMF/UGL/UCO":       ["UPRO","TMF","UGL","UCO"],
        "EW6 UPRO/TQQQ/SOXL/TECL/TMF/UGL":
            ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"],
        "EW7 core+finance+em":
            ["UPRO","TQQQ","SOXL","TMF","UGL","FAS","EDC"],
        "EW-ALL 17 LETFs":             LETF_LONG_2011,
    }
    for k, ts in baskets.items():
        recipes[k] = {t: 1.0 / len(ts) for t in ts}

    # (f) theme-balanced (25/25/25/25 equity/tech/bond/gold) families
    for eq in ["UPRO", "SSO"]:
        for tech in ["TQQQ", "QLD", "TECL"]:
            for bnd in ["TMF", "TYD", "UBT"]:
                for g in ["UGL", "NUGT"]:
                    recipes[f"theme4 {eq}/{tech}/{bnd}/{g} 25/25/25/25"] = {
                        eq: 0.25, tech: 0.25, bnd: 0.25, g: 0.25
                    }

    return recipes


def main():
    px = load_universe(LETF_LONG_2011, start=START).dropna(how="any")
    rets = common_window_returns(px)
    print(f"Universe: {list(rets.columns)}")
    print(f"Window:   {rets.index[0].date()} .. {rets.index[-1].date()} "
          f"({len(rets)} days)")

    recipes = build_recipes()
    print(f"Testing {len(recipes)} recipes × {len(REBAL_DAYS)} cadences "
          f"= {len(recipes)*len(REBAL_DAYS)} backtests")

    rows = []
    for name, w in recipes.items():
        for nd in REBAL_DAYS:
            r, _ = run_backtest(rets, w_fixed(w), rebal_days=nd, exec_lag=1)
            s = summarise(r.dropna(), f"{name} @ {nd}d")
            s["recipe"] = name
            s["rebal_days"] = nd
            rows.append(s)

    df = pd.DataFrame(rows)

    # Drop rebal-cadence noise: for static weights it barely matters, keep the
    # best rebal per recipe (by CAGR)
    best_per_recipe = (df.sort_values("cagr", ascending=False)
                        .groupby("recipe", sort=False).head(1))
    summary = df.sort_values("cagr", ascending=False).reset_index(drop=True)

    summary.to_csv(OUT_DIR / "letf_sweep_static.csv", index=False)
    print(f"\nSaved full sweep ({len(summary)} rows) to "
          f"letf_sweep_static.csv")

    print("\nTop 20 by CAGR:")
    top = summary.head(20)
    for _, r in top.iterrows():
        print(f"  {r['label']:58s}  CAGR={r['cagr']:>6.2f}%  "
              f"Vol={r['vol']:>5.1f}%  MDD={r['mdd']:>7.2f}%  "
              f"SR={r['sharpe']:>4.2f}  C/MDD={r['cagr_mdd']:>4.2f}")

    print("\nTop 20 by CAGR / |MDD|:")
    topm = summary.sort_values("cagr_mdd", ascending=False).head(20)
    for _, r in topm.iterrows():
        print(f"  {r['label']:58s}  C/MDD={r['cagr_mdd']:>4.2f}  "
              f"CAGR={r['cagr']:>5.2f}%  MDD={r['mdd']:>7.2f}%  "
              f"SR={r['sharpe']:>4.2f}")

    print("\nTop 20 by Sharpe:")
    tops = summary.sort_values("sharpe", ascending=False).head(20)
    for _, r in tops.iterrows():
        print(f"  {r['label']:58s}  SR={r['sharpe']:>4.2f}  "
              f"CAGR={r['cagr']:>5.2f}%  MDD={r['mdd']:>7.2f}%  "
              f"Vol={r['vol']:>5.1f}%")


if __name__ == "__main__":
    main()
