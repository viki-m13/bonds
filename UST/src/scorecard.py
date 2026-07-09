#!/usr/bin/env python3
"""Capstone scorecard: in-sample vs out-of-sample net Sharpe across every
strategy family honestly tested in this folder.

Numbers are taken from the committed result files where available
(final_results.json, long_tilt_result.json, bounce_decomposition.csv) and from
the logged IS/OOS runs of the other families (recorded here with their source
script so they are reproducible). The point of the figure is the pattern, not
any single bar: every family with a positive in-sample Sharpe collapses to
negative out-of-sample — the signature of a regime break (2010s ZIRP/QE ->
2020s hiking/QT), not of tradeable edge.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def g(path, key):
    d = json.loads((R / path).read_text())
    for m in d["metrics"]:
        if m["label"] == key:
            return m
    return {}


def main() -> int:
    # pull from committed JSONs
    rv = json.loads((R / "final_results.json").read_text())
    def rvget(lbl):
        return next(m for m in rv["metrics"] if m["label"] == lbl)
    rv_is = rvget("net_IS")["sharpe"]
    rv_oos = rvget("net_OOS")["sharpe"]
    lt = json.loads((R / "long_tilt_result.json").read_text())
    lt_is = next(m for m in lt["metrics"] if m["label"] == "IS")["active_sharpe"]
    lt_oos = next(m for m in lt["metrics"] if m["label"] == "OOS")["active_sharpe"]

    # families with IS/OOS net Sharpe. Source script noted in comments.
    rows = [
        # (name, IS, OOS, source)
        ("Curve+carry RV combo", rv_is, rv_oos, "run_final.py"),
        ("Carry sleeve alone", 0.38, -2.16, "run_final.py sleeve_carry"),
        ("Value-z sleeve alone", 0.41, -0.36, "run_final.py sleeve_value"),
        ("Local RV (lag>=1, net)", -1.27, -1.51, "rv_backtest.py step5 f0.2 l1"),
        ("Momentum (idio)", -1.22, None, "run_experiments.py (neg IS, dropped)"),
        ("Specialness L/S", 0.55, -0.55, "explore_special quintile sign-flip"),
        ("Butterfly curvature", -0.37, None, "butterfly.py (net neg IS, dropped)"),
        ("Long-only cheap/carry tilt", lt_is, lt_oos, "long_tilt.py"),
    ]

    names = [r[0] for r in rows]
    isv = [r[1] for r in rows]
    oosv = [r[2] if r[2] is not None else np.nan for r in rows]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    h = 0.38
    ax.barh(y + h/2, isv, height=h, label="in-sample 2010-2019", color="#268")
    ax.barh(y - h/2, oosv, height=h, label="out-of-sample 2020-2026", color="#c44")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(3, color="green", ls=":", lw=1.2, label="Sharpe = 3 target")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("net Sharpe ratio")
    ax.set_title("Every strategy with positive in-sample Sharpe collapses out-of-sample\n"
                 "(freely-available data: EOD prices + Fed specialness). None approaches 3.",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(R / "scorecard.png", dpi=150)

    print(f"{'strategy':30} {'IS':>6} {'OOS':>6}  source")
    for n, i, o, s in rows:
        os = f"{o:.2f}" if o is not None else "  -  "
        print(f"{n:30} {i:>6.2f} {os:>6}  {s}")
    (R / "scorecard.json").write_text(json.dumps(
        [{"strategy": n, "is_sharpe": i, "oos_sharpe": o, "source": s} for n, i, o, s in rows],
        indent=2))
    print("\nwrote results/scorecard.png, results/scorecard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
