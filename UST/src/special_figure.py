#!/usr/bin/env python3
"""Figure: the Fed-specialness signal identifies scarce bonds but does not
predict a tradeable price move, IS or OOS.

Left  : specialness quintile -> avg par borrowed from the Fed ($mm). Monotone,
        confirming the score really does rank scarcity.
Right : specialness quintile -> annualized forward 21-day duration-neutral
        idiosyncratic return (bp), IS and OOS. Flat/tiny -> no price edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
ROOT = SRC.parent
from explore_special import duration_neutral_returns

IS_END = pd.Timestamp("2019-12-31")
H = 21


def main() -> int:
    p = pd.read_parquet(ROOT / "data" / "processed" / "special_panel.parquet")
    p = p[p["sec_type"].isin(["MARKET BASED NOTE", "MARKET BASED BOND"])]
    p = p[p["tsy_years"] >= 1.0].sort_values(["cusip", "date"]).reset_index(drop=True)
    p = duration_neutral_returns(p)

    ridio = p.pivot_table(index="date", columns="cusip", values="r_idio", aggfunc="first")
    fwd = ridio.shift(-1).rolling(H).sum().shift(-(H - 1))
    fwd_long = fwd.stack().rename("fwd").reset_index()
    fwd_long.columns = ["date", "cusip", "fwd"]
    p = p.merge(fwd_long, on=["date", "cusip"], how="left")

    p = p[p["special"].notna()].copy()
    p["q"] = p.groupby("date")["special"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if s.nunique() >= 5 else np.nan)
    p = p.dropna(subset=["q"])
    p["win"] = np.where(p["date"] <= IS_END, "IS", "OOS")

    borrowed = p.groupby("q")["outstanding_loans"].mean() / 1e6
    fwd_bp = p.groupby(["win", "q"])["fwd"].mean() * 252 * 1e4  # annualized bp

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(borrowed.index, borrowed.values, color="#268")
    axes[0].set_title("Specialness quintile ranks scarcity")
    axes[0].set_xlabel("specialness quintile (4 = most special)")
    axes[0].set_ylabel("avg par borrowed from Fed ($mm)")
    axes[0].grid(alpha=0.3, axis="y")

    w = 0.38
    for i, win in enumerate(("IS", "OOS")):
        v = fwd_bp[win]
        axes[1].bar(v.index + (i - 0.5) * w, v.values, width=w, label=win)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("...but does not predict forward relative return")
    axes[1].set_xlabel("specialness quintile (4 = most special)")
    axes[1].set_ylabel("ann. forward 21d duration-neutral return (bp)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Fed securities-lending specialness: informative about scarcity, "
                 "not tradeable in price", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "specialness_signal.png", dpi=150)
    print("borrowed $mm by quintile:", borrowed.round(1).to_dict())
    print("fwd ann bp by win,quintile:")
    print(fwd_bp.round(1).to_string())
    print("wrote results/specialness_signal.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
