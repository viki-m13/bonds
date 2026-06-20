"""Produce the IGNITION summary figure and current live picks for the record.

Outputs:
  research/ignition_summary.png  — year-by-year parabolic hit-rate (IGNITION vs
      base rate vs random pool) and the IS/OOS lift bars.
  research/current_picks.md      — today's top-15 IGNITION names (live signal).

Run from parabolic/:  python make_assets.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dca"))
import data as dca_data  # noqa: E402
import features as feat  # noqa: E402
import strategy as strat  # noqa: E402
import backtest as bt  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def main():
    P = dca_data.build_panel()
    F = feat.build_features(P)
    close, member = P["close"], P["member"] & P["close"].notna()
    score = strat.ignition_score(P, F)

    m = bt.basket_metrics(score, close, member)
    df = pd.concat([m["is"], m["oos"]])
    df["year"] = df["date"].dt.year
    g = df.groupby("year").agg(hit=("hit", "mean"), base=("base", "mean"),
                               rand=("rand_mean", "mean"),
                               fwd=("mean_fwd6", "mean"))

    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    yrs = g.index.values
    w = 0.38
    ax[0].bar(yrs - w / 2, g["hit"] * 100, w, label="IGNITION top-10 hit-rate",
              color="#c0392b")
    ax[0].bar(yrs + w / 2, g["base"] * 100, w, label="universe base rate",
              color="#bbbbbb")
    ax[0].axvline(2015.5, color="k", ls="--", lw=1)
    ax[0].text(2015.6, ax[0].get_ylim()[1] * 0.9, "OOS ->", fontsize=9)
    ax[0].set_ylabel("P(pick goes parabolic, +50% / 6m)  [%]")
    ax[0].set_title("IGNITION — share of picks that go parabolic, by year "
                    "(PIT S&P 500)")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(axis="y", alpha=0.3)

    # lift vs base and vs random, IS/OOS
    sis, soos = bt.summarise(m["is"]), bt.summarise(m["oos"])
    cats = ["hit-rate\n(IGNITION)", "base rate\n(universe)",
            "random pool\nmean fwd6", "IGNITION\nmean fwd6"]
    isv = [sis["hit"] * 100, sis["base"] * 100, None, None]
    x = np.arange(2)
    labels = ["IS  (2005-2015)", "OOS (2016-2026)"]
    hit = [sis["hit"] * 100, soos["hit"] * 100]
    base = [sis["base"] * 100, soos["base"] * 100]
    lift = [sis["lift"], soos["lift"]]
    ax[1].bar(x - 0.2, hit, 0.4, label="IGNITION hit-rate %", color="#c0392b")
    ax[1].bar(x + 0.2, base, 0.4, label="base rate %", color="#bbbbbb")
    for i in range(2):
        ax[1].text(x[i], max(hit[i], base[i]) + 0.4, f"{lift[i]:.1f}x lift",
                   ha="center", fontsize=10, fontweight="bold")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("P(parabolic)  [%]")
    ax[1].set_title("Parabolic hit-rate vs base rate — in-sample vs "
                    "out-of-sample (lift is stable)")
    ax[1].legend(fontsize=9)
    ax[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(HERE, "ignition_summary.png")
    fig.savefig(out_png, dpi=110)
    print("[written]", out_png)

    # current picks
    last = close.index[-1]
    s = score.loc[last].dropna().sort_values(ascending=False)
    top = s.head(15)
    lines = [f"# IGNITION live signal — top picks as of {last.date()}\n",
             f"Eligible names: {s.shape[0]}. Score = conditioned blend of "
             "early-turn (dist_52w_low), episodic-pivot gap, energy (ADR), low "
             "correlation and frog-in-the-pan smoothness, restricted to the "
             "high-ADR pond. Information through the close of "
             f"{last.date()}; a real deployment buys at the next open.\n",
             "| rank | ticker | percentile score |", "|---|---|---|"]
    for i, (t, v) in enumerate(top.items(), 1):
        lines.append(f"| {i} | {t} | {v:.2f} |")
    with open(os.path.join(HERE, "current_picks.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("[written]", os.path.join(HERE, "current_picks.md"))
    print("top-15:", ", ".join(top.index))


if __name__ == "__main__":
    main()
