"""Equity-curve figure for the HYPERVOL writeup (log scale, IS/OOS marked)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import load_coin, backtest, Config
from .carry import load_carry_frame, backtest_carry

CORE = ["BTC", "ETH"]
OUT = Path("/home/user/bonds/hypervol/results")
OUT.mkdir(parents=True, exist_ok=True)


def port(rets):
    return pd.concat(rets, axis=1).mean(axis=1).dropna()


def main():
    bh = port([(load_coin(c)["ret"] - load_coin(c)["funding_day"]).fillna(0).rename(c)
               for c in CORE])
    dirp = port([backtest(load_coin(c), Config(mode="directional"))["strat_ret"].rename(c)
                 for c in CORE])
    carry = port([backtest_carry(load_carry_frame(c), gated=False)["strat_ret"].rename(c)
                  for c in CORE])

    fig, ax = plt.subplots(figsize=(11, 6))
    for series, lab, col in [(bh, "Buy&hold perp 50/50 (net funding)", "#888"),
                             (dirp, "Faithful directional Strat-4 L/S", "#d62728"),
                             (carry, "Delta-neutral funding carry", "#1f77b4")]:
        eq = (1 + series).cumprod()
        ax.plot(eq.index, eq.values, label=lab, color=col, lw=1.8)

    cut = carry.index[int(len(carry) * 0.6)]
    ax.axvline(cut, color="k", ls="--", lw=1, alpha=0.6)
    ax.text(cut, ax.get_ylim()[1] * 0.95, "  IS | OOS (60/40)", fontsize=9, va="top")

    ax.set_yscale("log")
    ax.set_title("HYPERVOL — porting the VIX-ETN volatility strategy to Hyperliquid perps\n"
                 "BTC+ETH, 2023-06 to 2026-06, costs+funding modeled", fontsize=11)
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png", dpi=130)
    print("saved", OUT / "equity_curves.png")


if __name__ == "__main__":
    main()
