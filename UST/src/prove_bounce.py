#!/usr/bin/env python3
"""Produce the decisive figure: why the high paper-Sharpe is bid-ask bounce.

For the best-looking local-RV config (daily rebalance, frac 0.2), sweep the
execution lag 0..4 and plot gross and net annualized Sharpe, IS and OOS.

Interpretation:
- lag 0 = trade at the SAME close whose marks generated the signal. This
  harvests mean-reversion in the end-of-day marks themselves (bid-ask
  bounce). It is not achievable: you cannot both observe the closing mark and
  trade at it, and much of the reversion is measurement noise in the mark.
- lag >= 1 = observe at close t, trade at a later close. Any real, tradeable
  convergence survives; pure bounce does not.

If gross Sharpe collapses from very high at lag 0 to ~0 at lag 1, and net
Sharpe is negative at every lag, there is no tradeable edge here.
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
import backtest
import strategies as st
from rv_backtest import cached_local, lagged_signal, rebalance_every

IS_END = pd.Timestamp("2019-12-31")
OOS_START = pd.Timestamp("2020-01-01")
LAGS = [0, 1, 2, 3, 4]


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else np.nan


def main() -> int:
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    fits = pd.read_parquet(ROOT / "data" / "processed" / "cache_full" / "curve_fits.parquet")
    params = pd.read_parquet(ROOT / "data" / "processed" / "cache_full" / "curve_params.parquet")
    ss = st.SignalSet(panel, fits, params)
    sig = cached_local(ss, 6)

    all_dates = ss.ret.index
    reb = rebalance_every(all_dates, 1)  # daily
    rows = []
    for lag in LAGS:
        w = st.build_ls_weights(lagged_signal(sig, lag), ss.dur, ss.tradeable_liq,
                                reb, frac=0.2, exit_mult=1.0)
        res = backtest.run(panel, w)
        idx = res.ret.index
        for win, mask in [("IS", idx <= IS_END), ("OOS", idx >= OOS_START)]:
            rows.append({
                "lag": lag, "window": win,
                "gross": sharpe(res.ret_gross[mask]),
                "net": sharpe(res.ret[mask]),
            })
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "bounce_decomposition.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, win in zip(axes, ("IS", "OOS")):
        d = df[df["window"] == win]
        ax.plot(d["lag"], d["gross"], "o-", label="gross (before costs)", color="#c44")
        ax.plot(d["lag"], d["net"], "s-", label="net (after FedInvest costs)", color="#268")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(3, color="green", ls=":", lw=1, label="Sharpe = 3 target")
        ax.set_title(f"{win}: local-RV Sharpe vs execution lag")
        ax.set_xlabel("execution lag (trading days between signal and trade)")
        ax.set_xticks(LAGS)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("annualized Sharpe")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("The high Sharpe is bid-ask bounce: it lives only at lag 0 and "
                 "dies once you must trade a day later", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "bounce_decomposition.png", dpi=150)
    print("wrote results/bounce_decomposition.{png,csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
