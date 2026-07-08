#!/usr/bin/env python3
"""Real cost-charging backtests of the local-RV signal, to find the honest
ceiling and to prove the high paper-Sharpe is bid-ask bounce.

Two experiments, IS only unless --oos:
  1. Sweep rebalance frequency x frac x execution lag through the ACTUAL
     engine (charges FedInvest half-spread per side). If net Sharpe is high
     only at lag 0-1 daily and dies with lag/turnover control, it's bounce.
  2. Report gross vs net so the cost wipeout is explicit.

Execution lag L: weights formed from the signal L trading days before the
rebalance date (signal is staler, so any 1-day mark-reversion is excluded).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
ROOT = SRC.parent
import backtest
import strategies as st

IS_END = pd.Timestamp("2019-12-31")
OOS_START = pd.Timestamp("2020-01-01")


def cached_local(ss: st.SignalSet, k: int = 6) -> pd.DataFrame:
    f = ROOT / "data" / "processed" / "cache_full" / f"sig_local_k{k}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    print(f"computing local signal k={k} (full panel, one-time)...")
    sig = ss.sig_local(k=k)
    f.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(f)
    return sig


def lagged_signal(sig: pd.DataFrame, lag: int) -> pd.DataFrame:
    return sig.shift(lag) if lag > 0 else sig


def rebalance_every(dates: pd.DatetimeIndex, step: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates[::step])


def run_one(panel, ss, sig, dates, step, frac, lag, exit_mult, cost_mult=1.0):
    reb = rebalance_every(dates, step)
    slag = lagged_signal(sig, lag)
    w = st.build_ls_weights(slag, ss.dur, ss.tradeable_liq, reb,
                            frac=frac, exit_mult=exit_mult)
    return backtest.run(panel, w, cost_mult=cost_mult)


def summ(res, window_mask, label):
    r = res.ret[window_mask(res.ret.index)]
    rg = res.ret_gross[window_mask(res.ret_gross.index)]
    def sh(x):
        x = x.dropna()
        return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan
    return {
        "label": label,
        "net_sharpe": round(sh(r), 2),
        "gross_sharpe": round(sh(rg), 2),
        "net_ann": round(float(r.mean() * 252), 4),
        "ann_cost": round(float(res.cost[window_mask(res.cost.index)].mean() * 252), 4),
        "turnover": round(float(res.turnover.mean()), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    fits = pd.read_parquet(ROOT / "data" / "processed" / "cache_full" / "curve_fits.parquet")
    params = pd.read_parquet(ROOT / "data" / "processed" / "cache_full" / "curve_params.parquet")
    ss = st.SignalSet(panel, fits, params)
    sig = cached_local(ss, args.k)

    is_dates = ss.ret.index[ss.ret.index <= IS_END]
    is_mask = lambda idx: idx <= IS_END

    print(f"{'step':>4} {'frac':>4} {'lag':>3} {'exit':>4} "
          f"{'net_Sh':>7} {'gross_Sh':>8} {'net_ann':>8} {'cost':>7} {'turn':>6}")
    rows = []
    for step in (1, 2, 5):
        for frac in (0.1, 0.2):
            for lag in (0, 1, 2):
                for exit_mult in (1.0, 3.0):
                    res = run_one(panel, ss, sig, is_dates, step, frac, lag, exit_mult)
                    s = summ(res, is_mask, f"s{step}_f{frac}_l{lag}_x{exit_mult}")
                    rows.append(s)
                    print(f"{step:>4} {frac:>4} {lag:>3} {exit_mult:>4} "
                          f"{s['net_sharpe']:>7} {s['gross_sharpe']:>8} "
                          f"{s['net_ann']:>8} {s['ann_cost']:>7} {s['turnover']:>6}",
                          flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "rv_is_sweep.csv", index=False)
    print("\nwrote results/rv_is_sweep.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
