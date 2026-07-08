#!/usr/bin/env python3
"""IN-SAMPLE experiment harness. Only ever run with --end <= the IS cutoff
(2019-12-31). All data after --end is never loaded, so no OOS information
can leak into strategy selection. Results go to results/is_experiments.csv.
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
import curve
import strategies as st

IS_CUTOFF = "2019-12-31"


def load_slice(end: str) -> pd.DataFrame:
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    return panel[panel["date"] <= pd.Timestamp(end)].copy()


def get_signalset(panel: pd.DataFrame, cache: Path) -> st.SignalSet:
    fits_f, params_f = cache / "curve_fits.parquet", cache / "curve_params.parquet"
    if fits_f.exists() and params_f.exists():
        fits = pd.read_parquet(fits_f)
        params = pd.read_parquet(params_f)
    else:
        print("fitting NSS curves per day...")
        fits, params = curve.fit_panel(panel)
        cache.mkdir(parents=True, exist_ok=True)
        fits.to_parquet(fits_f, index=False)
        params.to_parquet(params_f)
    return st.SignalSet(panel, fits, params)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=IS_CUTOFF)
    args = ap.parse_args()
    if pd.Timestamp(args.end) > pd.Timestamp(IS_CUTOFF):
        raise SystemExit(f"refusing to run experiments past IS cutoff {IS_CUTOFF}")

    panel = load_slice(args.end)
    ss = get_signalset(panel, ROOT / "data" / "processed" / f"cache_is_{args.end}")
    reb = st.weekly_rebalance_dates(ss.ret.index)
    cash = backtest.cash_series(panel)

    grid = []
    for zw in (40, 60, 120):
        grid.append((f"value_z{zw}", lambda zw=zw: ss.sig_value(z_window=zw)))
    grid.append(("value_raw", ss.sig_value_raw))
    for rh in (0.5, 1.0):
        grid.append((f"carry_h{rh}", lambda rh=rh: ss.sig_carry(roll_h=rh)))
    for lb in (20, 60, 120):
        grid.append((f"mom_{lb}", lambda lb=lb: ss.sig_momentum(lookback=lb)))

    rows = []
    for name, fn in grid:
        sig = fn()
        for frac in (0.1, 0.2):
            for em in (1.0, 2.0, 3.0):
                w = st.build_ls_weights(
                    sig, ss.dur, ss.tradeable_liq, reb, frac=frac, exit_mult=em
                )
                res = backtest.run(panel, w)
                m = backtest.metrics(res.ret, label=f"{name}_f{frac}_x{em}")
                mg = backtest.metrics(res.ret_gross, label="")
                m["gross_sharpe"] = mg["sharpe"]
                m["ann_cost"] = round(float(res.cost.mean() * 252), 5)
                m["avg_turnover"] = round(float(res.turnover.mean()), 4)
                rows.append(m)
                print({k: m[k] for k in ("label", "sharpe", "gross_sharpe", "ann_ret", "ann_vol", "ann_cost", "avg_turnover")}, flush=True)

    df = pd.DataFrame(rows)
    outf = ROOT / "results" / "is_experiments.csv"
    outf.parent.mkdir(exist_ok=True)
    df.to_csv(outf, index=False)
    print(f"\nwrote {outf}")
    print(df.sort_values("sharpe", ascending=False).head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
