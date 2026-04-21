"""Finer grid around best BASTION params (IS only)."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

from bastion_strategy import (
    ALL_LEV, CASH, EQ_POOL, RATES_POOL, RA_POOL, TC_RATE,
    IS_START, IS_END, OOS_START, OOS_END,
    build_panels, build_fred, load_underlying_close,
    pick_sleeve, freeze_monthly, compute_killswitch, backtest, perf_metrics,
)

from bastion_grid import run_once


def main():
    universe = ALL_LEV + [CASH]
    opens, closes = build_panels(universe)
    idx = opens.index
    fred = build_fred(idx)
    spy = load_underlying_close("SPY", idx)
    tlt = load_underlying_close("TLT", idx)
    gld = load_underlying_close("GLD", idx)
    print("Loaded, fine grid starting...")

    results = []
    for mom_lb in [126, 189, 252]:
        for w_eq, w_rt, w_ra in [(0.50, 0.30, 0.20), (0.55, 0.25, 0.20), (0.60, 0.25, 0.15)]:
            for gross in [1.25, 1.5, 1.75, 2.0]:
                for hy_z_thr in [0.8, 1.0, 1.2, 1.5]:
                    for vix_thr in [25.0, 28.0, 30.0]:
                        for corr_thr in [0.2, 0.3, 0.4]:
                            for corr_window in [30, 60]:
                                m = run_once(
                                    opens, closes, idx, fred, spy, tlt, gld,
                                    mom_lb=mom_lb, sleeve_mode="invvol",
                                    w_eq=w_eq, w_rt=w_rt, w_ra=w_ra, gross=gross,
                                    hy_z_thr=hy_z_thr, vix_thr=vix_thr,
                                    corr_window=corr_window, corr_thr=corr_thr,
                                    trend_ma=200,
                                )
                                results.append({
                                    "mom_lb": mom_lb,
                                    "w_eq": w_eq, "w_rt": w_rt, "w_ra": w_ra,
                                    "gross": gross,
                                    "hy_z": hy_z_thr, "vix": vix_thr,
                                    "corr_w": corr_window, "corr_t": corr_thr,
                                    "is_sh": m["is"]["sharpe"], "is_cagr": m["is"]["cagr"],
                                    "is_vol": m["is"]["vol"], "is_mdd": m["is"]["mdd"],
                                    "full_sh": m["full"]["sharpe"], "full_cagr": m["full"]["cagr"],
                                    "oos_sh": m["oos"]["sharpe"], "oos_cagr": m["oos"]["cagr"],
                                    "turnover": m["turnover"],
                                })
    df = pd.DataFrame(results)
    df = df.sort_values("is_sh", ascending=False)
    out = Path("/home/user/bonds/data/results/bastion_grid2.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(df)} rows)")
    print("\nTop 20 by IS Sharpe:")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
