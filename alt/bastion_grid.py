"""Fast IS-only grid for BASTION params. Loads data once, runs grid in-memory."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import itertools as it

from bastion_strategy import (
    ALL_LEV, CASH, EQ_POOL, RATES_POOL, RA_POOL, TC_RATE,
    IS_START, IS_END, OOS_START, OOS_END,
    build_panels, build_fred, load_underlying_close,
    pick_sleeve, freeze_monthly, compute_killswitch, backtest, perf_metrics,
)


def run_once(opens, closes, idx, fred, spy, tlt, gld,
             mom_lb=126, sma_lb=200, w_eq=0.4, w_rt=0.4, w_ra=0.2,
             gross=1.5, sleeve_mode="best",
             hy_z_thr=1.2, vix_thr=28.0, corr_window=60, corr_thr=0.4, smooth=5,
             trend_ma=150):
    eq_daily = pick_sleeve(closes, EQ_POOL, mom_lb, sma_lb, mode=sleeve_mode)
    rt_daily = pick_sleeve(closes, RATES_POOL, mom_lb, sma_lb, mode=sleeve_mode)
    ra_daily = pick_sleeve(closes, RA_POOL, mom_lb, sma_lb, mode=sleeve_mode)

    eq_w = freeze_monthly(eq_daily, idx)
    rt_w = freeze_monthly(rt_daily, idx)
    ra_w = freeze_monthly(ra_daily, idx)

    spy_lag = spy.shift(1); tlt_lag = tlt.shift(1); gld_lag = gld.shift(1)
    eq_trend = (spy_lag > spy_lag.rolling(trend_ma).mean()).astype(float).fillna(0.0)
    rt_trend = (tlt_lag > tlt_lag.rolling(trend_ma).mean()).astype(float).fillna(0.0)
    ra_trend = (gld_lag > gld_lag.rolling(trend_ma).mean()).astype(float).fillna(0.0)
    eq_w = eq_w.mul(eq_trend, axis=0).fillna(0.0)
    rt_w = rt_w.mul(rt_trend, axis=0).fillna(0.0)
    ra_w = ra_w.mul(ra_trend, axis=0).fillna(0.0)

    eq_final = eq_w.mul(w_eq * gross)
    rt_final = rt_w.mul(w_rt * gross)
    ra_final = ra_w.mul(w_ra * gross)

    part, diag, triggers = compute_killswitch(
        fred, spy, tlt, hy_z_thr=hy_z_thr, vix_thr=vix_thr,
        corr_window=corr_window, corr_thr=corr_thr, smooth=smooth)
    on_mult = part.shift(1).fillna(0.0)

    universe = ALL_LEV + [CASH]
    weights = pd.DataFrame(0.0, index=idx, columns=universe)
    for col in EQ_POOL:
        weights[col] = eq_final[col] * on_mult
    for col in RATES_POOL:
        weights[col] = rt_final[col] * on_mult
    for col in RA_POOL:
        weights[col] = ra_final[col] * on_mult
    cash_w = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    weights[CASH] = cash_w

    bt = backtest(opens, weights, tc_rate=TC_RATE).loc[IS_START:OOS_END]
    net = bt["net_ret"]
    is_r = net.loc[IS_START:IS_END]
    oos_r = net.loc[OOS_START:OOS_END]
    return {
        "is": perf_metrics(is_r),
        "oos": perf_metrics(oos_r),
        "full": perf_metrics(net),
        "turnover": float(bt["turnover"].sum() / ((net.index[-1] - net.index[0]).days / 365.25)),
    }


def main():
    universe = ALL_LEV + [CASH]
    opens, closes = build_panels(universe)
    idx = opens.index
    fred = build_fred(idx)
    spy = load_underlying_close("SPY", idx)
    tlt = load_underlying_close("TLT", idx)
    gld = load_underlying_close("GLD", idx)
    print("Loaded panels, grid starting...")

    results = []
    for mom_lb in [63, 126, 189, 252]:
        for mode in ["best", "invvol"]:
            for w_eq, w_rt, w_ra in [
                (0.40, 0.40, 0.20), (0.35, 0.35, 0.30),
                (0.50, 0.30, 0.20), (0.50, 0.25, 0.25),
                (0.60, 0.25, 0.15), (0.45, 0.35, 0.20),
            ]:
                for gross in [1.0, 1.25, 1.5, 1.75]:
                    for trend_ma in [100, 150, 200]:
                        m = run_once(
                            opens, closes, idx, fred, spy, tlt, gld,
                            mom_lb=mom_lb, w_eq=w_eq, w_rt=w_rt, w_ra=w_ra,
                            gross=gross, sleeve_mode=mode, trend_ma=trend_ma,
                        )
                        results.append({
                            "mom_lb": mom_lb, "mode": mode,
                            "w_eq": w_eq, "w_rt": w_rt, "w_ra": w_ra,
                            "gross": gross, "trend_ma": trend_ma,
                            "is_sh": m["is"]["sharpe"], "is_cagr": m["is"]["cagr"],
                            "is_vol": m["is"]["vol"], "is_mdd": m["is"]["mdd"],
                            "full_sh": m["full"]["sharpe"], "full_cagr": m["full"]["cagr"],
                            "oos_sh": m["oos"]["sharpe"], "oos_cagr": m["oos"]["cagr"],
                            "turnover": m["turnover"],
                        })
    df = pd.DataFrame(results)
    df = df.sort_values("is_sh", ascending=False)
    out = Path("/home/user/bonds/data/results/bastion_grid.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(df)} rows)")
    print("\nTop 25 by IS Sharpe (IS only — no OOS peek):")
    print(df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
