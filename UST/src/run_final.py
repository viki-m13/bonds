#!/usr/bin/env python3
"""Run the FROZEN strategy config on the full panel, once.

Reads config/final_strategy.json (committed after IS selection), rebuilds
signals over the full 2010-2026 panel, runs the backtest, and reports
in-sample vs out-of-sample metrics separately. Also runs cost stresses and
a duration-matched benchmark. Writes results/final_results.json and plots.
"""
from __future__ import annotations

import json
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
import curve
import strategies as st

OOS_START = pd.Timestamp("2020-01-01")


def get_signalset(panel: pd.DataFrame, cache: Path) -> st.SignalSet:
    fits_f, params_f = cache / "curve_fits.parquet", cache / "curve_params.parquet"
    if fits_f.exists():
        fits, params = pd.read_parquet(fits_f), pd.read_parquet(params_f)
    else:
        print("fitting NSS curves per day (full panel)...")
        fits, params = curve.fit_panel(panel)
        cache.mkdir(parents=True, exist_ok=True)
        fits.to_parquet(fits_f, index=False)
        params.to_parquet(params_f)
    return st.SignalSet(panel, fits, params)


def make_signal(ss: st.SignalSet, cfg: dict) -> pd.DataFrame:
    kind = cfg["signal"]
    if kind == "value_z":
        return ss.sig_value(z_window=cfg["z_window"])
    if kind == "value_raw":
        return ss.sig_value_raw()
    if kind == "carry":
        return ss.sig_carry(roll_h=cfg["roll_h"])
    if kind == "momentum":
        return ss.sig_momentum(lookback=cfg["lookback"])
    raise ValueError(kind)


def run_sleeve(panel, ss, reb, sleeve: dict, cost_mult: float = 1.0) -> backtest.BtResult:
    sig = make_signal(ss, sleeve)
    w = st.build_ls_weights(
        sig, ss.dur, ss.tradeable_liq, reb,
        frac=sleeve["frac"], exit_mult=sleeve["exit_mult"],
    )
    return backtest.run(panel, w, cost_mult=cost_mult)


def split_metrics(ret: pd.Series, label: str) -> list[dict]:
    return [
        backtest.metrics(ret[ret.index < OOS_START], label=f"{label}_IS"),
        backtest.metrics(ret[ret.index >= OOS_START], label=f"{label}_OOS"),
        backtest.metrics(ret, label=f"{label}_FULL"),
    ]


def main() -> int:
    cfg = json.loads((ROOT / "config" / "final_strategy.json").read_text())
    print("frozen config:", cfg)

    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    ss = get_signalset(panel, ROOT / "data" / "processed" / "cache_full")
    reb = st.weekly_rebalance_dates(ss.ret.index)

    sleeves = [(s["weight"], run_sleeve(panel, ss, reb, s)) for s in cfg["sleeves"]]
    ret_net = sum(wt * r.ret for wt, r in sleeves)
    ret_gross = sum(wt * r.ret_gross for wt, r in sleeves)
    cost = sum(wt * r.cost for wt, r in sleeves)
    turnover = sum(wt * r.turnover for wt, r in sleeves)

    all_metrics = []
    all_metrics += split_metrics(ret_net, "net")
    all_metrics += split_metrics(ret_gross, "gross")
    for (wt, r), s in zip(sleeves, cfg["sleeves"]):
        all_metrics += split_metrics(r.ret, f"sleeve_{s['signal']}_net")
    sleeves2x = [(s["weight"], run_sleeve(panel, ss, reb, s, cost_mult=2.0)) for s in cfg["sleeves"]]
    ret2x = sum(wt * r.ret for wt, r in sleeves2x)
    all_metrics += split_metrics(ret2x, "net_2xcost")

    # benchmark: long-only equal-1/dur ladder of all liquid notes/bonds (the
    # passive way to hold this universe), reported as excess over cash
    bench_w = []
    for t in reb:
        if t not in ss.tradeable_liq.index:
            continue
        ok = ss.tradeable_liq.loc[t]
        names = ok[ok].index
        d = ss.dur.loc[t].reindex(names).clip(lower=0.5)
        bw = (1.0 / d) / (1.0 / d).sum()
        bench_w.append(pd.DataFrame({"date": t, "cusip": bw.index, "weight": bw.values}))
    bench = backtest.run(panel, pd.concat(bench_w, ignore_index=True))
    cash = backtest.cash_series(panel)
    bench_xs = (bench.ret - cash.reindex(bench.ret.index).fillna(0)).rename("bench")
    all_metrics += split_metrics(bench_xs, "ladder_excess")

    dfm = pd.DataFrame(all_metrics)
    print(dfm.to_string(index=False))

    out = {
        "config": cfg,
        "oos_start": str(OOS_START.date()),
        "ann_cost_full": round(float(cost.mean() * 252), 5),
        "avg_weekly_turnover": round(float(turnover.mean()), 4),
        "sleeve_corr_is": round(float(
            sleeves[0][1].ret[sleeves[0][1].ret.index < OOS_START].corr(
                sleeves[1][1].ret[sleeves[1][1].ret.index < OOS_START])), 3)
        if len(sleeves) == 2 else None,
        "sleeve_corr_oos": round(float(
            sleeves[0][1].ret[sleeves[0][1].ret.index >= OOS_START].corr(
                sleeves[1][1].ret[sleeves[1][1].ret.index >= OOS_START])), 3)
        if len(sleeves) == 2 else None,
        "metrics": all_metrics,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "final_results.json").write_text(json.dumps(out, indent=2))

    # plots
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    nav_net = (1 + ret_net).cumprod()
    nav_gross = (1 + ret_gross).cumprod()
    ax = axes[0]
    ax.plot(nav_net.index, nav_net, label="net of costs", lw=1.2)
    ax.plot(nav_gross.index, nav_gross, label="gross", lw=0.8, alpha=0.6)
    ax.axvline(OOS_START, color="red", ls="--", lw=1)
    ax.text(OOS_START, ax.get_ylim()[1], " OOS →", color="red", va="top")
    ax.set_title(f"Duration-neutral long-short on individual Treasuries — {cfg['name']}")
    ax.set_ylabel("NAV (start = 1)")
    ax.legend()
    ax.grid(alpha=0.3)
    dd = nav_net / nav_net.cummax() - 1
    axes[1].fill_between(dd.index, dd, 0, alpha=0.5)
    axes[1].axvline(OOS_START, color="red", ls="--", lw=1)
    axes[1].set_ylabel("drawdown")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "final_equity_curve.png", dpi=150)

    # rolling 1y sharpe
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    roll = ret_net.rolling(252)
    rs = roll.mean() / roll.std() * np.sqrt(252)
    ax2.plot(rs.index, rs, lw=1)
    ax2.axvline(OOS_START, color="red", ls="--", lw=1)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_title("Rolling 1-year Sharpe (net)")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(ROOT / "results" / "final_rolling_sharpe.png", dpi=150)
    print("wrote results/final_results.json and plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
