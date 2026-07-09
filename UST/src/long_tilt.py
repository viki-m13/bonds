#!/usr/bin/env python3
"""Long-only cheapness/carry tilt vs the passive duration-matched ladder.

The honest, investable framing: you are going to hold Treasuries anyway. Does
tilting a long-only ladder toward cheap / high-carry bonds add active return
net of costs? Because the tilt keeps (nearly) the ladder's duration profile,
the ACTIVE return (tilt minus ladder) isolates security selection, not a rates
bet. Monthly rebalance => low turnover => low cost.

Signal per bond (cross-sectional z, coupon notes/bonds, maturity >= 1y):
  cheap  = local cheapness residual (yield vs nearest-maturity neighbours)
  carry  = yield-to-maturity minus the fitted short rate, per unit duration
  score  = cheap_z + carry_z
Weights = ladder_weight * exp(gamma * score), renormalized; ladder_weight
propto 1/duration. Active series = tilt return - ladder return, net of costs.

IS by default; --oos runs the frozen gamma once on the full sample.
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

IS_END = pd.Timestamp("2019-12-31")
OOS_START = pd.Timestamp("2020-01-01")


def zx(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0


def monthly_dates(dates):
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).max().values)


def build(panel, gamma, lag=1):
    """Return (tilt_weights, ladder_weights) as long-format frames."""
    dates = np.array(sorted(panel["date"].unique()))
    pos = {d: i for i, d in enumerate(dates)}
    reb = monthly_dates(pd.DatetimeIndex(dates))
    tilt_rows, lad_rows = [], []
    lf = ROOT / "data" / "processed" / "cache_full" / "sig_local_k6.parquet"
    loc = pd.read_parquet(lf) if lf.exists() else None
    for t in reb:
        i = pos.get(t)
        if i is None or i + lag >= len(dates):
            continue
        td = dates[i + lag]
        day = panel[panel["date"] == td]
        d = day.dropna(subset=["ytm", "mod_dur", "tsy_years"])
        d = d[(d["tsy_years"] >= 1.0) & (d["tsy_years"] <= 30.0)]
        if len(d) < 25:
            continue
        dur = d["mod_dur"].clip(lower=0.5).values
        lad = (1.0 / dur) / (1.0 / dur).sum()
        cheap = d["cusip"].map(loc.loc[t]) if (loc is not None and t in loc.index) else pd.Series(0.0, index=d.index)
        cheap = pd.Series(np.asarray(cheap, float), index=d.index).fillna(0.0)
        short_rate = d["ytm"].quantile(0.05)
        carry = (d["ytm"].values - short_rate) / dur
        score = zx(cheap.values) + zx(pd.Series(carry).values)
        tw = lad * np.exp(gamma * score)
        tw = tw / tw.sum()
        cus = d["cusip"].values
        tilt_rows.append(pd.DataFrame({"date": td, "cusip": cus, "weight": tw}))
        lad_rows.append(pd.DataFrame({"date": td, "cusip": cus, "weight": lad}))
    return (pd.concat(tilt_rows, ignore_index=True),
            pd.concat(lad_rows, ignore_index=True))


def active_metrics(tilt, ladder, panel, mask_fn, label):
    rt = backtest.run(panel, tilt)
    rl = backtest.run(panel, ladder)
    act = (rt.ret - rl.ret)
    m = mask_fn(act.index)
    a = act[m].dropna()
    sh = float(a.mean() / a.std() * np.sqrt(252)) if a.std() > 0 else np.nan
    nav = (1 + a).cumprod()
    dd = float((nav / nav.cummax() - 1).min())
    return {"label": label, "active_sharpe": round(sh, 2),
            "active_ann": round(a.mean() * 252, 4),
            "active_vol": round(a.std() * np.sqrt(252), 4),
            "max_dd": round(dd, 4), "n": len(a),
            "tilt_cost": round(float(rt.cost[m].mean() * 252), 4)}, act


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", action="store_true")
    args = ap.parse_args()
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    if not args.oos:
        panel = panel[panel["date"] <= IS_END].copy()

    if not args.oos:
        print(f"{'gamma':>6} {'active_Sh':>9} {'active_ann':>10} {'vol':>7} {'cost':>6} {'maxDD':>7}")
        for gamma in (0.5, 1.0, 2.0, 3.0):
            tilt, lad = build(panel, gamma)
            m, _ = active_metrics(tilt, lad, panel, lambda idx: idx <= IS_END, f"g{gamma}")
            print(f"{gamma:>6} {m['active_sharpe']:>9} {m['active_ann']:>10} "
                  f"{m['active_vol']:>7} {m['tilt_cost']:>6} {m['max_dd']:>7}", flush=True)
    else:
        import json
        cfg = json.loads((ROOT / "config" / "long_tilt.json").read_text())
        tilt, lad = build(panel, cfg["gamma"])
        rows = []
        for lbl, fn in [("IS", lambda i: i <= IS_END), ("OOS", lambda i: i >= OOS_START),
                        ("FULL", lambda i: np.ones(len(i), bool))]:
            m, act = active_metrics(tilt, lad, panel, fn, lbl)
            rows.append(m)
        print(pd.DataFrame(rows).to_string(index=False))
        (ROOT / "results" / "long_tilt_result.json").write_text(
            json.dumps({"config": cfg, "metrics": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
