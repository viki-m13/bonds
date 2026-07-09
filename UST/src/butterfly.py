#!/usr/bin/env python3
"""Curve butterfly (2y-5y-10y) mean-reversion, traded with individual bonds.

Diagnostics showed the fitted-curve curvature mean-reverts (weekly change
autocorr ~ -0.37). This strategy harvests that with cash Treasuries:

- Each week, measure the butterfly spread  fly = 2*y_belly - y_2y - y_10y
  from median yields of bonds in maturity windows around 5y / 2y / 10y.
- z-score fly over a rolling window (info up to the rebalance date only).
- When the belly is CHEAP (fly high, z>0): go LONG a belly basket and SHORT
  2y + 10y baskets, DV01-neutral with a 50/50 wing split, so it is a pure
  curvature bet (no level or slope exposure). Position size ~ the z-score,
  capped. Mean-reversion => belly richens, fly falls, trade profits.
- Execution lag L: trade L days after the signal date (default 1) so no
  same-close (bounce) fill. Weekly hold. FedInvest half-spread costs charged.

Runs IS by default; --oos runs the frozen config once on the full sample.
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

BELLY = (4.0, 6.5)
WING_S = (1.5, 2.75)
WING_L = (8.5, 11.5)


def basket(day, lo, hi):
    b = day[(day["tsy_years"] >= lo) & (day["tsy_years"] <= hi)]
    return b


def build_fly(panel):
    """Daily butterfly spread (bp) and per-basket member lists."""
    rows = []
    for date, day in panel.groupby("date"):
        d = day.dropna(subset=["ytm", "mod_dur"])
        bs, w2, w10 = basket(d, *BELLY), basket(d, *WING_S), basket(d, *WING_L)
        if len(bs) < 2 or len(w2) < 1 or len(w10) < 1:
            continue
        fly = (2 * bs["ytm"].median() - w2["ytm"].median() - w10["ytm"].median()) * 1e4
        rows.append((date, fly))
    return pd.Series(dict(rows)).sort_index().rename("fly")


def weekly_dates(dates):
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("W-WED")).max().values)


def build_weights(panel, fly, z_window, lag, cap, name_cap=0.34):
    dates = np.array(sorted(panel["date"].unique()))
    pos = {d: i for i, d in enumerate(dates)}
    z = (fly - fly.rolling(z_window, min_periods=z_window // 2).mean()) \
        / fly.rolling(z_window, min_periods=z_window // 2).std()
    reb = weekly_dates(pd.DatetimeIndex(dates))
    out = []
    for t in reb:
        if t not in z.index or not np.isfinite(z.loc[t]):
            continue
        i = pos.get(t)
        if i is None or i + lag >= len(dates):
            continue
        td = dates[i + lag]  # trade date
        day = panel[panel["date"] == td]
        d = day.dropna(subset=["ytm", "mod_dur"])
        bs, w2, w10 = basket(d, *BELLY), basket(d, *WING_S), basket(d, *WING_L)
        if len(bs) < 2 or len(w2) < 1 or len(w10) < 1:
            continue
        sign = float(np.clip(z.loc[t], -cap, cap))  # >0 belly cheap => long belly

        def eqw(b):
            w = pd.Series(1.0 / len(b), index=b["cusip"].values)
            return w, float((w.values * b["mod_dur"].values).sum())
        wb, Db = eqw(bs)
        w2w, D2 = eqw(w2)
        w10w, D10 = eqw(w10)
        # long belly notional 1 (times sign); wings short, 50/50 DV01 split
        belly_dv01 = Db
        s2 = 0.5 * belly_dv01 / D2
        s10 = 0.5 * belly_dv01 / D10
        w = {}
        for c, x in wb.items():
            w[c] = w.get(c, 0.0) + sign * x
        for c, x in w2w.items():
            w[c] = w.get(c, 0.0) - sign * s2 * x
        for c, x in w10w.items():
            w[c] = w.get(c, 0.0) - sign * s10 * x
        ws = pd.Series(w)
        g = ws.abs().sum()
        if g > 0:
            ws = ws * (2.0 / g)  # scale to gross 2
        ws = ws.clip(-name_cap, name_cap)
        out.append(pd.DataFrame({"date": td, "cusip": ws.index, "weight": ws.values}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "cusip", "weight"])


def metrics(ret, mask, label):
    r = ret[mask].dropna()
    sh = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    nav = (1 + r).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    return {"label": label, "sharpe": round(sh, 2), "ann_ret": round(r.mean() * 252, 4),
            "ann_vol": round(r.std() * np.sqrt(252), 4), "max_dd": round(float(dd), 4),
            "n": len(r)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", action="store_true")
    args = ap.parse_args()

    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    if not args.oos:
        panel = panel[panel["date"] <= IS_END].copy()
    panel = panel[panel["sec_type"].isin(["MARKET BASED NOTE", "MARKET BASED BOND"])]
    fly = build_fly(panel)
    print(f"fly built: {len(fly)} days, mean {fly.mean():.1f}bp")

    if not args.oos:
        print(f"\n{'zwin':>5} {'lag':>3} {'cap':>4} {'net_Sh':>7} {'gross_Sh':>8} "
              f"{'ann':>7} {'vol':>6} {'cost':>6}")
        for zw in (26, 52, 104):
            for lag in (1, 2):
                for cap in (1.5, 2.5):
                    w = build_weights(panel, fly, zw, lag, cap)
                    if len(w) == 0:
                        continue
                    res = backtest.run(panel, w)
                    m = metrics(res.ret, res.ret.index <= IS_END, "n")
                    mg = metrics(res.ret_gross, res.ret_gross.index <= IS_END, "g")
                    cost = res.cost.mean() * 252
                    print(f"{zw:>5} {lag:>3} {cap:>4} {m['sharpe']:>7} {mg['sharpe']:>8} "
                          f"{m['ann_ret']:>7} {m['ann_vol']:>6} {cost:>6.4f}", flush=True)
    else:
        import json
        cfg = json.loads((ROOT / "config" / "butterfly.json").read_text())
        w = build_weights(panel, fly, cfg["z_window"], cfg["lag"], cfg["cap"])
        res = backtest.run(panel, w)
        idx = res.ret.index
        rows = [metrics(res.ret, idx <= IS_END, "net_IS"),
                metrics(res.ret, idx >= OOS_START, "net_OOS"),
                metrics(res.ret, np.ones(len(idx), bool), "net_FULL"),
                metrics(res.ret_gross, idx >= OOS_START, "gross_OOS")]
        print(pd.DataFrame(rows).to_string(index=False))
        (ROOT / "results" / "butterfly_result.json").write_text(
            json.dumps({"config": cfg, "metrics": rows,
                        "ann_cost": round(float(res.cost.mean() * 252), 4)}, indent=2))
        res.ret.to_frame("ret").to_parquet(ROOT / "results" / "butterfly_ret.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
