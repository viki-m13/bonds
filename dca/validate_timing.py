"""Validation suite for the TIMING overlay (strategy_timing.py).

Runs the tests from research/VALIDATION_METHODOLOGY.md that killed SUMMIT:
  * risk-adjusted metrics (Sortino / max-drawdown), not just total return;
  * benchmarks = QQQ-DCA AND the always-in momentum book;
  * cutoff-date trajectory (the recency killer);
  * rolling-window beat-rate (not cumulative-to-peak);
  * strict pre-2018 / 2018+ out-of-sample split.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as data_mod                 # noqa: E402
import strategy_timing as st            # noqa: E402

ANN = 252


def metrics(res):
    v, inv, r = res["value"], res["invested"], res["ret"]
    mult = float(v.iloc[-1] / inv.iloc[-1])
    growth = (1 + r).cumprod()
    dd = float((growth / growth.cummax() - 1).min())
    mu, sd = r.mean() * ANN, r.std() * np.sqrt(ANN)
    downside = r[r < 0].std() * np.sqrt(ANN)
    sharpe = mu / sd if sd else np.nan
    sortino = mu / downside if downside else np.nan
    return {"mult": mult, "ann_ret": mu, "ann_vol": sd, "sharpe": sharpe,
            "sortino": sortino, "max_dd": dd}


def run_all(P, ron, k=3, start="2010-01-01", end=None):
    out = {}
    for mode in ("timing", "always", "qqq", "bond"):
        out[mode] = st.backtest(P, ron, k=k, start=start, end=end, mode=mode)
    return out


def main():
    P = data_mod.build_panel()
    path = os.path.join(HERE, "research", "timing_riskon.parquet")
    if os.path.exists(path):
        ron = pd.read_parquet(path)["riskon"]
    else:
        ron = st.build_riskon(P, k=3)
        ron.to_frame("riskon").to_parquet(path)

    print("\n=== Risk-adjusted metrics (active period 2010+, k=3, 5bps) ===")
    res = run_all(P, ron, start="2010-01-01")
    hdr = f"{'strategy':10s} {'mult':>6s} {'annRet':>7s} {'annVol':>7s} " \
          f"{'Sharpe':>7s} {'Sortino':>8s} {'maxDD':>7s}"
    print(hdr)
    for mode in ("timing", "always", "qqq", "bond"):
        m = metrics(res[mode])
        print(f"{mode:10s} {m['mult']:6.2f} {m['ann_ret']:+7.1%} "
              f"{m['ann_vol']:7.1%} {m['sharpe']:7.2f} {m['sortino']:8.2f} "
              f"{m['max_dd']:7.1%}")

    print("\n=== OOS split (does the edge persist out-of-sample?) ===")
    for lo, hi, tag in [("2010-01-01", "2017-12-31", "IN  2010-2017"),
                        ("2018-01-01", None, "OOS 2018-2026")]:
        r = run_all(P, ron, start=lo, end=hi)
        mt, ma, mq = metrics(r["timing"]), metrics(r["always"]), metrics(r["qqq"])
        print(f"  {tag}: timing Sortino {mt['sortino']:.2f} / maxDD "
              f"{mt['max_dd']:+.0%} | always {ma['sortino']:.2f} / "
              f"{ma['max_dd']:+.0%} | qqq {mq['sortino']:.2f} / {mq['max_dd']:+.0%} "
              f"| mult t/a/q {mt['mult']:.2f}/{ma['mult']:.2f}/{mq['mult']:.2f}")

    print("\n=== Cutoff-date trajectory (start 2010, ratio vs QQQ) ===")
    print(f"  {'cutoff':8s} {'timing':>7s} {'always':>7s} {'qqq':>7s} "
          f"{'t/qqq':>7s} {'a/qqq':>7s}")
    for cut in ("2014-12-31", "2017-12-31", "2019-12-31", "2021-12-31",
                "2023-12-31", "2025-12-31", None):
        r = run_all(P, ron, start="2010-01-01", end=cut)
        mt = float(r["timing"]["value"].iloc[-1] / r["timing"]["invested"].iloc[-1])
        ma = float(r["always"]["value"].iloc[-1] / r["always"]["invested"].iloc[-1])
        mq = float(r["qqq"]["value"].iloc[-1] / r["qqq"]["invested"].iloc[-1])
        lab = "end" if cut is None else cut[:4]
        print(f"  {lab:8s} {mt:7.2f} {ma:7.2f} {mq:7.2f} "
              f"{mt/mq:7.2f} {ma/mq:7.2f}")

    print("\n=== Rolling-window beat-rate (quarterly starts, 3y & 5y) ===")
    idx = P["close"].index
    starts = pd.date_range("2010-01-01", idx[-1] - pd.DateOffset(years=3),
                           freq="QS")
    for horizon in (3, 5):
        wins_t_q, wins_t_a = [], []
        for s in starts:
            e = s + pd.DateOffset(years=horizon)
            if e > idx[-1]:
                continue
            r = run_all(P, ron, start=str(s.date()), end=str(e.date()))
            mt = r["timing"]["value"].iloc[-1] / r["timing"]["invested"].iloc[-1]
            ma = r["always"]["value"].iloc[-1] / r["always"]["invested"].iloc[-1]
            mq = r["qqq"]["value"].iloc[-1] / r["qqq"]["invested"].iloc[-1]
            wins_t_q.append(mt > mq); wins_t_a.append(mt > ma)
        n = len(wins_t_q)
        print(f"  {horizon}y (n={n}): timing beats QQQ {np.mean(wins_t_q):.0%} | "
              f"timing beats always-momentum {np.mean(wins_t_a):.0%}")


if __name__ == "__main__":
    main()
