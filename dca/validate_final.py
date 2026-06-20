"""Finalist validation pipeline.

Given a signal builder f(panels)->scores (and optional sell builder), runs:
  1. truncation leakage audit (audit.py)
  2. reference-engine cross-check vs fast engine (3 windows)
  3. schedule-offset sweep (0..9) — cadence phase should not matter much
  4. cost sweep (5/10/20/40 bps)
  5. k sweep, biweekly vs monthly cadence
  6. NASDAQ-100 PIT universe transfer test (2015+), with its own benchmarks
  7. full window grid + regimes + random-control comparison

Produces a markdown + JSON report in research/final/.
"""
import json
import os

import numpy as np
import pandas as pd

import audit
import data as data_mod
import engine
import fast
import protocol

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "research", "final")
os.makedirs(OUT_DIR, exist_ok=True)


def validate(builder, name, k=3, every=10, cost_bps=5.0, sell_builder=None,
             skip_audit=False):
    report = {"name": name, "k": k, "every": every}
    P = data_mod.build_panel()
    scores = builder(P)
    sell = sell_builder(P) if sell_builder else None

    # 1. leakage audit
    if not skip_audit:
        print("== leakage audit ==")
        report["audit_ok"] = bool(audit.audit_builder(builder))
        if sell_builder:
            report["audit_sell_ok"] = bool(audit.audit_builder(sell_builder))

    # 2. reference engine cross-check
    print("== reference engine cross-check ==")
    fd = protocol.get_shared()["fd"]
    S = scores.reindex(index=fd.index, columns=fd.columns)
    checks = []
    for s, e in [("2007-01-03", "2015-01-02"), ("2012-01-03", "2020-01-02"),
                 ("2016-01-04", None)]:
        e_ts = pd.Timestamp(e) if e else P["close"].index[-1]
        res = engine.run_dca(P["open"], P["close"], S, P["member"], k=k,
                             every=every, start=s, end=e_ts,
                             cost_bps=cost_bps,
                             sell=sell.reindex(index=fd.index,
                                               columns=fd.columns)
                             if sell is not None else None)
        _, vals, inv = fast.run_fast(
            fd, S.to_numpy(float), k=k, every=every, start=s, end=e_ts,
            cost_bps=cost_bps,
            sell=sell.reindex(index=fd.index, columns=fd.columns)
            .fillna(False).to_numpy(bool) if sell is not None else None)
        checks.append({"start": s, "ref": res.final_multiple,
                       "fast": vals[0] / inv[0]})
        print(f"  {s}: ref={res.final_multiple:.4f} fast={vals[0]/inv[0]:.4f}")
    report["engine_check"] = checks

    # 3. offset sweep
    print("== offset sweep ==")
    report["offsets"] = {}
    for off in range(10):
        c = protocol.evaluate_signal(scores, f"{name}_off{off}", k=k,
                                     every=every, offset=off,
                                     cost_bps=cost_bps, sell=sell,
                                     save=False, quiet=True)
        report["offsets"][off] = {kk: c[kk] for kk in
                                  ("win_qqq", "win_spy", "med_vs_qqq",
                                   "worst_vs_qqq")}
        print(f"  off={off} win_qqq={c['win_qqq']:.0%} "
              f"med={c['med_vs_qqq']:+.1%} worst={c['worst_vs_qqq']:+.1%}")

    # 4. cost sweep
    print("== cost sweep ==")
    report["costs"] = {}
    for cb in (5, 10, 20, 40):
        c = protocol.evaluate_signal(scores, f"{name}_c{cb}", k=k,
                                     every=every, cost_bps=cb, sell=sell,
                                     save=False, quiet=True)
        report["costs"][cb] = {kk: c[kk] for kk in
                               ("win_qqq", "win_spy", "med_vs_qqq",
                                "worst_vs_qqq")}
        print(f"  {cb}bps win_qqq={c['win_qqq']:.0%} "
              f"med={c['med_vs_qqq']:+.1%}")

    # 5. k / cadence sweep
    print("== k & cadence sweep ==")
    report["k_cadence"] = {}
    for kk in (1, 2, 3, 4, 5):
        for ev in (10, 21):
            c = protocol.evaluate_signal(scores, f"{name}_k{kk}_e{ev}", k=kk,
                                         every=ev, cost_bps=cost_bps,
                                         sell=sell, save=False, quiet=True)
            report["k_cadence"][f"k{kk}_e{ev}"] = {
                m: c[m] for m in ("win_qqq", "win_spy", "med_vs_qqq",
                                  "worst_vs_qqq")}
            print(f"  k={kk} every={ev} win_qqq={c['win_qqq']:.0%} "
                  f"med={c['med_vs_qqq']:+.1%} worst={c['worst_vs_qqq']:+.1%}")

    # 6. N100 transfer
    print("== NASDAQ-100 transfer (2015+) ==")
    Pn = data_mod.build_panel_n100()
    sc_n = builder(Pn)
    sell_n = sell_builder(Pn) if sell_builder else None
    fdn = fast.FastData(Pn["open"], Pn["close"], Pn["member"])
    qqq = data_mod.load_benchmark("QQQ")
    spy = data_mod.load_benchmark("SPY")
    rows = []
    starts = pd.date_range("2016-01-01", "2023-04-01", freq="2QS")
    end = Pn["close"].index[-1]
    Sn = sc_n.reindex(index=fdn.index, columns=fdn.columns).to_numpy(float)
    selln = (sell_n.reindex(index=fdn.index, columns=fdn.columns)
             .fillna(False).to_numpy(bool) if sell_n is not None else None)
    rng = np.random.default_rng(5)
    for s in starts:
        _, vals, inv = fast.run_fast(fdn, Sn, k=k, every=every, start=s,
                                     end=end, cost_bps=cost_bps, sell=selln)
        (bq, biq), = fast.bench_fast(qqq.loc[:end], every=every, start=s,
                                     end=end, eval_dates=[end])
        (bs, bis), = fast.bench_fast(spy.loc[:end], every=every, start=s,
                                     end=end, eval_dates=[end])
        rmults = []
        for _ in range(10):
            R = rng.random(Sn.shape)
            _, rv, ri = fast.run_fast(fdn, R, k=k, every=every, start=s,
                                      end=end, cost_bps=cost_bps)
            rmults.append(rv[0] / ri[0])
        rows.append({"start": str(s.date()), "mult": vals[0] / inv[0],
                     "qqq": bq / biq, "spy": bs / bis,
                     "rand": float(np.mean(rmults))})
        print(f"  {s.date()} strat={rows[-1]['mult']:.2f} "
              f"qqq={rows[-1]['qqq']:.2f} rand={rows[-1]['rand']:.2f}")
    report["n100"] = rows

    # 7. headline grid + random control comparison
    print("== headline grid ==")
    card = protocol.evaluate_signal(scores, f"{name}_FINAL", k=k, every=every,
                                    cost_bps=cost_bps, sell=sell, save=True)
    report["headline"] = card

    with open(os.path.join(OUT_DIR, f"{name}_validation.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    return report
