"""Follow-up to the cadence study:
  (1) PHASE robustness — run every schedule phase at each cadence (daily has 1
      phase, weekly 5, biweekly 10, monthly 21) and report the median and
      [min,max] of win_qqq and full multiple. Answers: is ROTATOR's 30x
      biweekly result a lucky offset, or robust across phases?
  (2) UNIVERSE — rebuild on the combined S&P 500 + Nasdaq-100 point-in-time
      universe (ROTATOR's native universe) and re-run the cadence sweep.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import fast
import protocol
import strategy_dca
import strategy_rotator as R

HERE = os.path.dirname(__file__)
CAD = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}


def phase_study():
    P = data.build_panel()
    S = strategy_dca.build_scores(P)
    RS, RSell = R.build_scores(P), R.build_sell(P)
    configs = [("SUMMIT", S, None, 2, 5.0), ("ROTATOR", RS, RSell, 3, 10.0)]
    out = {}
    print("=== PHASE ROBUSTNESS (every phase per cadence) ===")
    print(f"{'strategy':9} {'cadence':9} {'phases':>6} "
          f"{'winQQQ med[min,max]':>22} {'fullx med[min,max]':>22}")
    for name, sc, sell, k, cb in configs:
        for cname, ev in CAD.items():
            wins, mults = [], []
            for off in range(ev):
                c = protocol.evaluate_signal(sc, f"{name}_{cname}_{off}", k=k,
                                             every=ev, offset=off, cost_bps=cb,
                                             sell=sell, save=False, quiet=True)
                wins.append(c["win_qqq"]); mults.append(c["full_mult"])
            out[f"{name}|{cname}"] = {"win": wins, "mult": mults}
            print(f"{name:9} {cname:9} {ev:6d} "
                  f"{np.median(wins)*100:5.0f}% [{min(wins)*100:.0f},{max(wins)*100:.0f}]   "
                  f"  {np.median(mults):5.1f}x [{min(mults):.1f},{max(mults):.1f}]")
        print()
    json.dump(out, open(os.path.join(HERE, "phase_study.json"), "w"),
              indent=1, default=str)


def build_combined():
    """Union S&P 500 PIT and Nasdaq-100 PIT panels, aligned to the S&P index.
    Prefer S&P prices where a ticker is in both; member = in either index."""
    Psp = data.build_panel()
    Pn = data.build_panel_n100()
    idx = Psp["close"].index
    cols = Psp["close"].columns.union(Pn["close"].columns)
    comb = {}
    for f in ("open", "close", "volume"):
        a = Psp[f].reindex(index=idx, columns=cols)
        b = Pn[f].reindex(index=idx, columns=cols)
        comb[f] = a.where(a.notna(), b)
    ma = Psp["member"].reindex(index=idx, columns=cols, fill_value=False)
    mb = Pn["member"].reindex(index=idx, columns=cols, fill_value=False)
    comb["member"] = ma | mb
    return comb


def universe_study():
    comb = build_combined()
    extra = comb["close"].shape[1] - data.build_panel()["close"].shape[1]
    print(f"=== COMBINED S&P 500 + NASDAQ-100 UNIVERSE "
          f"({comb['close'].shape[1]} tickers, +{extra} vs S&P-only) ===")
    # swap protocol's shared cache to the combined universe
    protocol._cache.clear()
    protocol._cache["panels"] = comb
    protocol._cache["fd"] = fast.FastData(comb["open"], comb["close"],
                                          comb["member"])
    protocol._cache["qqq"] = data.load_benchmark("QQQ")
    protocol._cache["spy"] = data.load_benchmark("SPY")

    S = strategy_dca.build_scores(comb)
    RS, RSell = R.build_scores(comb), R.build_sell(comb)
    configs = [("SUMMIT", S, None, 2, 5.0), ("ROTATOR", RS, RSell, 3, 10.0)]
    out = {}
    print(f"{'strategy':9} {'cadence':9} {'winQQQ':>7} {'winSPY':>7} "
          f"{'medQQQ':>8} {'p10QQQ':>8} {'worstQQQ':>9} {'fullx':>7}")
    for name, sc, sell, k, cb in configs:
        for cname, ev in CAD.items():
            c = protocol.evaluate_signal(sc, f"{name}_{cname}_uni", k=k,
                                         every=ev, cost_bps=cb, sell=sell,
                                         save=False, quiet=True)
            out[f"{name}|{cname}"] = c
            print(f"{name:9} {cname:9} {c['win_qqq']*100:6.0f}% "
                  f"{c['win_spy']*100:6.0f}% {c['med_vs_qqq']*100:+7.1f}% "
                  f"{c['p10_vs_qqq']*100:+7.1f}% {c['worst_vs_qqq']*100:+8.1f}% "
                  f"{c['full_mult']:6.1f}")
        print()
    json.dump(out, open(os.path.join(HERE, "universe_study.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    phase_study()
    universe_study()
