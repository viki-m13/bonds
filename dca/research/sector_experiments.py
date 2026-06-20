"""Sector-cap experiments for SUMMIT, on the full 244-window grid.

  * diversify: force the two biweekly picks into different sectors.
  * sector_cap X%: annually trim any sector above X% of the book back to X%
    (sector analogue of the single-name trim), redeploying into current picks.

SUMMIT's book is tech-heavy (it rode NVDA/AAPL), so these caps test whether
sector diversification helps or hurts. Reported honestly vs the baseline.
Sectors: current GICS (yfinance), a stable proxy; 61/720 names Unknown (-1).
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

P = data.build_panel()
fd = protocol.get_shared()["fd"]
cols = list(fd.columns)
sec = json.load(open(os.path.join(os.path.dirname(P["close"].index.name or ""),
                                  "..", "data", "pit", "sectors.json"))) \
    if False else json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "pit", "sectors.json")))
uniq = sorted({v for v in sec.values() if v and v != "Unknown"})
sid = {s: i for i, s in enumerate(uniq)}
sector_ids = np.array([sid.get(sec.get(c, "Unknown"), -1) for c in cols])

S = strategy_dca.build_scores(P)
Snp = S.reindex(index=fd.index, columns=fd.columns).to_numpy(float)
wins, bench = protocol._bench_grid(10, 0, 1000.0, 5.0)


def grid(**kw):
    vq, vs, full = [], [], None
    for wname, s, e in wins:
        if wname in protocol.REGIMES:
            continue
        _, vals, inv = fast.run_fast(fd, Snp, k=2, every=10, start=s, end=e,
                                     cost_bps=5.0, sector_ids=sector_ids, **kw)
        if inv[0] <= 0:
            continue
        m = vals[0] / inv[0]
        vq.append(m / bench["qqq"][wname] - 1)
        vs.append(m / bench["spy"][wname] - 1)
        if wname.endswith("_end") and full is None:
            full = m
    vq, vs = np.array(vq), np.array(vs)
    # top-sector weight of final book
    _, _, _, hold = fast.run_fast(fd, Snp, k=2, every=10, start="2006-01-03",
                                  cost_bps=5.0, sector_ids=sector_ids,
                                  return_holdings=True, **kw)
    tot = sum(x for x in hold.values() if x == x and x > 0)
    sw = {}
    for t, x in hold.items():
        if x == x and x > 0:
            s_ = sec.get(t, "Unknown")
            sw[s_] = sw.get(s_, 0.0) + x / tot
    top_sec = max(sw.items(), key=lambda z: z[1])
    return {"win_qqq": float((vq > 0).mean()), "win_spy": float((vs > 0).mean()),
            "med": float(np.median(vq)), "p10": float(np.quantile(vq, .1)),
            "worst": float(vq.min()), "full": float(full),
            "top_sector": f"{top_sec[0]} {top_sec[1]*100:.0f}%"}


CONFIGS = [
    ("baseline", {}),
    ("diversify 2 sectors", {"diversify": True}),
    ("sector cap 50%/yr", {"sector_cap": 0.50, "trim_period": "annual"}),
    ("sector cap 40%/yr", {"sector_cap": 0.40, "trim_period": "annual"}),
    ("sector cap 30%/yr", {"sector_cap": 0.30, "trim_period": "annual"}),
    ("diversify + cap40", {"diversify": True, "sector_cap": 0.40,
                           "trim_period": "annual"}),
]

if __name__ == "__main__":
    print(f"{'config':22} {'winQQQ':>7} {'winSPY':>7} {'med':>7} {'p10':>7} "
          f"{'worst':>8} {'full':>7}  top sector")
    out = {}
    for name, kw in CONFIGS:
        r = grid(**kw)
        out[name] = r
        print(f"{name:22} {r['win_qqq']*100:6.0f}% {r['win_spy']*100:6.0f}% "
              f"{r['med']*100:+6.1f}% {r['p10']*100:+6.1f}% {r['worst']*100:+7.1f}% "
              f"{r['full']:6.1f}x  {r['top_sector']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__),
                                     "sector_experiments.json"), "w"),
              indent=1, default=str)
