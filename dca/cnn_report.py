"""Fair head-to-head report for the CNN signal.

Grades the walk-forward CNN scores against the momentum baselines and the
random-pick control on the *active-period* window grid (starts >= 2010, where
the CNN is fully out-of-sample), and renders a Markdown table into CNN.md.
"""
import os

import numpy as np
import pandas as pd

import data as data_mod
import fast
import protocol

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE_START = "2010-01-01"


def _win_grid(Smat, k, start_min):
    sh = protocol.get_shared()
    fd = sh["fd"]
    wins, bench = protocol._bench_grid(10, 0, 1000.0, 5.0)
    vq, vs = [], []
    for wname, s, e in wins:
        if wname in protocol.REGIMES or s < pd.Timestamp(start_min):
            continue
        try:
            _, vals, inv = fast.run_fast(fd, Smat, k=k, every=10, start=s, end=e,
                                         contribution=1000.0, cost_bps=5.0)
        except (ValueError, IndexError):
            continue
        if inv[0] <= 0:
            continue
        mult = vals[0] / inv[0]
        vq.append(mult / bench["qqq"][wname] - 1)
        vs.append(mult / bench["spy"][wname] - 1)
    vq, vs = np.array(vq), np.array(vs)
    return {"n": len(vq), "win_qqq": (vq > 0).mean(), "win_spy": (vs > 0).mean(),
            "med_qqq": np.median(vq), "worst_qqq": vq.min()}


def momentum_scores(P, skip=21, form=252):
    c = P["close"]
    return c.shift(skip).pct_change(form - skip, fill_method=None)


def main():
    P = data_mod.build_panel()
    sh = protocol.get_shared()
    fd = sh["fd"]
    S = pd.read_parquet(os.path.join(HERE, "research", "cnn_scores.parquet"))
    Scnn = S.reindex(index=fd.index, columns=fd.columns).to_numpy(float)
    Smom = (momentum_scores(P).reindex(index=fd.index, columns=fd.columns)
            .to_numpy(float))

    rng = np.random.default_rng(11)
    Srand = rng.random(Scnn.shape)

    rows = []
    for label, Smat, k in [("CNN k=2", Scnn, 2), ("CNN k=3", Scnn, 3),
                           ("CNN k=10", Scnn, 10),
                           ("9-1 momentum k=3", Smom, 3),
                           ("random k=3", Srand, 3)]:
        r = _win_grid(Smat, k, ACTIVE_START)
        rows.append((label, r))
        print(f"{label:20s} n={r['n']} win_qqq={r['win_qqq']:.0%} "
              f"win_spy={r['win_spy']:.0%} med_vs_qqq={r['med_qqq']:+.1%} "
              f"worst={r['worst_qqq']:+.1%}")

    hdr = ("| signal | win vs QQQ | win vs SPY | median vs QQQ | worst vs QQQ |\n"
           "|---|---|---|---|---|\n")
    body = "".join(
        f"| {lab} | {r['win_qqq']:.0%} | {r['win_spy']:.0%} | "
        f"{r['med_qqq']:+.1%} | {r['worst_qqq']:+.1%} |\n"
        for lab, r in rows)
    n = rows[0][1]["n"]
    table = (f"Active-period DCA grid ({n} windows, quarterly starts "
             f"{ACTIVE_START[:4]}+, horizons 3/5/10y + to-end, biweekly, "
             f"5 bps/trade):\n\n" + hdr + body)

    cnn_md = os.path.join(HERE, "CNN.md")
    txt = open(cnn_md).read()
    txt = txt.replace("<!-- RESULTS_TABLE -->", table)
    open(cnn_md, "w").write(txt)
    print("\nwrote results table into CNN.md")


if __name__ == "__main__":
    main()
