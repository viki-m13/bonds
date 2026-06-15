"""APEX search — profitability-frontier scan for a higher-return DCA stock picker.

Goal (user mandate): the *most profitable* DCA strategy that still significantly
beats QQQ-DCA, validated honestly. SUMMIT sits at a robust optimum at k=2; this
script climbs the concentration/return frontier with a SMALL, pre-registered set
of literature-grounded levers, gated on an IS (2006-2014 starts) / OOS
(2015-2023 starts) split so we can see overfitting, and counts every trial.

Every lever has a citation in dca/research/literature_review_cited.md:
  * concentration k, size-tilt weight  -> Bessembinder 2018 (return concentration
    in a few mega-cap winners) + cap-weighted-benchmark gap.
  * momentum horizon tilt              -> Jegadeesh-Titman; long-horizon trend.
  * conviction (score-weighted) sizing -> double-down on the right tail.
  * winner-reinforcement               -> add-to-existing is the engine (panel).

Nothing here uses point-in-time fundamentals (we don't have them); the quality
overlay remains future work per the cited review.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import data as data_mod
import protocol
import fast


def eval_config(scores: pd.DataFrame, k=2, every=10, offset=0, cost_bps=5.0):
    """Run the full window grid once; return per-window dataframe with start
    dates and vs_qqq, plus split summaries (full / IS / OOS)."""
    sh = protocol.get_shared()
    fd = sh["fd"]
    S = scores.reindex(index=fd.index, columns=fd.columns).to_numpy(float)
    wins, bench = protocol._bench_grid(every, offset, 1000.0, cost_bps)
    rows = []
    for wname, s, e in wins:
        try:
            _, vals, inv = fast.run_fast(fd, S, k=k, every=every, offset=offset,
                                         start=s, end=e, cost_bps=cost_bps)
        except (ValueError, IndexError):
            continue
        if inv[0] <= 0:
            continue
        mult = vals[0] / inv[0]
        rows.append({"window": wname, "start": pd.Timestamp(s),
                     "mult": mult, "vs_qqq": mult / bench["qqq"][wname] - 1,
                     "vs_spy": mult / bench["spy"][wname] - 1})
    df = pd.DataFrame(rows)
    grid = df[~df["window"].isin(protocol.REGIMES)].copy()
    reg = df[df["window"].isin(protocol.REGIMES)].copy()
    return df, grid, reg


def summarize(grid, label=""):
    def block(sub):
        if not len(sub):
            return None
        return dict(n=len(sub),
                    win_qqq=float((sub["vs_qqq"] > 0).mean()),
                    med=float(sub["vs_qqq"].median()),
                    p10=float(sub["vs_qqq"].quantile(0.10)),
                    worst=float(sub["vs_qqq"].min()))
    full = block(grid)
    is_ = block(grid[grid["start"] < "2015-01-01"])
    oos = block(grid[grid["start"] >= "2015-01-01"])
    fullm = grid.loc[grid["window"].str.endswith("_end"), "mult"]
    out = dict(label=label, full=full, IS=is_, OOS=oos,
               full_mult=float(fullm.iloc[0]) if len(fullm) else None)
    return out


def line(s):
    f, i, o = s["full"], s["IS"], s["OOS"]
    return (f"{s['label']:<26} mult={s['full_mult']:5.1f}x | "
            f"full win={f['win_qqq']:.0%} med={f['med']:+.0%} "
            f"p10={f['p10']:+.0%} worst={f['worst']:+.0%} | "
            f"IS win={i['win_qqq']:.0%} OOS win={o['win_qqq']:.0%} "
            f"OOS med={o['med']:+.0%} OOS p10={o['p10']:+.0%}")
