"""Robustness gate for the candidate improvements from improve_experiments.py.
For each candidate: full-grid metrics, phase-offset spread (overfitting tell),
and IS (2006-2014 starts) vs OOS (2015-2023 starts) split. A real improvement
must hold across phases and in BOTH eras, not just the aggregate.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import protocol
import strategy_dca as S

P = data.build_panel()
close, vol = P["close"], P["volume"]
ret = close.pct_change(fill_method=None)


def xr(df):
    return df.rank(axis=1, pct=True)


SKIP = 21
mom_mh = sum(xr(close.shift(SKIP).pct_change(h - SKIP, fill_method=None))
             for h in (63, 126, 189, 252))
mom_mh_r = xr(mom_mh)
mom252 = xr(close.shift(SKIP).pct_change(252 - SKIP, fill_method=None))
size = xr((close * vol).rolling(63).mean())
frac50 = (close > close.rolling(50).mean()).rolling(126).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


rsi14 = rsi(close)
WS = 5.0


def compose(bull):
    bear = S.bear_scores(P)
    off = S.risk_off(P).to_numpy()[:, None]
    return pd.DataFrame(np.where(off, bear.to_numpy(float), bull.to_numpy(float)),
                        index=bull.index, columns=bull.columns)


CANDS = {
    "BASELINE": mom_mh_r + WS * size,
    "mom_12_1_only": mom252 + WS * size,
    "gate_rsi_lt80": (mom_mh_r + WS * size).where(rsi14 < 80),
    "12_1 + rsi<80": (mom252 + WS * size).where(rsi14 < 80),
    "mom_x_trendqual": xr(mom_mh_r + 0.5 * xr(frac50)) + WS * size,
}


def grid_split(scores):
    # phase spread
    wins, fulls, worsts = [], [], []
    for off in range(10):
        c = protocol.evaluate_signal(scores, "x", k=2, every=10, offset=off,
                                     save=False, quiet=True)
        wins.append(c["win_qqq"]); fulls.append(c["full_mult"])
        worsts.append(c["worst_vs_qqq"])
    # IS/OOS from the offset-0 windows scorecard
    import json
    c0 = protocol.evaluate_signal(scores, "tmp_split", k=2, every=10, save=True,
                                  quiet=True)
    w = pd.DataFrame(json.load(open(os.path.join(
        protocol.RESULTS_DIR, "tmp_split.json")))["windows"])
    g = w[~w["window"].isin(protocol.REGIMES)].copy()
    g["ys"] = g["start"].str[:4].astype(int)
    is_ = g[g.ys <= 2014]; oos = g[g.ys >= 2015]
    return {
        "win": np.median(wins), "win_rng": (min(wins), max(wins)),
        "worst": np.median(worsts),
        "full_med": np.median(fulls), "full_rng": (min(fulls), max(fulls)),
        "is_win": (is_["vs_qqq"] > 0).mean(), "is_med": is_["vs_qqq"].median(),
        "is_worst": is_["vs_qqq"].min(),
        "oos_win": (oos["vs_qqq"] > 0).mean(), "oos_med": oos["vs_qqq"].median(),
        "oos_worst": oos["vs_qqq"].min(),
    }


if __name__ == "__main__":
    print("Phase-robust + IS/OOS check (all biweekly k=2 5bps):\n")
    for name, bull in CANDS.items():
        r = grid_split(compose(bull))
        print(f"{name}")
        print(f"  phase: win {r['win']*100:.0f}% [{r['win_rng'][0]*100:.0f}-{r['win_rng'][1]*100:.0f}]  "
              f"worst(med) {r['worst']*100:+.1f}%  full {r['full_med']:.1f}x [{r['full_rng'][0]:.1f}-{r['full_rng'][1]:.1f}]")
        print(f"  IS (06-14): win {r['is_win']*100:.0f}%  med {r['is_med']*100:+.1f}%  worst {r['is_worst']*100:+.1f}%"
              f"   | OOS (15-23): win {r['oos_win']*100:.0f}%  med {r['oos_med']*100:+.1f}%  worst {r['oos_worst']*100:+.1f}%")
