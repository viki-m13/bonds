"""How does SUMMIT perform if its universe is ETFs instead of stocks?

Three universes, all on the same 244-window grid vs QQQ/SPY DCA:
  1. sector ETFs only (XLK, XLF, ... ~20)
  2. all unlevered ETFs (broad / intl / bond / commodity / sector, ~220)
  3. all ETFs INCLUDING leveraged (TQQQ, SOXL, ...) and inverse (SQQQ, ...)

Caveats baked in: ETF universes are survivorship-biased (dead ETFs missing) so
each is checked against a random-pick control. Adding leveraged ETFs turns
SUMMIT into a leveraged momentum strategy — higher return, far higher risk —
which is the whole point of universe (3).
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ETF_DIRS = [os.path.join(ROOT, "data", "etfs_extended"),
            os.path.join(ROOT, "data", "etfs")]
CATS = json.load(open("/tmp/etf_cats.json")) if os.path.exists("/tmp/etf_cats.json") else None


def _load_etf(t):
    for d in ETF_DIRS:                       # prefer curated etfs_extended
        p = os.path.join(d, f"{t}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df = df[~df.index.duplicated()]
            return df
    return None


def build_etf_panel(tickers):
    cal = _load_etf("SPY").index                       # master trading calendar
    cal = cal[cal >= "2004-01-01"]
    o, c, v = {}, {}, {}
    for t in tickers:
        df = _load_etf(t)
        if df is None or "Close" not in df or len(df) < 300:
            continue
        df = df.reindex(cal)
        o[t], c[t], v[t] = df["Open"], df["Close"], df["Volume"]
    close = pd.DataFrame(c)
    member = close.notna()
    return {"open": pd.DataFrame(o), "close": close,
            "volume": pd.DataFrame(v), "member": member}


def swap(P):
    protocol._cache.clear()
    protocol._cache["panels"] = P
    protocol._cache["fd"] = fast.FastData(P["open"], P["close"], P["member"])
    protocol._cache["qqq"] = data.load_benchmark("QQQ")
    protocol._cache["spy"] = data.load_benchmark("SPY")


def study(P):
    s = strategy_dca.build_scores(P)
    card = protocol.evaluate_signal(s, "x", k=2, every=10, cost_bps=5.0,
                                    save=False, quiet=True)
    rc = protocol.random_control(k=2, every=10, cost_bps=5.0, n_draws=12)
    g = rc[~rc["window"].isin(protocol.REGIMES)]
    agg = g.groupby("window").agg(mult=("mult", "mean"), qqq=("qqq", "first"))
    ctrl = float((agg["mult"] > agg["qqq"]).mean())
    # current holdings
    fd = protocol._cache["fd"]
    Snp = s.reindex(index=fd.index, columns=fd.columns).to_numpy(float)
    _, vv, ii, hold = fast.run_fast(fd, Snp, k=2, every=10, start="2006-01-03",
                                    cost_bps=5.0, return_holdings=True)
    tot = sum(x for x in hold.values() if x == x and x > 0)
    top = sorted(((t, x / tot) for t, x in hold.items() if x == x and x > 0),
                 key=lambda z: -z[1])[:8]
    return card, ctrl, [(t, round(w * 100, 1)) for t, w in top]


if __name__ == "__main__":
    universes = [
        ("sector ETFs (~20)", CATS["sector"]),
        ("all unlevered ETFs (~220)", CATS["unlev"]),
        ("ALL ETFs + leveraged + inverse", CATS["all"]),
    ]
    out = {}
    for name, ticks in universes:
        P = build_etf_panel(ticks)
        swap(P)
        card, ctrl, top = study(P)
        out[name] = {"card": card, "ctrl": ctrl, "n": int(P["close"].shape[1]),
                     "top": top}
        print(f"\n=== {name}  ({P['close'].shape[1]} ETFs with data) ===")
        print(f"  SUMMIT: beat QQQ {card['win_qqq']*100:.0f}%  beat SPY {card['win_spy']*100:.0f}%  "
              f"median {card['med_vs_qqq']*100:+.1f}%  p10 {card['p10_vs_qqq']*100:+.1f}%  "
              f"worst {card['worst_vs_qqq']*100:+.1f}%  grew {card['full_mult']:.1f}x")
        print(f"  random-pick beats QQQ {ctrl*100:.0f}%   |  top holdings: {top}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__),
                                     "etf_universe_study.json"), "w"),
              indent=1, default=str)
