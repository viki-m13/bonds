"""Does a broader large+mid-cap universe (Russell-1000-style) beat the clean
point-in-time S&P 500 for SUMMIT?

Universe = S&P 500 PIT (clean, PIT membership) UNION ~265 mid-caps + NASDAQ-100
names (OHLCV from Yahoo). IMPORTANT: only the S&P names carry point-in-time
membership; the mid-cap / Nasdaq names are eligible whenever they have price
data, i.e. they are SURVIVORSHIP-BIASED (only names that survived to today are
in the file). So any apparent improvement must be checked against the
random-pick control on the *same* universe, which carries the identical bias.

Reports SUMMIT and a random-pick control on (a) S&P-500 PIT baseline and
(b) the broad universe, on the same 244-window grid.
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
EXTRA_DIRS = [os.path.join(ROOT, "data", "pit", "prices_broad"),
              os.path.join(ROOT, "data", "pit", "prices_n100")]


def build_broad():
    Psp = data.build_panel()
    idx = Psp["close"].index
    frames = {f: {} for f in ("open", "close", "volume")}
    for d in EXTRA_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".csv"):
                continue
            t = fn[:-4]
            df = pd.read_csv(os.path.join(d, fn), index_col=0, parse_dates=True)
            df = df[~df.index.duplicated()]
            gaps = df.index.to_series().diff().dt.days
            brk = gaps[gaps > 30]
            if len(brk):
                df = df.loc[:brk.index[0] - pd.Timedelta(days=1)]
            df = data._clean_prices(df)
            if len(df) < 252:
                continue
            for f in ("open", "close", "volume"):
                col = f.capitalize()
                if col in df.columns:
                    frames[f][t] = df[col]
    extra = {f: pd.DataFrame(frames[f]).reindex(idx) for f in frames}
    allcols = Psp["close"].columns.union(extra["close"].columns)
    comb = {}
    for f in ("open", "close", "volume"):
        a = Psp[f].reindex(columns=allcols)
        b = extra[f].reindex(columns=allcols)
        comb[f] = a.where(a.notna(), b)
    msp = Psp["member"].reindex(columns=allcols, fill_value=False)
    nonsp = [c for c in allcols if c not in Psp["close"].columns]
    em = pd.DataFrame(False, index=idx, columns=allcols)
    em[nonsp] = comb["close"][nonsp].notna()
    comb["member"] = msp | em
    return comb, len(nonsp)


def swap(P):
    protocol._cache.clear()
    protocol._cache["panels"] = P
    protocol._cache["fd"] = fast.FastData(P["open"], P["close"], P["member"])
    protocol._cache["qqq"] = data.load_benchmark("QQQ")
    protocol._cache["spy"] = data.load_benchmark("SPY")


def summit_card(P):
    s = strategy_dca.build_scores(P)
    return protocol.evaluate_signal(s, "x", k=2, every=10, cost_bps=5.0,
                                    save=False, quiet=True)


def control(P, n=12):
    rc = protocol.random_control(k=2, every=10, cost_bps=5.0, n_draws=n)
    g = rc[~rc["window"].isin(protocol.REGIMES)]
    agg = g.groupby("window").agg(mult=("mult", "mean"), qqq=("qqq", "first"),
                                  spy=("spy", "first"))
    return float((agg["mult"] > agg["qqq"]).mean()), \
        float((agg["mult"] / agg["qqq"] - 1).median())


def show(name, card, ctrl):
    print(f"{name:28} SUMMIT: win_qqq {card['win_qqq']*100:4.0f}%  "
          f"win_spy {card['win_spy']*100:4.0f}%  med {card['med_vs_qqq']*100:+5.1f}%  "
          f"p10 {card['p10_vs_qqq']*100:+5.1f}%  worst {card['worst_vs_qqq']*100:+6.1f}%  "
          f"full {card['full_mult']:5.1f}x   | random beats QQQ {ctrl[0]*100:3.0f}% "
          f"(med {ctrl[1]*100:+.0f}%)")


if __name__ == "__main__":
    out = {}
    print("Building baseline (S&P 500 PIT)...")
    Psp = data.build_panel()
    swap(Psp)
    c_sp, ctrl_sp = summit_card(Psp), control(Psp)
    out["sp500_pit"] = {"card": c_sp, "ctrl": ctrl_sp}

    print("Building broad universe...")
    Pb, n_extra = build_broad()
    print(f"  broad universe: {Pb['close'].shape[1]} tickers "
          f"(+{n_extra} non-S&P, survivorship-biased)")
    swap(Pb)
    c_b, ctrl_b = summit_card(Pb), control(Pb)
    out["broad"] = {"card": c_b, "ctrl": ctrl_b, "n_extra": n_extra,
                    "n_total": int(Pb["close"].shape[1])}

    print()
    show("S&P 500 PIT (clean)", c_sp, ctrl_sp)
    show(f"Broad large+mid (survivorship)", c_b, ctrl_b)
    print("\nSUMMIT edge over its own random control (skill, bias-robust):")
    print(f"  S&P 500 PIT : win {c_sp['win_qqq']*100:.0f}% vs control {ctrl_sp[0]*100:.0f}%"
          f"  -> +{(c_sp['win_qqq']-ctrl_sp[0])*100:.0f}pp")
    print(f"  Broad       : win {c_b['win_qqq']*100:.0f}% vs control {ctrl_b[0]*100:.0f}%"
          f"  -> +{(c_b['win_qqq']-ctrl_b[0])*100:.0f}pp")
    json.dump(out, open(os.path.join(os.path.dirname(__file__),
                                     "universe_broad_study.json"), "w"),
              indent=1, default=str)
