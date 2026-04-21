"""Robustness check 1 — inv-vol & momentum on the FULL 17-LETF universe.

The headline results in the earlier sweep used "core6" — a hand-picked
basket of UPRO/TQQQ/SOXL/TECL/TMF/UGL. That's the winning basket in
hindsight. This script re-runs the same strategies on:

  A. core6  (6 tickers)        — hand-picked basket
  B. noeq  (SSO/QLD + bonds/gold, the conservative picks)
  C. tech4 (TQQQ/TECL/SOXL/UPRO)
  D. all17 (every long LETF)   — no cherry-picking
  E. all17+BTC
  F. all17+BTC+ETH
  G. long_noenergy (drop UCO/ERX which crashed)
  H. long_defensive (TMF/UGL/TYD/UBT/NUGT only)

Strategies: inv-vol {lb=63, 126}, momentum top-{3,4,5}, all at 21-day rebal.

The point: show whether core6's dominance is the basket or the strategy.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise)
from letf_crypto_universe import load_with_crypto
from letf_universe import LETF_LONG_2011

OUT = Path("/home/user/bonds/data/results")


def invvol_fn(tickers, lookback):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        inv = 1 / r.std().replace(0, np.nan).fillna(0)
        w = inv / inv.sum()
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


def mom_fn(tickers, lookback, top_n):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        cum = (1 + r).prod() - 1
        picks = cum.sort_values(ascending=False).head(top_n).index.tolist()
        out = pd.Series(0.0, index=hist.columns)
        if picks:
            out.loc[picks] = 1.0 / len(picks)
        return out
    return fn


BASKETS = {
    "core6":         ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"],
    "tech4":         ["UPRO","TQQQ","SOXL","TECL"],
    "long_noenergy": [t for t in LETF_LONG_2011 if t not in ("UCO","ERX","NUGT")],
    "long_defensiv": ["TMF","UGL","TYD","UBT"],
    "all17":         LETF_LONG_2011,
    "all17+BTC":     LETF_LONG_2011 + ["BTC_USD"],
    "all17+BTC+ETH": LETF_LONG_2011 + ["BTC_USD","ETH_USD"],
}


def main():
    # Biggest universe at 2011 start (no crypto, 17 LETFs)
    px_a = load_with_crypto([], start="2011-01-01")
    rets_a = common_window_returns(px_a)
    # Crypto-inclusive 2015 start
    px_b = load_with_crypto(["BTC_USD"], start="2015-01-01")
    rets_b = common_window_returns(px_b)
    # Crypto (B+E) 2018 start
    px_c = load_with_crypto(["BTC_USD","ETH_USD"], start="2018-01-01")
    rets_c = common_window_returns(px_c)

    rows = []
    # LETF-only tests on 2011 window
    for bname, ts in BASKETS.items():
        if any(t not in rets_a.columns for t in ts): continue
        for lb in (63, 126):
            r, _ = run_backtest(rets_a, invvol_fn(ts, lb),
                                rebal_days=21, exec_lag=1)
            s = summarise(r, f"invvol {bname} lb={lb}")
            s["basket"] = bname; s["family"] = "invvol"
            s["window"] = "2011-2026"; rows.append(s)

        for n in (3, 4, 5):
            if len(ts) < n: continue
            r, _ = run_backtest(rets_a, mom_fn(ts, 126, n),
                                rebal_days=21, exec_lag=1)
            s = summarise(r, f"mom top{n} {bname} lb=126")
            s["basket"] = bname; s["family"] = "mom"
            s["window"] = "2011-2026"; rows.append(s)

    # Crypto-inclusive variants on 2015 window
    for bname, ts in BASKETS.items():
        if any(t not in rets_b.columns for t in ts): continue
        for lb in (63, 126):
            r, _ = run_backtest(rets_b, invvol_fn(ts, lb),
                                rebal_days=21, exec_lag=1)
            s = summarise(r, f"invvol {bname} lb={lb}")
            s["basket"] = bname; s["family"] = "invvol"
            s["window"] = "2015-2026"; rows.append(s)

    # 2018 window with BTC+ETH
    for bname, ts in BASKETS.items():
        if any(t not in rets_c.columns for t in ts): continue
        for lb in (63, 126):
            r, _ = run_backtest(rets_c, invvol_fn(ts, lb),
                                rebal_days=21, exec_lag=1)
            s = summarise(r, f"invvol {bname} lb={lb}")
            s["basket"] = bname; s["family"] = "invvol"
            s["window"] = "2018-2026"; rows.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_robust_universe.csv", index=False)
    print(f"Saved {len(df)} rows to letf_robust_universe.csv")

    for w in ["2011-2026", "2015-2026", "2018-2026"]:
        sub = df[df.window == w].sort_values("cagr", ascending=False)
        if len(sub) == 0: continue
        print(f"\n=== {w} ===")
        for _, r in sub.iterrows():
            print(f"  {r['label']:30s}  basket={r['basket']:14s}  "
                  f"CAGR={r['cagr']:>6.2f}%  Vol={r['vol']:>5.1f}%  "
                  f"MDD={r['mdd']:>7.2f}%  SR={r['sharpe']:>4.2f}")


if __name__ == "__main__":
    main()
