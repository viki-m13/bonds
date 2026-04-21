"""Step 5 — momentum sweep.

At each rebalance:
  * Pick top-N tickers by trailing lookback cumulative return
  * Equal-weight the picks
  * Optionally apply a "trend gate": only include a pick if its lookback
    return is also positive (otherwise allocate to BIL cash)

Sweep:
  universe : core6, big8, all17
  lookback : 21, 63, 126, 252
  top_N    : 1, 2, 3, 4, 5
  rebal    : 3, 5, 10, 21
  trend_gate : off, on (require lookback return > 0)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (load_universe, common_window_returns,
                         run_backtest, summarise)
from letf_universe import LETF_LONG_2011


UNIVERSES = {
    "core6": ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"],
    "big8":  ["UPRO","TQQQ","SOXL","TECL","TMF","UGL","FAS","EDC"],
    "all17": LETF_LONG_2011,
}
LOOKBACKS = [21, 63, 126, 252]
TOP_NS = [1, 2, 3, 4, 5]
REBALS = [3, 5, 10, 21]
START = "2011-01-01"
OUT = Path("/home/user/bonds/data/results")


def momentum_fn(tickers, lookback, top_n, trend_gate=False):
    def fn(d, hist):
        if len(hist) < lookback + 5:
            return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0:
            return None
        cum = (1 + r).prod() - 1
        picks = cum.sort_values(ascending=False).head(top_n)
        if trend_gate:
            picks = picks[picks > 0]
        out = pd.Series(0.0, index=hist.columns)
        if len(picks) > 0:
            share = 1.0 / len(picks)
            out.loc[picks.index] = share
            # If trend_gate removed some picks, put the rest in BIL if avail
            if trend_gate and len(picks) < top_n and "BIL" in hist.columns:
                out["BIL"] = (top_n - len(picks)) * share
        return out
    return fn


def main():
    # Include BIL so trend gate can park cash there
    tickers = sorted(set(sum([U for U in UNIVERSES.values()], [])) | {"BIL"})
    px = load_universe(tickers, start=START).dropna(how="any")
    rets = common_window_returns(px)
    print(f"Window: {rets.index[0].date()} .. {rets.index[-1].date()} "
          f"({len(rets)} days)  cols: {list(rets.columns)}")

    rows = []
    for uname, ts in UNIVERSES.items():
        for lb in LOOKBACKS:
            for n in TOP_NS:
                if n > len(ts):
                    continue
                for nd in REBALS:
                    for tg in (False, True):
                        r, _ = run_backtest(rets,
                                            momentum_fn(ts, lb, n, tg),
                                            rebal_days=nd, exec_lag=1)
                        tag = "gate" if tg else "raw "
                        s = summarise(r, f"mom[{tag}] {uname} lb={lb} top{n} @ {nd}d")
                        s["universe"] = uname
                        s["lookback"] = lb
                        s["top_n"] = n
                        s["rebal"] = nd
                        s["gate"] = tg
                        rows.append(s)

    df = pd.DataFrame(rows).sort_values("cagr", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "letf_sweep_momentum.csv", index=False)
    print(f"\nSaved {len(df)} rows to letf_sweep_momentum.csv")

    print("\nTop 15 by CAGR:")
    for _, r in df.head(15).iterrows():
        print(f"  {r['label']:50s}  CAGR={r['cagr']:>6.2f}%  "
              f"Vol={r['vol']:>5.1f}%  MDD={r['mdd']:>7.2f}%  "
              f"SR={r['sharpe']:>4.2f}  C/MDD={r['cagr_mdd']:>4.2f}")

    print("\nTop 15 by Sharpe:")
    for _, r in df.sort_values("sharpe", ascending=False).head(15).iterrows():
        print(f"  {r['label']:50s}  SR={r['sharpe']:>4.2f}  "
              f"CAGR={r['cagr']:>6.2f}%  MDD={r['mdd']:>7.2f}%")

    print("\nTop 15 by CAGR/|MDD|:")
    for _, r in df.sort_values("cagr_mdd", ascending=False).head(15).iterrows():
        print(f"  {r['label']:50s}  C/MDD={r['cagr_mdd']:>4.2f}  "
              f"CAGR={r['cagr']:>6.2f}%  MDD={r['mdd']:>7.2f}%  "
              f"SR={r['sharpe']:>4.2f}")


if __name__ == "__main__":
    main()
