"""Step 4 — inverse-vol / risk-parity sweep on LETFs.

At each rebalance, compute trailing `lookback` vol per ticker and allocate
proportional to 1/vol, normalised to sum=1. Also tests a scaled variant
that targets a naive portfolio vol (ignoring correlation).

Baskets tested:
  * core6  UPRO/TQQQ/SOXL/TECL/TMF/UGL
  * big8   + FAS, EDC
  * all17  every long LETF
  * noexp5 UPRO/TQQQ/TMF/UGL/UCO (diversified across asset classes)

Lookbacks: 21, 63, 126, 252 days. Rebals: 3, 5, 10, 21.

Also tests SCALED inv-vol — multiplies the inv-vol weights by a factor so
the book targets a fixed naive vol (sum of inv × vol × scalar = target).
This adds static leverage at the portfolio level (periodically set only at
rebalance, not daily) — testing vol targets of 25/40/60/80%.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (load_universe, common_window_returns,
                         run_backtest, summarise)
from letf_universe import LETF_LONG_2011


BASKETS = {
    "core6":  ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"],
    "big8":   ["UPRO","TQQQ","SOXL","TECL","TMF","UGL","FAS","EDC"],
    "all17":  LETF_LONG_2011,
    "mix5":   ["UPRO","TQQQ","TMF","UGL","UCO"],
    "clean4": ["UPRO","TQQQ","TMF","UGL"],
}
LOOKBACKS = [21, 63, 126, 252]
REBALS = [3, 5, 10, 21]
START = "2011-01-01"
OUT = Path("/home/user/bonds/data/results")


def invvol_fn(tickers, lookback):
    def fn(d, hist):
        if len(hist) < lookback + 5:
            return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0:
            return None
        vol = r.std()
        inv = 1 / vol.replace(0, np.nan)
        inv = inv.fillna(0)
        w = inv / inv.sum()
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


def invvol_scaled_fn(tickers, lookback, target_vol):
    """Inv-vol weights scaled so naive-indep portfolio vol = target_vol.
    Naive portfolio vol ≈ sqrt(Σ (w_i σ_i)²) but inv-vol equalises w_i σ_i,
    so naive vol ≈ σ̄ * sqrt(N)/(N) = σ̄/√N … actually for inv-vol,
    w_i = (1/σ_i) / Σ(1/σ_j), so (w_i σ_i) = 1 / Σ(1/σ_j) is constant ≡ c,
    naive indep vol = c*√N. We want total leverage k * c * √N = target_vol.
    So k = target_vol / (c * √N).  Capped at 5x."""
    def fn(d, hist):
        if len(hist) < lookback + 5:
            return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0:
            return None
        sig = r.std() * np.sqrt(252)   # annualised
        if (sig <= 0).all():
            return None
        inv = 1 / sig.replace(0, np.nan).fillna(0)
        S = inv.sum()
        if S <= 0:
            return None
        w = inv / S                   # sums to 1
        c = 1.0 / S                   # constant (w_i * sig_i = c) in ann vol units
        n = r.shape[1]
        naive_vol = c * np.sqrt(n)
        k = min(target_vol / naive_vol, 5.0) if naive_vol > 0 else 1.0
        w = w * k
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


def main():
    px = load_universe(LETF_LONG_2011, start=START).dropna(how="any")
    rets = common_window_returns(px)
    print(f"Window: {rets.index[0].date()} .. {rets.index[-1].date()} "
          f"({len(rets)} days)")

    rows = []
    # Part A — plain inv-vol
    for bname, ts in BASKETS.items():
        for lb in LOOKBACKS:
            for nd in REBALS:
                r, _ = run_backtest(rets, invvol_fn(ts, lb),
                                    rebal_days=nd, exec_lag=1)
                s = summarise(r, f"invvol {bname} lb={lb} @ {nd}d")
                s["family"] = "invvol"
                rows.append(s)

    # Part B — vol-scaled inv-vol (period-only scaling, not daily)
    for bname, ts in BASKETS.items():
        for lb in LOOKBACKS:
            for tv in [0.25, 0.40, 0.60, 0.80]:
                for nd in REBALS:
                    r, _ = run_backtest(rets,
                                        invvol_scaled_fn(ts, lb, tv),
                                        rebal_days=nd, exec_lag=1)
                    s = summarise(r, f"invvol-s {bname} lb={lb} tv={int(tv*100)}% @ {nd}d")
                    s["family"] = "invvol-scaled"
                    rows.append(s)

    df = pd.DataFrame(rows).sort_values("cagr", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "letf_sweep_invvol.csv", index=False)
    print(f"\nSaved {len(df)} rows to letf_sweep_invvol.csv")

    print("\nTop 15 by CAGR:")
    for _, r in df.head(15).iterrows():
        print(f"  {r['label']:48s}  CAGR={r['cagr']:>6.2f}%  "
              f"Vol={r['vol']:>5.1f}%  MDD={r['mdd']:>7.2f}%  "
              f"SR={r['sharpe']:>4.2f}  C/MDD={r['cagr_mdd']:>4.2f}")

    print("\nTop 15 by Sharpe:")
    for _, r in df.sort_values("sharpe", ascending=False).head(15).iterrows():
        print(f"  {r['label']:48s}  SR={r['sharpe']:>4.2f}  "
              f"CAGR={r['cagr']:>6.2f}%  MDD={r['mdd']:>7.2f}%")

    print("\nTop 15 by CAGR/|MDD|:")
    for _, r in df.sort_values("cagr_mdd", ascending=False).head(15).iterrows():
        print(f"  {r['label']:48s}  C/MDD={r['cagr_mdd']:>4.2f}  "
              f"CAGR={r['cagr']:>6.2f}%  MDD={r['mdd']:>7.2f}%  "
              f"SR={r['sharpe']:>4.2f}")


if __name__ == "__main__":
    main()
