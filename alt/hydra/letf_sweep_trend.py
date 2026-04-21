"""Step 6 — 200-day trend overlay variants.

For each selected strategy, apply a trend gate on each LETF's underlying
(SPY/QQQ/TLT/GLD/SMH/XLK/XLF/XBI/FXI/...).
If underlying > SMA(trend_lookback), hold target weight; else park weight
in BIL.

Trend lookbacks tested: 50, 100, 150, 200, 250.
Rebals: 3, 5, 10, 21.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (load_universe, common_window_returns,
                         run_backtest, summarise)
from letf_universe import LETF_CATALOG, LETF_LONG_2011

TREND_LOOKBACKS = [50, 100, 150, 200, 250]
REBALS = [3, 5, 10, 21]
START = "2011-01-01"
OUT = Path("/home/user/bonds/data/results")


def all_underlyings(tickers):
    return sorted(set(LETF_CATALOG[t]["under"] for t in tickers
                       if t in LETF_CATALOG and LETF_CATALOG[t].get("under")))


def build_sma_signals(underlyings, lookback, dates):
    """Return DataFrame[date x underlying] = 1 if close > SMA(lookback) lagged."""
    from hydra_core import load_etf
    px = pd.DataFrame({u: load_etf(u) for u in underlyings}).sort_index()
    sma = px.rolling(lookback).mean()
    sig = (px > sma).astype(float).shift(1)   # T-1 close signal, act at T
    return sig.reindex(dates).fillna(0.0)


def trend_gated_fn(target_weights, trend_lb, dates):
    """Returns a weights_fn that at each rebal:
         for each ticker t with target wt w_t:
           if underlying signal > 0.5: keep w_t
           else: move w_t to BIL
    """
    tickers = list(target_weights.keys())
    unders = [LETF_CATALOG[t]["under"] for t in tickers]
    sig = build_sma_signals(sorted(set(unders)), trend_lb, dates)

    def fn(d, hist):
        out = pd.Series(0.0, index=hist.columns)
        if d not in sig.index:
            return None
        s = sig.loc[d]
        bil = 0.0
        for t, wt in target_weights.items():
            u = LETF_CATALOG[t]["under"]
            if s.get(u, 0) > 0.5:
                out[t] = out.get(t, 0) + wt
            else:
                bil += wt
        if "BIL" in out.index:
            out["BIL"] = out.get("BIL", 0) + bil
        return out
    return fn


# Candidates we want to overlay (small, picked from static+invvol results)
CANDIDATES = {
    "100% TQQQ":           {"TQQQ": 1.0},
    "100% UPRO":           {"UPRO": 1.0},
    "100% SOXL":           {"SOXL": 1.0},
    "100% TECL":           {"TECL": 1.0},
    "HFEA-Tech 80/20 TQQQ/TMF": {"TQQQ": 0.80, "TMF": 0.20},
    "HFEA-Tech 60/40 TQQQ/TMF": {"TQQQ": 0.60, "TMF": 0.40},
    "3sleeve 70/10/20 TQQQ/TMF/UGL": {"TQQQ": 0.70, "TMF": 0.10, "UGL": 0.20},
    "3sleeve 50/20/30 TQQQ/TMF/UGL": {"TQQQ": 0.50, "TMF": 0.20, "UGL": 0.30},
    "3sleeve 80/10/10 TQQQ/TMF/UGL": {"TQQQ": 0.80, "TMF": 0.10, "UGL": 0.10},
    "EW5 UPRO/TQQQ/SOXL/TMF/UGL": {"UPRO": 0.20, "TQQQ": 0.20, "SOXL": 0.20,
                                    "TMF": 0.20, "UGL": 0.20},
    "EW6 UPRO/TQQQ/SOXL/TECL/TMF/UGL": {"UPRO": 1/6, "TQQQ": 1/6, "SOXL": 1/6,
                                         "TECL": 1/6, "TMF": 1/6, "UGL": 1/6},
    "theme4 SSO/TQQQ/UBT/UGL 25/25/25/25":
        {"SSO": 0.25, "TQQQ": 0.25, "UBT": 0.25, "UGL": 0.25},
    "theme4 UPRO/TQQQ/TMF/UGL 25/25/25/25":
        {"UPRO": 0.25, "TQQQ": 0.25, "TMF": 0.25, "UGL": 0.25},
}


def main():
    tickers = sorted(set(sum([list(w.keys()) for w in CANDIDATES.values()], [])) | {"BIL"})
    px = load_universe(tickers, start=START).dropna(how="any")
    rets = common_window_returns(px)
    print(f"Universe: {list(rets.columns)}")
    print(f"Window:   {rets.index[0].date()} .. {rets.index[-1].date()} "
          f"({len(rets)} days)")

    rows = []
    for name, w in CANDIDATES.items():
        for tlb in TREND_LOOKBACKS:
            for nd in REBALS:
                fn = trend_gated_fn(w, tlb, rets.index)
                r, _ = run_backtest(rets, fn, rebal_days=nd, exec_lag=1)
                s = summarise(r, f"{name} + trend{tlb}d @ {nd}d")
                rows.append(s)

    df = pd.DataFrame(rows).sort_values("cagr", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "letf_sweep_trend.csv", index=False)
    print(f"\nSaved {len(df)} rows to letf_sweep_trend.csv")

    print("\nTop 15 by CAGR:")
    for _, r in df.head(15).iterrows():
        print(f"  {r['label']:58s}  CAGR={r['cagr']:>6.2f}%  "
              f"Vol={r['vol']:>5.1f}%  MDD={r['mdd']:>7.2f}%  "
              f"SR={r['sharpe']:>4.2f}  C/MDD={r['cagr_mdd']:>4.2f}")

    print("\nTop 15 by CAGR/|MDD|:")
    for _, r in df.sort_values("cagr_mdd", ascending=False).head(15).iterrows():
        print(f"  {r['label']:58s}  C/MDD={r['cagr_mdd']:>4.2f}  "
              f"CAGR={r['cagr']:>6.2f}%  MDD={r['mdd']:>7.2f}%  "
              f"SR={r['sharpe']:>4.2f}")

    print("\nTop 15 by Sharpe:")
    for _, r in df.sort_values("sharpe", ascending=False).head(15).iterrows():
        print(f"  {r['label']:58s}  SR={r['sharpe']:>4.2f}  "
              f"CAGR={r['cagr']:>6.2f}%  MDD={r['mdd']:>7.2f}%")


if __name__ == "__main__":
    main()
