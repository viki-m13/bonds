"""Step 10 — ensemble of the 3 survivors.

Equal-weight combination of:
  A. invvol core6 lb=63 + VM vw=126d  (stability pick)
  B. invvol clean4 lb=21              (mean-SR pick)
  C. TSMOM K=3m tv=20%                (pre-reg literature pick)

Two reasons:
  1. Each relies on a different alpha source:
       A: cross-sectional risk-parity + portfolio vol-targeting
       B: cross-sectional risk-parity on smaller basket
       C: time-series momentum + vol-targeting
     Different signals => partial diversification.
  2. Model averaging reduces parameter-choice risk ("am I sure lb=63 not 126?")

We run all three and combine daily-return series (no rebal logic across
them — they're separate sleeves 1/3 each).  Then compare holdout + full.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise,
                         w_fixed)
from letf_crypto_universe import load_with_crypto
from letf_tsmom import tsmom_with_vol_target, prep as tsmom_prep
from letf_volmanaged import vol_managed_backtest


OUT = Path("/home/user/bonds/data/results")
DISC_END = "2023-01-01"
HO_END   = "2026-04-02"


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


def stats(r, label):
    full = summarise(r, label)
    ho = summarise(r.loc[DISC_END:HO_END], label + " [holdout]")
    dc = summarise(r.loc["2011-01-01":DISC_END], label + " [disc]")
    return full, dc, ho


def main():
    tsmom_px = tsmom_prep()
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]

    # A
    base_a, _ = run_backtest(rets, invvol_fn(core6, 63), rebal_days=21)
    a, _ = vol_managed_backtest(base_a, vol_window=126)
    # B
    b, _ = run_backtest(rets, invvol_fn(clean4, 21), rebal_days=21)
    # C
    c, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.20)

    a = a.reindex(rets.index).fillna(0)
    b = b.reindex(rets.index).fillna(0)
    c = c.reindex(rets.index).fillna(0)

    # Pairwise correlations to justify ensemble
    df = pd.DataFrame({"A": a, "B": b, "C": c})
    print("Pairwise correlation of daily returns:")
    print(df.corr().round(3).to_string())

    ens = (a + b + c) / 3
    # Also AB, AC, BC
    rows = []
    series = {
        "A  invvol core6 lb=63 + VM vw=126d": a,
        "B  invvol clean4 lb=21": b,
        "C  TSMOM K=3m tv=20%": c,
        "A+B (1/2 each)": (a+b)/2,
        "A+C (1/2 each)": (a+c)/2,
        "B+C (1/2 each)": (b+c)/2,
        "ENSEMBLE A+B+C (1/3 each)": ens,
    }
    for name, r in series.items():
        full, dc, ho = stats(r, name)
        rows.append({
            "label": name,
            "full_SR": full["sharpe"], "full_CAGR": full["cagr"],
            "full_MDD": full["mdd"], "full_vol": full["vol"],
            "disc_SR": dc["sharpe"],  "disc_CAGR": dc["cagr"],
            "disc_MDD": dc["mdd"],
            "ho_SR":   ho["sharpe"],   "ho_CAGR": ho["cagr"],
            "ho_MDD":  ho["mdd"],      "ho_vol": ho["vol"],
        })

    dfo = pd.DataFrame(rows)
    dfo.to_csv(OUT / "letf_ensemble.csv", index=False)

    print(f"\n{'Strategy':<40s} {'Full SR':>7s} {'Full CAGR':>10s} "
          f"{'Full MDD':>9s} {'HO SR':>7s} {'HO CAGR':>9s} {'HO MDD':>8s}")
    print("-"*100)
    for r in rows:
        print(f"{r['label']:<40s} {r['full_SR']:>7.2f} {r['full_CAGR']:>9.2f}% "
              f"{r['full_MDD']:>8.2f}% {r['ho_SR']:>7.2f} "
              f"{r['ho_CAGR']:>8.2f}% {r['ho_MDD']:>7.2f}%")


if __name__ == "__main__":
    main()
