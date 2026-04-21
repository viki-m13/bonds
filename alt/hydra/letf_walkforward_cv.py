"""Priority 8 — walk-forward k-fold CV with stability-aware selection.

For the final recommended strategy to be defensible, we want:
  - k non-overlapping folds (4 folds of ~4 years each, 2011-2026)
  - Per-fold Sharpe AND per-fold CAGR
  - Stability score = mean(SR) - k*std(SR)  for k in {0, 0.5, 1}
  - Pick the config that maximises stability score, not mean

Strategy candidates (pre-registered literature + survivors):
  - invvol clean4 lb=21 (survivor)
  - invvol core6 lb=63 (survivor)
  - TSMOM K=3m tv=20%  (Moreira-Muir + Moskowitz params)
  - TSMOM K=6m plain
  - TSMOM K=12m tv=20% (published Moskowitz canonical)
  - Vol-managed invvol core6 lb=63 VM vw=126d (stacked)
  - SPY/TLT 60/40 (the embarrassingly good benchmark)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise,
                         w_fixed)
from letf_crypto_universe import load_with_crypto
from hydra_core import load_etf
from letf_tsmom import tsmom_backtest, tsmom_with_vol_target, prep as tsmom_prep
from letf_volmanaged import vol_managed_backtest


OUT = Path("/home/user/bonds/data/results")

FOLDS = [
    ("F1 2011-2014", "2011-01-01", "2015-01-01"),
    ("F2 2015-2018", "2015-01-01", "2019-01-01"),
    ("F3 2019-2022", "2019-01-01", "2023-01-01"),
    ("F4 2023-2026", "2023-01-01", "2026-12-31"),
]


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


def slice_rets(rets, start, end):
    mask = (rets.index >= pd.Timestamp(start)) & (rets.index < pd.Timestamp(end))
    return rets.loc[mask]


def candidate_returns(tsmom_px, letf_rets):
    """Return dict[name] -> pd.Series (daily port return) over full window."""
    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]

    out = {}
    r, _ = run_backtest(letf_rets, invvol_fn(clean4, 21), rebal_days=21)
    out["invvol clean4 lb=21"] = r
    r, _ = run_backtest(letf_rets, invvol_fn(core6, 63), rebal_days=21)
    out["invvol core6 lb=63"] = r

    # TSMOM variants
    r, _ = tsmom_backtest(tsmom_px, K_months=3)
    out["TSMOM K=3m plain"] = r
    r, _ = tsmom_backtest(tsmom_px, K_months=6)
    out["TSMOM K=6m plain"] = r
    r, _ = tsmom_backtest(tsmom_px, K_months=12)
    out["TSMOM K=12m plain"] = r
    r, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.20)
    out["TSMOM K=3m tv=20%"] = r
    r, _ = tsmom_with_vol_target(tsmom_px, K_months=12, target_vol=0.20)
    out["TSMOM K=12m tv=20%"] = r

    # Vol-managed stacked
    base, _ = run_backtest(letf_rets, invvol_fn(core6, 63), rebal_days=21)
    vm, _ = vol_managed_backtest(base, vol_window=126)
    out["invvol core6 lb=63 + VM"] = vm

    # Benchmark
    spy = load_etf("SPY").pct_change().fillna(0)
    tlt = load_etf("TLT").pct_change().fillna(0)
    ref = pd.concat([spy, tlt], axis=1).dropna()
    ref.columns = ["SPY","TLT"]
    ref = ref.loc[letf_rets.index[0]:letf_rets.index[-1]]
    out["SPY/TLT 60/40 (1x)"] = 0.6 * ref["SPY"] + 0.4 * ref["TLT"]

    # Simple UPRO buy-hold (leverage benchmark)
    upro = load_etf("UPRO").pct_change().fillna(0)
    upro = upro.loc[letf_rets.index[0]:letf_rets.index[-1]]
    out["UPRO buy-hold"] = upro

    return out


def fold_metrics(r, start, end):
    r = r.loc[(r.index >= pd.Timestamp(start)) & (r.index < pd.Timestamp(end))]
    s = summarise(r, "")
    return {"cagr": s["cagr"], "sharpe": s["sharpe"], "mdd": s["mdd"],
            "vol": s["vol"]}


def main():
    tsmom_px = tsmom_prep()
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)
    print(f"LETF window: {rets.index[0].date()}..{rets.index[-1].date()}")
    print(f"TSMOM window: {tsmom_px.index[0].date()}..{tsmom_px.index[-1].date()}")

    cands = candidate_returns(tsmom_px, rets)
    print(f"\nCandidates: {len(cands)}")

    rows = []
    for name, r in cands.items():
        row = {"label": name}
        srs = []
        for fname, fs, fe in FOLDS:
            m = fold_metrics(r, fs, fe)
            row[f"{fname} SR"] = m["sharpe"]
            row[f"{fname} CAGR"] = m["cagr"]
            row[f"{fname} MDD"] = m["mdd"]
            srs.append(m["sharpe"])
        arr = np.array(srs)
        row["mean_SR"] = arr.mean()
        row["std_SR"] = arr.std(ddof=1)
        row["min_SR"] = arr.min()
        row["score_meanMinus1std"] = arr.mean() - arr.std(ddof=1)
        row["score_meanMinusHalfStd"] = arr.mean() - 0.5 * arr.std(ddof=1)
        row["worst_fold"] = FOLDS[int(arr.argmin())][0]
        # Full period
        full = summarise(r, "")
        row["full_SR"] = full["sharpe"]
        row["full_CAGR"] = full["cagr"]
        row["full_MDD"] = full["mdd"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_walkforward_cv.csv", index=False)

    # Pretty print
    print("\n=== Per-fold Sharpe ===")
    cols = [f"{f[0]} SR" for f in FOLDS]
    print(df[["label"] + cols + ["mean_SR","std_SR","min_SR",
                                  "score_meanMinus1std"]]
          .round(2).to_string(index=False))

    print("\n=== Per-fold CAGR ===")
    cols = [f"{f[0]} CAGR" for f in FOLDS]
    print(df[["label"] + cols + ["full_CAGR"]].round(2).to_string(index=False))

    print("\n=== Per-fold MDD ===")
    cols = [f"{f[0]} MDD" for f in FOLDS]
    print(df[["label"] + cols + ["full_MDD"]].round(1).to_string(index=False))

    print("\n=== Ranked by stability score (mean − 1σ of per-fold SR) ===")
    srt = df.sort_values("score_meanMinus1std", ascending=False)
    print(srt[["label","mean_SR","std_SR","min_SR","worst_fold",
               "score_meanMinus1std","score_meanMinusHalfStd"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
