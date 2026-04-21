"""Priority 9 — pre-registered final holdout.

Protocol:
  1. Select the strategy using ONLY pre-2023 data (discovery window).
     The stability-score winner on that window = "picked-blind" candidate.
  2. Report its OOS metrics on 2023-01-01 .. 2026-04-02 WITHOUT ever
     having tuned on that window.
  3. Compare against two baselines selected by the same rule:
       - Best-Sharpe-on-discovery: winner by mean Sharpe on discovery only
       - Benchmarks: SPY BH, SPY/TLT 60/40

This is the closest we can get to a true OOS test given we've already
looked at 2023-26 data.  To increase honesty we compute the discovery
window Sharpe/MDD SEPARATELY from the holdout — no cross-contamination.
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

DISCOVERY_END = "2023-01-01"
HOLDOUT_END   = "2026-04-02"

DISCOVERY_FOLDS = [
    ("D1 2011-2014", "2011-01-01", "2015-01-01"),
    ("D2 2015-2018", "2015-01-01", "2019-01-01"),
    ("D3 2019-2022", "2019-01-01", "2023-01-01"),
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


def fold_sr(r, start, end):
    sl = r.loc[(r.index >= pd.Timestamp(start)) & (r.index < pd.Timestamp(end))]
    return summarise(sl, "")["sharpe"]


def main():
    tsmom_px = tsmom_prep()
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]

    # Build candidate daily return series
    cands = {}
    r, _ = run_backtest(rets, invvol_fn(clean4, 21), rebal_days=21)
    cands["invvol clean4 lb=21"] = r
    r, _ = run_backtest(rets, invvol_fn(core6, 63), rebal_days=21)
    cands["invvol core6 lb=63"] = r

    # Vol-managed stacked
    vm, _ = vol_managed_backtest(cands["invvol core6 lb=63"], vol_window=126)
    cands["invvol core6 lb=63 + VM vw=126d"] = vm

    r, _ = tsmom_backtest(tsmom_px, K_months=3)
    cands["TSMOM K=3m plain"] = r
    r, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.20)
    cands["TSMOM K=3m tv=20%"] = r
    r, _ = tsmom_with_vol_target(tsmom_px, K_months=12, target_vol=0.20)
    cands["TSMOM K=12m tv=20%"] = r

    # Score on DISCOVERY only
    disc_scores = {}
    for name, r in cands.items():
        srs = []
        for fn, fs, fe in DISCOVERY_FOLDS:
            srs.append(fold_sr(r, fs, fe))
        arr = np.array(srs)
        disc_scores[name] = {
            "mean_SR_disc": arr.mean(),
            "std_SR_disc": arr.std(ddof=1),
            "min_SR_disc": arr.min(),
            "stability_disc": arr.mean() - arr.std(ddof=1),
            "per_fold": srs,
        }

    print("=== Discovery-window (2011-2022) per-fold Sharpe ===")
    for name, sc in disc_scores.items():
        print(f"  {name:<42s} folds={[round(x,2) for x in sc['per_fold']]}  "
              f"mean={sc['mean_SR_disc']:.2f}  σ={sc['std_SR_disc']:.2f}  "
              f"min={sc['min_SR_disc']:.2f}  stab={sc['stability_disc']:.2f}")

    # Pick winner by stability_disc
    winner = max(disc_scores.items(), key=lambda kv: kv[1]["stability_disc"])
    best_mean = max(disc_scores.items(), key=lambda kv: kv[1]["mean_SR_disc"])
    print(f"\nStability winner on discovery (pre-registered): {winner[0]}")
    print(f"Best-Sharpe winner on discovery:                 {best_mean[0]}")

    # Now evaluate on HOLDOUT
    print(f"\n=== Holdout {DISCOVERY_END} .. {HOLDOUT_END} (21.0 months) ===")

    def holdout_stats(r, label):
        sl = r.loc[(r.index >= pd.Timestamp(DISCOVERY_END)) &
                   (r.index < pd.Timestamp(HOLDOUT_END))]
        s = summarise(sl, label)
        return s

    # Benchmarks
    spy = load_etf("SPY").pct_change().fillna(0)
    tlt = load_etf("TLT").pct_change().fillna(0)
    ref = pd.concat([spy, tlt], axis=1).dropna()
    ref.columns = ["SPY","TLT"]
    ref = ref.loc["2011-01-01":HOLDOUT_END]
    cands["SPY/TLT 60/40 (1x)"] = 0.6*ref["SPY"] + 0.4*ref["TLT"]
    cands["SPY BH"] = ref["SPY"]
    upro = load_etf("UPRO").pct_change().fillna(0).loc["2011-01-01":HOLDOUT_END]
    cands["UPRO buy-hold"] = upro

    # EW-all17 baseline
    from letf_universe import LETF_LONG_2011
    ew_all17 = {t: 1/len(LETF_LONG_2011) for t in LETF_LONG_2011}
    r, _ = run_backtest(rets, w_fixed(ew_all17), rebal_days=21)
    cands["EW-all17"] = r

    rows = []
    for name, r in cands.items():
        disc_full = summarise(r.loc["2011-01-01":DISCOVERY_END], "")
        ho = summarise(r.loc[DISCOVERY_END:HOLDOUT_END], "")
        rows.append({
            "label": name,
            "disc_SR":  disc_full["sharpe"], "disc_CAGR": disc_full["cagr"],
            "disc_MDD": disc_full["mdd"],
            "ho_SR":    ho["sharpe"],        "ho_CAGR":   ho["cagr"],
            "ho_MDD":   ho["mdd"],           "ho_vol":    ho["vol"],
            "delta_SR": ho["sharpe"] - disc_full["sharpe"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_final_holdout.csv", index=False)

    print(f"\n{'Strategy':<42s} {'disc SR':>8s} {'disc CAGR':>10s} "
          f"{'ho SR':>7s} {'ho CAGR':>9s} {'ho MDD':>8s} {'ΔSR':>7s}")
    print("-"*100)
    for r in rows:
        tag = ""
        if r["label"] == winner[0]: tag = " ***STABILITY PICK***"
        elif r["label"] == best_mean[0] and best_mean[0] != winner[0]:
            tag = " (mean-SR pick)"
        print(f"{r['label']:<42s} {r['disc_SR']:>8.2f} {r['disc_CAGR']:>9.2f}% "
              f"{r['ho_SR']:>7.2f} {r['ho_CAGR']:>8.2f}% "
              f"{r['ho_MDD']:>7.2f}% {r['delta_SR']:>+7.2f}{tag}")


if __name__ == "__main__":
    main()
