"""Phase 3 validation — pre-registered holdout of DD-throttled TSMOM.

Pre-registration (2026-04-21, branch claude/audit-nova-strategy-hypNB):
  Candidate: TSMOM K=3m tv=15% + DD-throttle-tight (-5/-10/-20)
  Pre-reg'd params: K=3m, tv=15%, rebal=21d, vol_lb=63d, exec_lag=1d,
                    tc=15bps, peak_window=252d, dd_start=-0.05,
                    dd_mid=-0.10, dd_floor=-0.20, smooth=5d
  Primary deployability target: 2yr MDD not worse than -30% in 95% of windows
  Secondary: CAGR median >= 12%, SR median >= 0.8

Method:
  1. Discovery 2011-01-01..2023-01-01: fit/compare all parameter choices
  2. Holdout  2023-01-01..today:        apply the SINGLE pre-reg'd config

  Any difference in holdout stats is honest; no re-tuning allowed.

Outputs the side-by-side table for the candidate and comparators.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import summarise
from letf_tsmom import tsmom_with_vol_target, prep as tsmom_prep
from letf_dd_throttle import apply_dd_throttle
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")
DISC_END = "2023-01-01"


def ho_stats(r, label):
    disc = summarise(r.loc[:DISC_END], label + " [disc 11-23]")
    ho   = summarise(r.loc[DISC_END:], label + " [HO 23-26]")
    full = summarise(r, label + " [full]")
    return disc, ho, full


def rolling_worst_mdd(r, win=504):
    """Worst 2-yr-window MDD in the sample."""
    r = r.dropna()
    if len(r) < win + 10:
        return float("nan"), float("nan")
    worst = 0
    pct_bad = 0
    count = 0
    for i in range(win, len(r)):
        w = r.iloc[i - win:i]
        nav = (1 + w).cumprod()
        dd = (nav / nav.cummax() - 1).min()
        if dd < worst:
            worst = dd
        if dd < -0.30:
            pct_bad += 1
        count += 1
    return worst, pct_bad / count if count else float("nan")


def main():
    tsmom_px = tsmom_prep()
    base, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.15)

    # Pre-registered candidate
    cand, m = apply_dd_throttle(
        base, peak_window=252, dd_start=-0.05, dd_mid=-0.10, dd_floor=-0.20,
        w_start=1.0, w_mid=0.5, w_floor=0.25, smooth_days=5, tc_bps=15,
    )
    # Comparators
    wide, _ = apply_dd_throttle(base)  # -10/-20/-30 defaults

    # Buy-and-hold humility
    spy = load_etf("SPY").pct_change().dropna()
    tlt = load_etf("TLT").pct_change().dropna()
    idx = spy.index.intersection(tlt.index)
    sixty40 = (0.6 * spy.loc[idx] + 0.4 * tlt.loc[idx])
    upro = load_etf("UPRO").pct_change().dropna()

    strats = {
        "TSMOM base (no overlay)": base,
        "TSMOM + DD-wide  (-10/-20/-30)": wide,
        "TSMOM + DD-tight (-5/-10/-20)  ← PRE-REG": cand,
        "SPY buy-hold":                             spy,
        "60/40 SPY/TLT":                            sixty40,
        "UPRO buy-hold (3x SPY)":                   upro,
    }

    rows = []
    for name, r in strats.items():
        try:
            disc, ho, full = ho_stats(r, name)
            ho_worst_2y, ho_pct_bad = rolling_worst_mdd(r.loc[DISC_END:])
            rows.append({
                "strategy": name,
                "disc_SR":  disc["sharpe"],  "disc_CAGR": disc["cagr"],  "disc_MDD": disc["mdd"],
                "ho_SR":    ho["sharpe"],    "ho_CAGR":   ho["cagr"],    "ho_MDD":   ho["mdd"],
                "full_SR":  full["sharpe"],  "full_CAGR": full["cagr"],  "full_MDD": full["mdd"],
                "ho_worst_2y_MDD": ho_worst_2y * 100 if np.isfinite(ho_worst_2y) else np.nan,
            })
        except Exception as e:
            rows.append({"strategy": name, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_invention_holdout.csv", index=False)
    print("=" * 110)
    print("PRE-REGISTERED HOLDOUT TEST — DD-throttled TSMOM candidate")
    print("Discovery:  2011-01-01..2023-01-01  (parameter choice)")
    print("Holdout:    2023-01-01..2026-04-21  (zero re-tuning)")
    print("=" * 110)
    cols_top = ["strategy", "disc_SR", "disc_CAGR", "disc_MDD",
                "ho_SR", "ho_CAGR", "ho_MDD", "full_SR", "full_CAGR", "full_MDD"]
    print(df[cols_top].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    # Deployability quick-read for the candidate
    print("\n--- DEPLOYABILITY (holdout-only) ---")
    cand_ho = cand.loc[DISC_END:]
    ho_dd = summarise(cand_ho, "PRE-REG candidate holdout")
    print(f"Holdout period:     {cand_ho.index[0].date()} .. {cand_ho.index[-1].date()}")
    print(f"Holdout SR:         {ho_dd['sharpe']:.2f}")
    print(f"Holdout CAGR:       {ho_dd['cagr']:.2f}%")
    print(f"Holdout vol:        {ho_dd['vol']:.2f}%")
    print(f"Holdout MDD:        {ho_dd['mdd']:.2f}%")


if __name__ == "__main__":
    main()
