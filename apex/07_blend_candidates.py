"""APEX — Build the ensemble from best candidate engines."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd

import util

OUT = Path("/home/user/bonds/data/apex")


def main():
    RDF = pd.read_csv(OUT / "big_exp_returns.csv", parse_dates=[0], index_col=0)

    # Candidates: must have IS Sharpe > 0.45, OOS Sharpe > 0.4, MDD > -60%
    with open(OUT / "big_exp_metrics.json") as f:
        meta = json.load(f)

    candidates = []
    for name, m in meta.items():
        if m["is"].get("sharpe", 0) >= 0.45 and m["oos"].get("sharpe", 0) >= 0.4 \
           and m["full"].get("mdd", -1) > -0.65:
            candidates.append(name)
    print(f"Qualifying candidates ({len(candidates)}):")
    for c in candidates:
        m = meta[c]
        print(f"  {c:25s} IS={m['is'].get('sharpe',0):.2f} OOS={m['oos'].get('sharpe',0):.2f} MDD={m['full'].get('mdd',0)*100:.1f}%")

    # Correlation matrix over IS
    C = RDF[candidates].loc[:"2018-12-31"].corr()
    print("\nIS correlations:")
    print(C.round(2))

    # Greedy selection: pick top-SR, then add the one with lowest max corr to already-selected
    selected = []
    ranked_by_is = sorted(candidates, key=lambda n: -meta[n]["is"].get("sharpe", 0))
    selected.append(ranked_by_is[0])
    max_n = 7
    while len(selected) < max_n:
        best_next = None
        best_score = -np.inf
        for c in ranked_by_is:
            if c in selected:
                continue
            # score = IS_sharpe - lambda * avg_correlation_to_selected
            max_corr = max(abs(C.loc[c, s]) for s in selected)
            score = meta[c]["is"].get("sharpe", 0) - 1.0 * max_corr
            if score > best_score:
                best_score = score
                best_next = c
        if best_next is None:
            break
        selected.append(best_next)

    print(f"\nGreedy-selected {len(selected)} engines:")
    for s in selected:
        m = meta[s]
        print(f"  {s:25s} IS={m['is'].get('sharpe',0):.2f} OOS={m['oos'].get('sharpe',0):.2f}")

    R_sel = RDF[selected]
    print("\nSelected correlation matrix (IS):")
    print(R_sel.loc[:"2018-12-31"].corr().round(2))

    # --- Blend: inverse-variance weights fit on IS ---
    is_R = R_sel.loc[:"2018-12-31"].dropna(how="any")
    iv = 1.0 / (is_R.std() * np.sqrt(util.DPY)).replace(0, np.nan)
    iv = iv / iv.sum()
    print(f"\nIS blend weights:")
    print(iv.round(3))

    blend = (R_sel * iv).sum(axis=1)

    # --- Apply DD throttle + vol target ---
    def finalize(r, target_vol=0.15, dd_floor=-0.12):
        c = (1 + r).cumprod()
        hwm = c.rolling(252, min_periods=30).max()
        dd = c / hwm - 1
        m = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)
        r2 = r * m
        rv = r2.rolling(60).std() * np.sqrt(util.DPY)
        vm = (target_vol / rv).clip(lower=0.25, upper=1.5).shift(1).fillna(1.0)
        return r2 * vm

    print("\n=== Blend summary ===")
    for tv in (0.10, 0.15, 0.20, 0.25):
        rf = finalize(blend, target_vol=tv)
        util.summarize(rf, f"tv={tv}")

    # Show full window breakdowns at best target
    print("\n=== Final breakdowns at target_vol=0.20 ===")
    rf = finalize(blend, 0.20)
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC", ("2007-01-01", "2009-12-31")),
                        ("COVID", ("2020-01-01", "2020-12-31")),
                        ("2022RH", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(rf, s, e), f"  {lbl}")


if __name__ == "__main__":
    main()
