"""APEX v27 — try swapping in the novel KALMAN and ACCEL_MOM sleeves."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
from itertools import combinations
import numpy as np
import pandas as pd
import util
import sleeves_phoenix_exact as PX
import sleeves_v12 as SV12
import sleeves_v15 as SV15
import sleeves_v16 as SV16
import sleeves_v17 as SV17
import sleeves_v18 as SV18
import sleeves_v19 as SV19
import sleeves_v20 as SV20
import sleeves_v26 as SV26
from apex_v13 import extend_cp
from apex_v17 import run_v17

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


POOL = {
    "PX_HELIOS":        lambda cp: PX.sleeve_helios_exact(cp),
    "PX_ORION":         lambda cp: PX.sleeve_orion_exact(cp),
    "BREAKOUT52":       lambda cp: SV18.sleeve_breakout52(cp),
    "VOLZ_MOM":         lambda cp: SV19.sleeve_volz_mom(cp),
    "HMM_REGIME":       lambda cp: SV18.sleeve_hmm(cp),
    "DIVERGENCE":       lambda cp: SV18.sleeve_divergence(cp),
    "CROSS_DECORR":     lambda cp: SV16.sleeve_cross_decorr(cp),
    "PCA":              lambda cp: SV17.sleeve_pca(cp),
    "DUALBEAR":         lambda cp: SV12.sleeve_dualbear_defense(cp),
    "KALMAN":           lambda cp: SV26.sleeve_kalman(cp),
    "ACCEL_MOM":        lambda cp: SV26.sleeve_accel_mom(cp),
    "HURST":            lambda cp: SV26.sleeve_hurst(cp),
}


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    print("Pre-building all candidate sleeves...")
    full_sw = {}
    for name, fn in POOL.items():
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        full_sw[name] = W.mul(m, axis=0)

    # Exhaustive search 3-5 combos with 12 sleeves including new ones
    all_results = []
    for k in [3, 4, 5]:
        combos = list(combinations(POOL.keys(), k))
        print(f"Testing {len(combos)} combos of size {k}...")
        for combo in combos:
            sw = {name: full_sw[name] for name in combo}
            # Try multiple crypto weights
            for cw in [0.50, 0.55, 0.60]:
                net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=0.18)
                m = util.metrics(net)
                om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
                m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
                mis = util.metrics(util.regime_slice(net, "2010-03-11", IS_END))
                all_results.append({
                    "combo": combo, "k": k, "cw": cw,
                    "full_sr": m["sharpe"], "full_cagr": m["cagr"],
                    "is_sr": mis.get("sharpe", 0),
                    "oos_sr": om.get("sharpe", 0), "oos_cagr": om.get("cagr", 0),
                    "y22": m22.get("sharpe", 0), "mdd": m["mdd"], "net": net,
                })

    all_results.sort(key=lambda r: -r["oos_sr"])
    print(f"\n\nTop 20 by OOS SR:")
    print(f"{'k':>2} {'cw':>5}  {'FULL':>5}  {'IS':>5}  {'OOS':>5}  {'OOS_CAGR':>7}  {'2022':>6}  {'MDD':>6}  Sleeves")
    for r in all_results[:20]:
        sleeves_str = "+".join(s[:8] for s in r["combo"])
        print(f"  {r['k']}  {r['cw']:.2f}  {r['full_sr']:>5.2f}  {r['is_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['oos_cagr']*100:>6.1f}%  {r['y22']:>6.2f}  {r['mdd']*100:>5.1f}%  {sleeves_str}")

    overall_best = all_results[0]
    net = overall_best["net"]
    print(f"\nOVERALL BEST: {overall_best['combo']} cw={overall_best['cw']}")
    print(f"OOS SR = {overall_best['oos_sr']:.2f}")

    print("\n=== DETAIL ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix win 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 10-18", ("2010-03-11", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2008 cal", ("2008-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31")),
                        ("2025+", ("2025-01-01", "2027-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    net.to_frame("apex_v27_ret").to_csv(OUT / "apex_v27_returns.csv")
    (OUT / "apex_v27_meta.json").write_text(json.dumps({
        "best_combo": overall_best["combo"],
        "best_cw": overall_best["cw"],
        "best_oos_sr": overall_best["oos_sr"],
        "best_full_sr": overall_best["full_sr"],
        "top_10_combos": [[list(r["combo"]), r["cw"], r["oos_sr"], r["full_sr"]]
                          for r in all_results[:10]],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
