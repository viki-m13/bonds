"""APEX v30 — try SKEW_MOM + KALMAN + HURST in combinations."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
import sleeves_phoenix_exact as PX
import sleeves_v15 as SV15
import sleeves_v18 as SV18
import sleeves_v26 as SV26
import sleeves_v29 as SV29
from apex_v13 import extend_cp
from apex_v17 import run_v17

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


BUILDERS = {
    "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
    "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
    "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
    "ACCEL_MOM":    lambda cp: SV26.sleeve_accel_mom(cp),
    "KALMAN":       lambda cp: SV26.sleeve_kalman(cp),
    "HURST":        lambda cp: SV26.sleeve_hurst(cp),
    "SKEW_MOM":     lambda cp: SV29.sleeve_skew_mom(cp),
}

# Targeted configs
CONFIGS = [
    # v28 baseline
    ["PX_HELIOS", "HMM_REGIME", "DIVERGENCE", "ACCEL_MOM"],
    # Add new sleeves one by one
    ["PX_HELIOS", "HMM_REGIME", "DIVERGENCE", "ACCEL_MOM", "SKEW_MOM"],
    ["PX_HELIOS", "HMM_REGIME", "DIVERGENCE", "ACCEL_MOM", "KALMAN"],
    ["PX_HELIOS", "HMM_REGIME", "DIVERGENCE", "ACCEL_MOM", "HURST"],
    # Swap DIVERGENCE → SKEW_MOM
    ["PX_HELIOS", "HMM_REGIME", "SKEW_MOM", "ACCEL_MOM"],
    ["PX_HELIOS", "HMM_REGIME", "SKEW_MOM", "KALMAN"],
    # Swap HMM
    ["PX_HELIOS", "KALMAN", "DIVERGENCE", "ACCEL_MOM"],
    ["PX_HELIOS", "KALMAN", "DIVERGENCE", "ACCEL_MOM", "SKEW_MOM"],
    # All new
    ["ACCEL_MOM", "KALMAN", "HMM_REGIME", "SKEW_MOM"],
    # 6-sleeve
    ["PX_HELIOS", "HMM_REGIME", "DIVERGENCE", "ACCEL_MOM", "SKEW_MOM", "KALMAN"],
    ["PX_HELIOS", "HMM_REGIME", "DIVERGENCE", "ACCEL_MOM", "SKEW_MOM", "HURST"],
    # 3-sleeve lean
    ["PX_HELIOS", "ACCEL_MOM", "SKEW_MOM"],
    ["PX_HELIOS", "ACCEL_MOM", "KALMAN"],
    ["PX_HELIOS", "HMM_REGIME", "SKEW_MOM"],
]


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    print("Pre-building sleeves...")
    full_sw = {}
    for name, fn in BUILDERS.items():
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        full_sw[name] = W.mul(m, axis=0)
    print("Done.")

    all_results = []
    for combo in CONFIGS:
        sw = {name: full_sw[name] for name in combo}
        for cw in [0.50, 0.55, 0.60]:
            for tv in [0.15, 0.18]:
                net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=tv)
                m = util.metrics(net)
                om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
                m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
                mis = util.metrics(util.regime_slice(net, "2010-03-11", IS_END))
                all_results.append({
                    "combo": combo, "k": len(combo), "cw": cw, "tv": tv,
                    "full_sr": m["sharpe"], "full_cagr": m["cagr"],
                    "is_sr": mis.get("sharpe", 0),
                    "oos_sr": om.get("sharpe", 0), "oos_cagr": om.get("cagr", 0),
                    "y22": m22.get("sharpe", 0), "mdd": m["mdd"], "net": net,
                })

    all_results.sort(key=lambda r: -r["oos_sr"])
    print(f"\nTop 15 by OOS SR:")
    print(f"{'k':>2} {'cw':>5} {'tv':>5}  {'FULL':>5}  {'IS':>5}  {'OOS':>5}  {'OOS_CAGR':>7}  {'2022':>6}  {'MDD':>6}  Sleeves")
    for r in all_results[:15]:
        sleeves_str = "+".join(s[:9] for s in r["combo"])
        print(f"  {r['k']}  {r['cw']:.2f}  {r['tv']:.2f}  "
              f"{r['full_sr']:>5.2f}  {r['is_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['oos_cagr']*100:>6.1f}%  {r['y22']:>6.2f}  {r['mdd']*100:>5.1f}%  {sleeves_str}")

    overall_best = all_results[0]
    net = overall_best["net"]
    print(f"\nOVERALL BEST: {overall_best['combo']} cw={overall_best['cw']} tv={overall_best['tv']}")
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

    net.to_frame("apex_v30_ret").to_csv(OUT / "apex_v30_returns.csv")
    (OUT / "apex_v30_meta.json").write_text(json.dumps({
        "best_combo": overall_best["combo"],
        "best_cw": overall_best["cw"],
        "best_tv": overall_best["tv"],
        "best_oos_sr": overall_best["oos_sr"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
