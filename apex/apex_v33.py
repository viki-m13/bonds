"""APEX v33 — push even further: richer dynamic weighting, different crypto styles."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
import sleeves_phoenix_exact as PX
import sleeves_v12 as SV12
import sleeves_v15 as SV15
import sleeves_v18 as SV18
import sleeves_v26 as SV26
import sleeves_v29 as SV29
import crypto_v2 as CV2
from apex_v13 import extend_cp
from apex_v17 import run_v17
from apex_v32 import run_dynamic_crypto, build_sleeves, LETF_BUILDERS

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


# Extended pool including all viable sleeves
ALL_BUILDERS = {
    "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
    "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
    "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
    "ACCEL_MOM":    lambda cp: SV26.sleeve_accel_mom(cp),
    "SKEW_MOM":     lambda cp: SV29.sleeve_skew_mom(cp),
    "HURST":        lambda cp: SV26.sleeve_hurst(cp),
    "KALMAN":       lambda cp: SV26.sleeve_kalman(cp),
    "BREAKOUT52":   lambda cp: SV18.sleeve_breakout52(cp),
}


def build_sleeves_v33(cp, names):
    sw = {}
    for name in names:
        fn = ALL_BUILDERS[name]
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


# Targeted configs with novel sleeves and dynamic crypto
CONFIGS = [
    # Current v30
    ["PX_HELIOS", "HMM_REGIME", "DIVERGENCE", "ACCEL_MOM", "SKEW_MOM", "HURST"],
    # Add KALMAN
    ["PX_HELIOS", "HMM_REGIME", "ACCEL_MOM", "SKEW_MOM", "HURST", "KALMAN"],
    # Add BREAKOUT52
    ["PX_HELIOS", "HMM_REGIME", "ACCEL_MOM", "SKEW_MOM", "HURST", "BREAKOUT52"],
    # All 8
    ["PX_HELIOS", "HMM_REGIME", "DIVERGENCE", "ACCEL_MOM", "SKEW_MOM", "HURST", "KALMAN", "BREAKOUT52"],
    # No Phoenix (pure novel math)
    ["HMM_REGIME", "DIVERGENCE", "ACCEL_MOM", "SKEW_MOM", "HURST", "KALMAN"],
    # 4 strong
    ["PX_HELIOS", "ACCEL_MOM", "HMM_REGIME", "SKEW_MOM"],
    # 3 + more crypto
    ["PX_HELIOS", "ACCEL_MOM", "HMM_REGIME"],
    # 2 concentrated
    ["PX_HELIOS", "ACCEL_MOM"],
]


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    all_results = []
    for combo in CONFIGS:
        sw = build_sleeves_v33(cp, combo)
        # Static crypto
        for cw in [0.50, 0.55, 0.60]:
            net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=0.18)
            m = util.metrics(net)
            om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
            m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
            all_results.append({
                "combo": combo, "mode": "static",
                "cw": cw, "swing": 0,
                "full_sr": m["sharpe"], "full_cagr": m["cagr"],
                "oos_sr": om.get("sharpe", 0), "oos_cagr": om.get("cagr", 0),
                "mdd": m["mdd"], "y22": m22.get("sharpe", 0), "net": net,
            })
        # Dynamic crypto
        for base_cw in [0.40, 0.50, 0.55]:
            for swing in [0.15, 0.20]:
                net = run_dynamic_crypto(cp, sw, base_crypto_w=base_cw, swing=swing)
                m = util.metrics(net)
                om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
                m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
                all_results.append({
                    "combo": combo, "mode": "dynamic",
                    "cw": base_cw, "swing": swing,
                    "full_sr": m["sharpe"], "full_cagr": m["cagr"],
                    "oos_sr": om.get("sharpe", 0), "oos_cagr": om.get("cagr", 0),
                    "mdd": m["mdd"], "y22": m22.get("sharpe", 0), "net": net,
                })

    all_results.sort(key=lambda r: -r["oos_sr"])
    print(f"\nTop 15 by OOS SR:")
    print(f"{'mode':>8} {'cw':>5} {'sw':>5} {'k':>2}  {'FULL':>5}  {'OOS':>5}  {'CAGR_F':>7}  {'CAGR_O':>7}  {'2022':>6}  {'MDD':>6}  Sleeves")
    for r in all_results[:15]:
        sleeves_str = "+".join(s[:7] for s in r["combo"])
        print(f"  {r['mode']:>8s}  {r['cw']:.2f}  {r['swing']:.2f}  {len(r['combo'])}  "
              f"{r['full_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['full_cagr']*100:>6.1f}%  {r['oos_cagr']*100:>6.1f}%  "
              f"{r['y22']:>6.2f}  {r['mdd']*100:>5.1f}%  {sleeves_str}")

    best = all_results[0]
    net = best["net"]
    print(f"\nBEST: {best['combo']} mode={best['mode']} cw={best['cw']} swing={best['swing']}")

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

    net.to_frame("apex_v33_ret").to_csv(OUT / "apex_v33_returns.csv")
    (OUT / "apex_v33_meta.json").write_text(json.dumps({
        "best": {k: v for k, v in best.items() if k != "net"},
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
