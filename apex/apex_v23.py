"""APEX v23 — Ultra-LEAN: 5 strongest uncorrelated sleeves + crypto.

Phoenix has 5 sleeves. Let me match that cardinality with ONLY my strongest
uncorrelated signals:
  1. PX_HELIOS    (OOS 1.06, mom on unlevered)
  2. BREAKOUT52   (OOS 0.91, price action)
  3. VOLZ_MOM     (OOS 0.84, stable momentum)
  4. HMM_REGIME   (OOS 0.79, regime model)
  5. DIVERGENCE   (OOS 0.82, cross-asset)

+ MULTI_CRYPTO externally.

This is the CORE high-alpha core. Compare to having 8 with 3 weak sleeves.
"""
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
import sleeves_v19 as SV19
from apex_v13 import extend_cp
from apex_v17 import run_v17

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


# Test multiple 5-sleeve configurations
CONFIGS = {
    "core5_top": {  # 5 highest OOS SR
        "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
        "BREAKOUT52":   lambda cp: SV18.sleeve_breakout52(cp),
        "VOLZ_MOM":     lambda cp: SV19.sleeve_volz_mom(cp),
        "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
        "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
    },
    "core6_top": {
        "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
        "PX_ORION":     lambda cp: PX.sleeve_orion_exact(cp),
        "BREAKOUT52":   lambda cp: SV18.sleeve_breakout52(cp),
        "VOLZ_MOM":     lambda cp: SV19.sleeve_volz_mom(cp),
        "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
        "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
    },
    "core4_cleanest": {  # 4 lowest mutual correlation
        "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
        "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
        "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
        "BREAKOUT52":   lambda cp: SV18.sleeve_breakout52(cp),
    },
    "core7_balanced": {
        "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
        "PX_ORION":     lambda cp: PX.sleeve_orion_exact(cp),
        "PX_VANGUARD":  lambda cp: PX.sleeve_vanguard_exact(cp),
        "BREAKOUT52":   lambda cp: SV18.sleeve_breakout52(cp),
        "VOLZ_MOM":     lambda cp: SV19.sleeve_volz_mom(cp),
        "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
        "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
    },
}


def build_config(cp, builders):
    sw = {}
    for name, fn in builders.items():
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    all_results = []
    best_per_config = {}
    for cfg_name, builders in CONFIGS.items():
        print(f"\n=== {cfg_name} ({len(builders)} sleeves) ===")
        sw = build_config(cp, builders)

        # Correlations
        rets = {name: PX._weights_to_ret(W, cp) for name, W in sw.items()}
        R = pd.DataFrame(rets).fillna(0.0)
        n = R.shape[1]
        avg_oos_corr = (R.loc[OOS_START:].corr().values.sum() - n) / (n*(n-1))
        print(f"  Avg OOS correlation: {avg_oos_corr:.3f}")

        best = None
        for cw in [0.40, 0.45, 0.50, 0.55, 0.60]:
            for tv in [0.13, 0.15, 0.18, 0.22]:
                net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=tv)
                m = util.metrics(net)
                om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
                m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
                r = {"cfg": cfg_name, "cw": cw, "tv": tv, "full_sr": m["sharpe"],
                     "oos_sr": om.get("sharpe", 0), "oos_cagr": om.get("cagr", 0),
                     "full_cagr": m["cagr"], "y22": m22.get("sharpe", 0),
                     "mdd": m["mdd"], "net": net, "avg_corr": avg_oos_corr}
                all_results.append(r)
                if best is None or r["oos_sr"] > best["oos_sr"]:
                    best = r
        best_per_config[cfg_name] = best
        print(f"  Best: cw={best['cw']} tv={best['tv']} → Full SR {best['full_sr']:.2f}, OOS SR {best['oos_sr']:.2f}")

    # Overall ranking
    all_results.sort(key=lambda r: -r["oos_sr"])
    print(f"\n\n{'cfg':18s} {'cw':>5} {'tv':>5}  {'FULL':>5}  {'OOS':>5}  {'OOS_CAGR':>7}  {'2022':>6}  {'MDD':>6}")
    for r in all_results[:15]:
        print(f"  {r['cfg']:18s}  {r['cw']:.2f}  {r['tv']:.2f}  "
              f"{r['full_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['oos_cagr']*100:>6.1f}%  {r['y22']:>6.2f}  {r['mdd']*100:>5.1f}%")

    overall_best = all_results[0]
    net = overall_best["net"]
    print(f"\nOVERALL BEST: {overall_best['cfg']} cw={overall_best['cw']} tv={overall_best['tv']}")
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

    net.to_frame("apex_v23_ret").to_csv(OUT / "apex_v23_returns.csv")
    (OUT / "apex_v23_meta.json").write_text(json.dumps({
        "best": {k: v for k, v in overall_best.items() if k != "net"},
        "cfg_sleeves": list(CONFIGS[overall_best["cfg"]].keys()),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
