"""APEX v22 — 8 BEST-OOS sleeves with near-zero correlation.

Replace the "clean momentum" sleeve with PX_HELIOS (Phoenix's best, OOS 1.06).
Pick the strongest OOS sleeves that are also uncorrelated.
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
import sleeves_v12 as SV12
import sleeves_v15 as SV15
import sleeves_v16 as SV16
import sleeves_v17 as SV17
import sleeves_v18 as SV18
import sleeves_v19 as SV19
import sleeves_v20 as SV20
from apex_v13 import extend_cp
from apex_v17 import run_v17

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


# STRONGEST uncorrelated sleeves
LETF_BUILDERS = {
    "PX_HELIOS":      lambda cp: PX.sleeve_helios_exact(cp),   # OOS 1.06
    "BREAKOUT52":     lambda cp: SV18.sleeve_breakout52(cp),   # OOS 0.91
    "VOLZ_MOM":       lambda cp: SV19.sleeve_volz_mom(cp),     # OOS 0.84
    "DIVERGENCE":     lambda cp: SV18.sleeve_divergence(cp),   # OOS 0.82
    "HMM_REGIME":     lambda cp: SV18.sleeve_hmm(cp),          # OOS 0.79
    "RATE_MOM":       lambda cp: SV20.sleeve_rate_momentum(cp),# 2022 +0.59
    "CALENDAR":       lambda cp: SV12.sleeve_calendar(cp),     # near-zero corr
    "DUALBEAR":       lambda cp: SV12.sleeve_dualbear_defense(cp),
}


def build_v22(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
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

    print(f"Building v22 ({len(LETF_BUILDERS)} STRONGEST sleeves)...")
    sw = build_v22(cp)

    rets = {name: PX._weights_to_ret(W, cp) for name, W in sw.items()}
    R = pd.DataFrame(rets).fillna(0.0)

    print(f"\n{'Sleeve':15s}  {'SR':>5}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sw.items():
        r = rets[name]
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(r, "2022-01-01", "2022-12-31"))
        m08 = util.metrics(util.regime_slice(r, "2008-01-01", "2008-12-31"))
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")

    print("\nOOS pairwise correlations (2019+):")
    print(R.loc[OOS_START:].corr().round(2))
    n = R.shape[1]
    avg_oos_corr = (R.loc[OOS_START:].corr().values.sum() - n) / (n*(n-1))
    print(f"\nAvg OOS correlation: {avg_oos_corr:.3f}")

    # Blend configs
    print("\nBlend sweep:")
    print(f"{'cw':>5} {'tv':>5}  {'FULL':>5}  {'OOS':>5}  {'CAGR_O':>7}  {'2022':>6}  {'MDD':>6}")
    best = None
    for cw in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for tv in [0.15, 0.18, 0.22, 0.25]:
            net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=tv)
            m = util.metrics(net)
            om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
            m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
            result = {"cw": cw, "tv": tv, "full_sr": m["sharpe"], "oos_sr": om.get("sharpe", 0),
                      "oos_cagr": om.get("cagr", 0), "full_cagr": m["cagr"],
                      "y22": m22.get("sharpe", 0), "mdd": m["mdd"], "net": net}
            if best is None or result["oos_sr"] > best["oos_sr"]:
                best = result

    # Show top 6
    import copy
    all_results = []
    for cw in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for tv in [0.15, 0.18, 0.22, 0.25]:
            net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=tv)
            m = util.metrics(net)
            om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
            m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
            all_results.append({"cw": cw, "tv": tv, "full_sr": m["sharpe"],
                                "oos_sr": om.get("sharpe", 0),
                                "oos_cagr": om.get("cagr", 0),
                                "full_cagr": m["cagr"], "y22": m22.get("sharpe", 0),
                                "mdd": m["mdd"]})
    all_results.sort(key=lambda r: -r["oos_sr"])
    for r in all_results[:10]:
        print(f"  {r['cw']:.2f}  {r['tv']:.2f}  {r['full_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['oos_cagr']*100:>6.1f}%  {r['y22']:>6.2f}  {r['mdd']*100:>5.1f}%")

    net = best["net"]
    print(f"\nBEST: cw={best['cw']} tv={best['tv']}  OOS SR {best['oos_sr']:.2f}")
    print("\n=== BEST DETAIL ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix window 10-26", ("2010-03-11", "2027-12-31")),
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

    net.to_frame("apex_v22_ret").to_csv(OUT / "apex_v22_returns.csv")
    (OUT / "apex_v22_meta.json").write_text(json.dumps({
        "best": {k: v for k, v in best.items() if k != "net"},
        "sleeves": list(sw.keys()),
        "avg_oos_correlation": float(avg_oos_corr),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
