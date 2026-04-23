"""APEX v14 — tune v13 for best OOS Sharpe + CAGR.

Relaxes the DBS override (only trigger at DBS>=3, not >=2) and pushes crypto
back to 35%. Tests multiple configs.
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
import crypto_sleeve as CS
from apex_v13 import build, extend_cp

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


def run_v14(cp, sw, blend_w, crypto_w, target_vol, dd_floor,
            dbs_override="strict"):
    """Version with configurable DBS override strictness."""
    letf_cap = 1.0 - crypto_w
    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw.items():
        if name in blend_w:
            P = P + W.fillna(0.0) * blend_w[name]
    P = P.clip(upper=letf_cap, lower=0.0)

    rets = cp.pct_change()

    # Phoenix vol-regime gate
    spy_rv60 = cp["SPY"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = (regime_ok + (1 - regime_ok) * 0.5).shift(1).fillna(1.0)
    P = P.mul(regime_mult, axis=0)

    # Dual-bear override (strict: only at DBS>=3; aggressive: at DBS>=2)
    dbs = SV12.dual_bear_score(cp)
    dbs_mult = pd.Series(1.0, index=cp.index)
    if dbs_override == "strict":
        dbs_mult[dbs >= 3] = 0.5
        dbs_mult[dbs >= 4] = 0.25
    elif dbs_override == "aggressive":
        dbs_mult[dbs >= 2] = 0.5
        dbs_mult[dbs >= 3] = 0.25
    dbs_mult = dbs_mult.shift(1).fillna(1.0)
    P = P.mul(dbs_mult, axis=0)

    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (letf_cap / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, letf_cap / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    letf_net = gross_ret - drag

    crypto_r = CS.crypto_sleeve_returns(cp.index, target_vol=0.18)
    net = letf_net + crypto_w * crypto_r
    return net, w_eff


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    print("Building v14 sleeves...")
    sw = build(cp)

    # Sweep: crypto_weight, dbs_override, target_vol
    configs = []

    # Base blend (weights of LETF sleeves, sum to 1.0 -- then scaled by (1-crypto_w))
    base_blend = {
        "PX_VANGUARD": 0.20, "PX_ORION": 0.20, "PX_HELIOS": 0.20,
        "SL_DUALBEAR": 0.20, "SL_CALENDAR": 0.12, "SL_VRP": 0.08,
    }

    for cw in [0.20, 0.30, 0.35, 0.40]:
        for dbs in ["strict", "aggressive", "off"]:
            for tv in [0.18, 0.22, 0.28]:
                bw = {k: v * (1 - cw) for k, v in base_blend.items()}
                net, _ = run_v14(cp, sw, bw, cw, tv, -0.10, dbs_override=dbs)
                m = util.metrics(net)
                om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
                m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
                m08 = util.metrics(util.regime_slice(net, "2008-01-01", "2008-12-31"))
                configs.append({
                    "cw": cw, "dbs": dbs, "tv": tv,
                    "full_sr": m["sharpe"], "full_cagr": m["cagr"], "full_mdd": m["mdd"],
                    "oos_sr": om.get("sharpe", 0), "oos_cagr": om.get("cagr", 0),
                    "y22_sr": m22.get("sharpe", 0),
                    "y08_sr": m08.get("sharpe", 0),
                    "net": net,
                })

    configs.sort(key=lambda c: -c["oos_sr"])
    print(f"\n{'cw':>4} {'dbs':>11} {'tv':>5}  {'FULL_SR':>7}  {'OOS_SR':>7}  {'FULL_CAGR':>9}  {'OOS_CAGR':>8}  {'2008':>7}  {'2022':>7}  {'MDD':>7}")
    for c in configs[:20]:
        print(f"  {c['cw']:>.2f}  {c['dbs']:>11s}  {c['tv']:>.2f}  {c['full_sr']:>7.2f}  {c['oos_sr']:>7.2f}  "
              f"{c['full_cagr']*100:>8.1f}%  {c['oos_cagr']*100:>7.1f}%  "
              f"{c['y08_sr']:>7.2f}  {c['y22_sr']:>7.2f}  {c['full_mdd']*100:>6.1f}%")

    best = configs[0]
    print(f"\n\nBEST OOS SR = {best['oos_sr']:.2f}  (cw={best['cw']}, dbs={best['dbs']}, tv={best['tv']})")
    print("\n=== BEST DETAIL ===")
    net = best["net"]
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix window 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2008 cal", ("2008-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31")),
                        ("2025+", ("2025-01-01", "2027-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    net.to_frame("apex_v14_ret").to_csv(OUT / "apex_v14_returns.csv")
    (OUT / "apex_v14_meta.json").write_text(json.dumps({
        "best": {"cw": best["cw"], "dbs": best["dbs"], "tv": best["tv"]},
        "blend": base_blend,
    }, indent=2))


if __name__ == "__main__":
    main()
