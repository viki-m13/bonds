"""APEX v15 — v14 + INVERSE hedge + MULTI_CRYPTO instead of BTC-only.

Changes from v14:
  + Add SL_INVERSE (10-15% allocation, crisis-active long-inverse-LETFs)
  + Replace BTC-only crypto with MULTI_CRYPTO (BTC + ETH + SOL)
  + Tune weights via sweep
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
from apex_v13 import build as v13_build, extend_cp
from apex_v14 import run_v14

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


LETF_BUILDERS = {
    "PX_VANGUARD":  lambda cp: PX.sleeve_vanguard_exact(cp),
    "PX_ORION":     lambda cp: PX.sleeve_orion_exact(cp),
    "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
    "SL_DUALBEAR":  lambda cp: SV12.sleeve_dualbear_defense(cp),
    "SL_CALENDAR":  lambda cp: SV12.sleeve_calendar(cp),
    "SL_VRP":       lambda cp: SV12.sleeve_vrp(cp),
    "SL_INVERSE":   lambda cp: SV15.sleeve_inverse(cp),
}


def build_v15(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
        W = fn(cp)
        # Each sleeve scaled to 15% vol
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def run_v15(cp, sw, blend_w, crypto_w, target_vol, dd_floor, dbs_override="strict",
            use_multi_crypto=True):
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

    # Dual-bear override
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

    # Crypto: multi or BTC-only
    if use_multi_crypto:
        crypto_r = SV15.multi_crypto_returns(cp.index, target_vol=0.20)
    else:
        import crypto_sleeve as CS
        crypto_r = CS.crypto_sleeve_returns(cp.index, target_vol=0.18)
    net = letf_net + crypto_w * crypto_r
    return net, w_eff


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    # Load inverse LETFs
    for t in ["SH", "PSQ", "SDS", "TBF"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s
    sw = build_v15(cp)
    print("Sleeves:", list(sw.keys()))

    # Sweep configs
    base_blends = {
        "v14+inverse10": {
            "PX_VANGUARD": 0.18, "PX_ORION": 0.18, "PX_HELIOS": 0.18,
            "SL_DUALBEAR": 0.18, "SL_CALENDAR": 0.10, "SL_VRP": 0.08,
            "SL_INVERSE": 0.10,
        },
        "v14+inverse15": {
            "PX_VANGUARD": 0.17, "PX_ORION": 0.17, "PX_HELIOS": 0.17,
            "SL_DUALBEAR": 0.17, "SL_CALENDAR": 0.09, "SL_VRP": 0.08,
            "SL_INVERSE": 0.15,
        },
        "v14+inverse20": {
            "PX_VANGUARD": 0.15, "PX_ORION": 0.15, "PX_HELIOS": 0.15,
            "SL_DUALBEAR": 0.17, "SL_CALENDAR": 0.10, "SL_VRP": 0.08,
            "SL_INVERSE": 0.20,
        },
        "heavy_dualbear": {
            "PX_VANGUARD": 0.15, "PX_ORION": 0.15, "PX_HELIOS": 0.15,
            "SL_DUALBEAR": 0.25, "SL_CALENDAR": 0.10, "SL_VRP": 0.05,
            "SL_INVERSE": 0.15,
        },
    }

    results = []
    for blend_name, blend in base_blends.items():
        for cw in [0.25, 0.30, 0.35, 0.40, 0.45]:
            for multi in [True, False]:
                for tv in [0.18, 0.22]:
                    # scale blend by (1-cw)
                    bw = {k: v * (1 - cw) for k, v in blend.items()}
                    net, _ = run_v15(cp, sw, bw, cw, tv, -0.10, "strict", multi)
                    m = util.metrics(net)
                    om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
                    m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
                    m08 = util.metrics(util.regime_slice(net, "2008-01-01", "2008-12-31"))
                    results.append({
                        "blend": blend_name, "cw": cw, "tv": tv, "multi": multi,
                        "full_sr": m["sharpe"], "full_cagr": m["cagr"], "full_mdd": m["mdd"],
                        "oos_sr": om.get("sharpe", 0), "oos_cagr": om.get("cagr", 0),
                        "y22_sr": m22.get("sharpe", 0),
                        "y08_sr": m08.get("sharpe", 0),
                        "net": net,
                    })

    # Sort by OOS Sharpe
    results.sort(key=lambda r: -r["oos_sr"])
    print(f"\n{'Blend':18s} {'cw':>5} {'tv':>5} {'multi':>5}  {'FULL':>5}  {'OOS':>5}  {'CAGR_F':>7}  {'CAGR_O':>7}  {'2008':>6}  {'2022':>6}  {'MDD':>6}")
    for r in results[:20]:
        print(f"  {r['blend']:18s}  {r['cw']:.2f}  {r['tv']:.2f}  {str(r['multi'])[:5]:>5s}  "
              f"{r['full_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['full_cagr']*100:>6.1f}%  {r['oos_cagr']*100:>6.1f}%  "
              f"{r['y08_sr']:>6.2f}  {r['y22_sr']:>6.2f}  {r['full_mdd']*100:>5.1f}%")

    best = results[0]
    print(f"\n\nBEST: {best['blend']} cw={best['cw']} tv={best['tv']} multi={best['multi']}")
    print(f"OOS SR = {best['oos_sr']:.2f}, Full SR = {best['full_sr']:.2f}")

    net = best["net"]
    print("\n=== BEST DETAIL ===")
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

    net.to_frame("apex_v15_ret").to_csv(OUT / "apex_v15_returns.csv")
    (OUT / "apex_v15_meta.json").write_text(json.dumps({
        "best": {k: v for k, v in best.items() if k != "net"},
        "blend": base_blends[best["blend"]],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
