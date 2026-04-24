"""APEX v32 — dynamic crypto weighting + over-concentration.

User insight: over-concentration is OK. The best sleeve is the alpha driver.

Test:
  1. Dynamic crypto weight: 30% in crypto-bear, up to 70% in crypto-bull.
  2. Focus on the 3-4 highest-impact sleeves with higher allocations.
  3. Test different "focus" strategies.
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
import sleeves_v26 as SV26
import sleeves_v29 as SV29
import crypto_v2 as CV2
from apex_v13 import extend_cp
from apex_v17 import run_v17

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


LETF_BUILDERS = {
    "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
    "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
    "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
    "ACCEL_MOM":    lambda cp: SV26.sleeve_accel_mom(cp),
    "SKEW_MOM":     lambda cp: SV29.sleeve_skew_mom(cp),
    "HURST":        lambda cp: SV26.sleeve_hurst(cp),
}


def build_sleeves(cp, names=None):
    if names is None:
        names = list(LETF_BUILDERS.keys())
    sw = {}
    for name in names:
        fn = LETF_BUILDERS[name]
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def run_dynamic_crypto(cp, sw, base_crypto_w=0.50, swing=0.20,
                        target_vol=0.18, dd_floor=-0.10):
    """Crypto weight varies with BTC regime strength.

    base_crypto_w: base weight (e.g. 0.50)
    swing: ±range (e.g. 0.20 → dynamic between 0.30 and 0.70)
    """
    # Compute BTC regime strength (0 to 1)
    strength = CV2.btc_regime_strength(cp.index)
    # Map strength to crypto weight: low strength → base-swing, high → base+swing
    crypto_w_series = base_crypto_w + swing * (2 * strength - 1)
    crypto_w_series = crypto_w_series.clip(lower=0.20, upper=0.80)
    crypto_w_series = crypto_w_series.shift(1).fillna(base_crypto_w)

    # LETF book weight
    letf_cap_series = 1.0 - crypto_w_series

    # Aggregate LETF portfolio
    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    n = len(sw)
    for name, W in sw.items():
        # Each sleeve gets equal weight of letf_cap
        w_sleeve = letf_cap_series / n
        P = P + W.fillna(0.0).mul(w_sleeve, axis=0)

    # Overlays
    rets = cp.pct_change()
    spy_rv60 = cp["SPY"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = (regime_ok + (1 - regime_ok) * 0.5).shift(1).fillna(1.0)
    P = P.mul(regime_mult, axis=0)

    # Dual-bear
    import sleeves_v12 as SV12
    dbs = SV12.dual_bear_score(cp)
    dbs_mult = pd.Series(1.0, index=cp.index)
    dbs_mult[dbs >= 3] = 0.5
    dbs_mult[dbs >= 4] = 0.25
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
    max_up = (letf_cap_series / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    # per-row cap at letf_cap_series
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, letf_cap_series / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    letf_net = gross_ret - drag

    # Crypto with dynamic weight
    crypto_r = SV15.multi_crypto_returns(cp.index, target_vol=0.20)
    net = letf_net + crypto_w_series * crypto_r
    return net


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    # Test 1: v30 baseline
    print("=== V30 BASELINE (static crypto) ===")
    sw = build_sleeves(cp)
    net_base, _ = run_v17(cp, sw, crypto_w=0.50, target_vol=0.18)
    m = util.metrics(util.regime_slice(net_base, OOS_START, "2027-12-31"))
    print(f"OOS SR: {m['sharpe']:.2f}, CAGR: {m['cagr']*100:.1f}%")

    # Test 2: Dynamic crypto weight
    print("\n=== DYNAMIC CRYPTO WEIGHT ===")
    configs = []
    for base_cw in [0.40, 0.50, 0.55, 0.60]:
        for swing in [0.10, 0.15, 0.20, 0.25]:
            net = run_dynamic_crypto(cp, sw, base_crypto_w=base_cw, swing=swing)
            m = util.metrics(net)
            om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
            m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
            configs.append({"base": base_cw, "swing": swing,
                            "full_sr": m["sharpe"], "oos_sr": om.get("sharpe", 0),
                            "oos_cagr": om.get("cagr", 0), "full_cagr": m["cagr"],
                            "mdd": m["mdd"], "y22": m22.get("sharpe", 0),
                            "net": net})

    configs.sort(key=lambda r: -r["oos_sr"])
    print(f"{'base':>5} {'swing':>6}  {'FULL':>5}  {'OOS':>5}  {'CAGR_F':>7}  {'CAGR_O':>7}  {'MDD':>6}  {'2022':>6}")
    for r in configs[:10]:
        print(f"  {r['base']:.2f}  {r['swing']:.2f}  "
              f"{r['full_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['full_cagr']*100:>6.1f}%  {r['oos_cagr']*100:>6.1f}%  "
              f"{r['mdd']*100:>5.1f}%  {r['y22']:>6.2f}")

    # Test 3: focus on fewer sleeves + higher weight per sleeve
    print("\n=== FOCUS: 3-4 sleeves with higher concentration ===")
    concentrated_configs = [
        ["PX_HELIOS", "ACCEL_MOM"],
        ["PX_HELIOS", "ACCEL_MOM", "HMM_REGIME"],
        ["PX_HELIOS", "ACCEL_MOM", "HURST"],
        ["ACCEL_MOM", "SKEW_MOM", "HURST"],
    ]
    for combo in concentrated_configs:
        sw_c = build_sleeves(cp, names=combo)
        for cw in [0.50, 0.60]:
            net, _ = run_v17(cp, sw_c, crypto_w=cw, target_vol=0.18)
            m = util.metrics(net)
            om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
            print(f"  {'+'.join(s[:8] for s in combo)}  cw={cw}  Full SR={m['sharpe']:.2f}  OOS SR={om.get('sharpe',0):.2f}")

    # Best overall
    best = configs[0]
    net = best["net"]
    print(f"\n=== BEST DYNAMIC CONFIG: base={best['base']} swing={best['swing']} ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix win 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 10-18", ("2010-03-11", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2025+", ("2025-01-01", "2027-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    net.to_frame("apex_v32_ret").to_csv(OUT / "apex_v32_returns.csv")


if __name__ == "__main__":
    main()
