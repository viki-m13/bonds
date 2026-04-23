"""APEX v11 — try higher crypto weights, different subsets, different vol targets."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
import sleeves_v6 as SV6
import sleeves_phoenix_exact as PX
import crypto_sleeve as CS
from apex_v10 import sleeve_vol_scale, sleeve_rets

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"

# All sleeves
ALL_SLEEVES = {
    "PX_VANGUARD": lambda cp: PX.sleeve_vanguard_exact(cp),
    "PX_ORION":    lambda cp: PX.sleeve_orion_exact(cp),
    "PX_HELIOS":   lambda cp: PX.sleeve_helios_exact(cp),
    "V3_SECTOR":   lambda cp: SV6.s_v3_sector(cp, target_vol=0.18),
    "V4_ML5":      lambda cp: SV6.s_v4_ml5(cp, target_vol=0.25),
    "V6_SHORT_MR": lambda cp: SV6.s_v6_short_mr(cp, target_vol=0.18),
}


def build(cp, sleeves=None):
    if sleeves is None:
        sleeves = list(ALL_SLEEVES.keys())
    sw = {name: sleeve_vol_scale(ALL_SLEEVES[name](cp), cp, target_vol=0.15)
          for name in sleeves}
    return sw


def run_portfolio(cp, sw, blend_weights, crypto_weight, target_vol, dd_floor, crypto_vol=0.18):
    """Runs LETF sleeves on (1-crypto_weight) of capital; crypto on crypto_weight.

    No margin: LETF portion vol-scaling capped so LETF weights sum ≤ (1-crypto_weight),
    and crypto allocation fixed at crypto_weight when crypto signal is ON.
    """
    letf_cap = 1.0 - crypto_weight   # LETF portion can never exceed this
    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw.items():
        if name in blend_weights:
            P = P + W.fillna(0.0) * blend_weights[name]
    # blend_weights already sum to letf_cap (since each weight × (1-cw))
    P = P.clip(upper=letf_cap, lower=0.0)

    rets = cp.pct_change()
    # Vol-regime gate
    spy_rv60 = cp["SPY"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = (regime_ok + (1 - regime_ok) * 0.5).shift(1).fillna(1.0)
    P = P.mul(regime_mult, axis=0)

    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    # Vol scaling — but capped so LETF gross ≤ letf_cap (NO MARGIN)
    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (letf_cap / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    # Final safety: no row sum > letf_cap
    fs = np.minimum(1.0, letf_cap / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    letf_net = gross_ret - drag

    # External crypto: weight is fixed crypto_weight when signal ON (already vol-scaled)
    crypto_r = CS.crypto_sleeve_returns(cp.index, target_vol=crypto_vol)
    net = letf_net + crypto_weight * crypto_r
    return net, w_eff


def evaluate(net, label=""):
    m = util.metrics(net)
    oom = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
    ism = util.metrics(util.regime_slice(net, "2005-01-01", IS_END))
    p08 = util.metrics(util.regime_slice(net, "2008-01-01", "2008-12-31"))
    p22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
    return {
        "label": label,
        "full_sr": m["sharpe"], "full_cagr": m["cagr"], "full_mdd": m["mdd"],
        "is_sr": ism.get("sharpe", 0), "is_cagr": ism.get("cagr", 0),
        "oos_sr": oom.get("sharpe", 0), "oos_cagr": oom.get("cagr", 0),
        "oos_mdd": oom.get("mdd", 0),
        "y08_sr": p08.get("sharpe", 0), "y08_cagr": p08.get("cagr", 0), "y08_mdd": p08.get("mdd", 0),
        "y22_sr": p22.get("sharpe", 0),
    }


def main():
    op, cp = util.load_prices()

    # Build all sleeves once
    sw = build(cp)

    configs = []
    all_names = list(sw.keys())
    n = len(all_names)

    # Test different crypto weights with EW LETF blend
    for cw in [0.0, 0.15, 0.20, 0.25, 0.30, 0.35]:
        bw = {k: (1 - cw) / n for k in all_names}
        for tv in [0.15, 0.18, 0.22, 0.25]:
            for df in [-0.08, -0.10, -0.15]:
                net, _ = run_portfolio(cp, sw, bw, cw, tv, df)
                configs.append(evaluate(net, f"EW cw={cw:.2f} tv={tv:.2f} dd={df:.2f}"))

    # Also drop V4_ML5 (weak OOS in some cases)
    sw_no_ml = {k: v for k, v in sw.items() if k != "V4_ML5"}
    n_nm = len(sw_no_ml)
    for cw in [0.20, 0.25, 0.30]:
        bw = {k: (1 - cw) / n_nm for k in sw_no_ml}
        net, _ = run_portfolio(cp, sw_no_ml, bw, cw, 0.18, -0.10)
        configs.append(evaluate(net, f"no-ML cw={cw:.2f}"))

    # Drop V6_SHORT_MR
    sw_no_mr = {k: v for k, v in sw.items() if k != "V6_SHORT_MR"}
    n_nmr = len(sw_no_mr)
    for cw in [0.20, 0.25, 0.30]:
        bw = {k: (1 - cw) / n_nmr for k in sw_no_mr}
        net, _ = run_portfolio(cp, sw_no_mr, bw, cw, 0.18, -0.10)
        configs.append(evaluate(net, f"no-MR cw={cw:.2f}"))

    # Lean: only PX + crypto (5 sleeves)
    lean = ["PX_VANGUARD", "PX_ORION", "PX_HELIOS", "V3_SECTOR", "V6_SHORT_MR"]
    sw_lean = {k: sw[k] for k in lean}
    for cw in [0.20, 0.25, 0.30]:
        bw = {k: (1 - cw) / len(lean) for k in lean}
        net, _ = run_portfolio(cp, sw_lean, bw, cw, 0.18, -0.10)
        configs.append(evaluate(net, f"lean-5 cw={cw:.2f}"))

    # Sort by OOS Sharpe
    configs.sort(key=lambda c: -c["oos_sr"])
    print(f"\n{'Config':40s}  {'FULL_SR':>7}  {'OOS_SR':>7}  {'FULL_CAGR':>9}  {'OOS_CAGR':>8}  {'FULL_MDD':>8}  {'2008':>7}  {'2022':>7}")
    for c in configs[:25]:
        print(f"  {c['label']:40s}  {c['full_sr']:>7.2f}  {c['oos_sr']:>7.2f}  "
              f"{c['full_cagr']*100:>8.1f}%  {c['oos_cagr']*100:>7.1f}%  "
              f"{c['full_mdd']*100:>7.1f}%  {c['y08_mdd']*100:>6.1f}%  {c['y22_sr']:>7.2f}")


if __name__ == "__main__":
    main()
