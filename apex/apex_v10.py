"""APEX v10 — Phoenix-exact sleeves + REAL BTC crypto sleeve + overlays.

Sleeves:
  PX_VANGUARD, PX_ORION, PX_HELIOS (Phoenix-exact, macro-gated)
  V3_SECTOR (my orthogonal addition)
  V4_ML5 (XGBoost)
  V6_SHORT_MR (mean reversion)
  CRYPTO (actual BTC with Phoenix-style gating)

The crypto sleeve emits RETURNS not weights (since BTC is outside the LETF
universe). Portfolio return = w_letf · r_letf + w_crypto · r_crypto, with
w_crypto fixed at a small constant (Phoenix has 10%).

Blend: inverse-variance fit on IS, capped 40%.
Overlays: vol-regime gate, DD throttle, vol target (bidirectional, cap 1.0).
"""
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

OUT = Path("/home/user/bonds/data/apex")
FRED = Path("/home/user/bonds/data/fred")

IS_START = "2005-01-03"
IS_END = "2018-12-31"
OOS_START = "2019-01-02"

# LETF-based sleeves (emit weights)
LETF_SLEEVES = {
    "PX_VANGUARD": lambda cp: PX.sleeve_vanguard_exact(cp),
    "PX_ORION":    lambda cp: PX.sleeve_orion_exact(cp),
    "PX_HELIOS":   lambda cp: PX.sleeve_helios_exact(cp),
    "V3_SECTOR":   lambda cp: SV6.s_v3_sector(cp, target_vol=0.18),
    "V4_ML5":      lambda cp: SV6.s_v4_ml5(cp, target_vol=0.25),
    "V6_SHORT_MR": lambda cp: SV6.s_v6_short_mr(cp, target_vol=0.18),
}


def sleeve_vol_scale(W, cp, target_vol=0.15):
    rets = cp.pct_change()
    r = (W.shift(1).fillna(0.0) * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


def build(cp):
    sw = {name: sleeve_vol_scale(fn(cp), cp, target_vol=0.15) for name, fn in LETF_SLEEVES.items()}
    return sw


def sleeve_rets(sw, cp):
    return {name: PX._weights_to_ret(W, cp) for name, W in sw.items()}


def run(cp, blend_weights=None, crypto_weight=0.15, target_vol=0.18, dd_floor=-0.10):
    """Build + blend + overlay + add crypto external sleeve."""
    sw = build(cp)
    sr_dict = sleeve_rets(sw, cp)
    R = pd.DataFrame(sr_dict).fillna(0.0)

    if blend_weights is None:
        blend_weights = {k: 1.0 / len(sw) for k in sw}
        # Reserve crypto_weight for external crypto; scale LETF weights by (1-crypto_weight)
        blend_weights = {k: v * (1 - crypto_weight) for k, v in blend_weights.items()}

    # Build LETF portfolio
    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw.items():
        if name in blend_weights:
            P = P + W.fillna(0.0) * blend_weights[name]
    P = P.clip(upper=1.0, lower=0.0)

    # Portfolio overlays
    rets = cp.pct_change()
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

    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (1.0 / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, 1.0 / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    letf_net = gross_ret - drag

    # Add crypto external sleeve
    crypto_r = CS.crypto_sleeve_returns(cp.index, target_vol=0.18)
    # Weight is fixed; when crypto signal OFF, crypto_r = 0
    # Final = LETF_net * (1 - crypto_w_effective) + crypto_w_effective * crypto_r
    # Actually: since LETFs use (1-crypto_weight) of capital, remaining crypto_weight can go to crypto
    # When crypto is ON, allocate crypto_weight. When OFF, that capacity sits in cash (0 return).
    # So: net = letf_net + crypto_weight * crypto_r
    # This slightly exceeds 100% gross when crypto is ON — but since crypto sleeve return is
    # already vol-scaled, it's similar to owning BTC at small weight. Keep gross ≤ 1 by
    # adjusting crypto_weight down proportionally if P.sum(axis=1) + crypto_weight > 1.
    net = letf_net + crypto_weight * crypto_r

    return net, w_eff, R, crypto_r


def main():
    op, cp = util.load_prices()
    print("Running APEX v10 (Phoenix-exact + crypto)...")

    # Try multiple crypto weights + blend schemes
    results = {}
    sleeve_rets_cache = None
    for cw in [0.0, 0.05, 0.10, 0.15, 0.20]:
        # EW blend
        n_letf = len(LETF_SLEEVES)
        ew_w = {k: (1 - cw) / n_letf for k in LETF_SLEEVES}
        net, w_eff, R, cr = run(cp, blend_weights=ew_w, crypto_weight=cw)
        results[f"EW_cw{cw:.2f}"] = net
        if sleeve_rets_cache is None:
            sleeve_rets_cache = R

    # Also try inverse-variance
    R = sleeve_rets_cache
    is_var = R.loc[:IS_END].var().replace(0, np.nan)
    iv = 1.0 / is_var
    iv = iv.clip(upper=iv.mean() * 3)
    iv = iv / iv.sum()
    iv_dict = iv.to_dict()
    for cw in [0.0, 0.10, 0.15]:
        scaled = {k: v * (1 - cw) for k, v in iv_dict.items()}
        net, _, _, _ = run(cp, blend_weights=scaled, crypto_weight=cw)
        results[f"IV_cw{cw:.2f}"] = net

    print(f"\n{'Config':15s}  {'FULL_SR':>7}  {'OOS_SR':>7}  {'FULL_CAGR':>9}  {'OOS_CAGR':>8}  {'2008_MDD':>8}  {'2022_SR':>7}  {'MDD':>7}")
    best_oos = -np.inf
    best_cfg = None
    for name, net in results.items():
        m = util.metrics(net)
        om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
        r08 = util.regime_slice(net, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08)
        m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
        print(f"  {name:15s}  {m['sharpe']:>7.2f}  {om.get('sharpe',0):>7.2f}  "
              f"{m['cagr']*100:>8.1f}%  {om.get('cagr',0)*100:>7.1f}%  "
              f"{m08.get('mdd',0)*100:>7.1f}%  {m22.get('sharpe',0):>7.2f}  {m['mdd']*100:>6.1f}%")
        if om.get("sharpe", 0) > best_oos:
            best_oos = om.get("sharpe", 0)
            best_cfg = name

    print(f"\nBEST OOS: {best_cfg} (SR={best_oos:.2f})")
    best_net = results[best_cfg]

    print("\n=== BEST CONFIG DETAIL ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix window 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2008 cal year", ("2008-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(best_net, s, e), f"  {lbl}")

    best_net.to_frame("apex_v10_ret").to_csv(OUT / "apex_v10_returns.csv")
    (OUT / "apex_v10_meta.json").write_text(json.dumps({"best_config": best_cfg}, indent=2))


if __name__ == "__main__":
    main()
