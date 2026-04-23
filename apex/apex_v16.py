"""APEX v16 — v15 + 3 novel sleeves.

Adds: SL_CRASH_CONTRARIAN, SL_REAL_YIELD, SL_CROSS_DECORR
Plus adaptive EWMA-Sharpe softmax blending (research-agent suggestion).
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
from apex_v13 import extend_cp

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


LETF_BUILDERS = {
    "PX_VANGUARD":       lambda cp: PX.sleeve_vanguard_exact(cp),
    "PX_ORION":          lambda cp: PX.sleeve_orion_exact(cp),
    "PX_HELIOS":         lambda cp: PX.sleeve_helios_exact(cp),
    "SL_DUALBEAR":       lambda cp: SV12.sleeve_dualbear_defense(cp),
    "SL_CALENDAR":       lambda cp: SV12.sleeve_calendar(cp),
    "SL_VRP":            lambda cp: SV12.sleeve_vrp(cp),
    "SL_INVERSE":        lambda cp: SV15.sleeve_inverse(cp),
    "SL_CRASH_CONTRA":   lambda cp: SV16.sleeve_crash_contrarian(cp),
    "SL_REAL_YIELD":     lambda cp: SV16.sleeve_real_yield(cp),
    "SL_CROSS_DECORR":   lambda cp: SV16.sleeve_cross_decorr(cp),
}


def build_v16(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def adaptive_weights(R: pd.DataFrame, half_life_mu: int = 63,
                      half_life_sigma: int = 21, tau: float = 1.0,
                      lambda_shrink: float = 0.5) -> pd.DataFrame:
    """EWMA-Sharpe softmax weights (from research agent recipe).
    Returns a (T x N) matrix of daily weights.
    """
    ewma_mu = R.ewm(halflife=half_life_mu, min_periods=30).mean() * util.DPY
    ewma_sd = R.ewm(halflife=half_life_sigma, min_periods=30).std() * np.sqrt(util.DPY)
    sharpe = (ewma_mu / ewma_sd.replace(0, np.nan))
    sharpe_shift = sharpe.shift(1).fillna(0)
    # Softmax with temperature
    # Prevent huge values
    sharpe_shift = sharpe_shift.clip(lower=-3, upper=3)
    exp_s = np.exp(sharpe_shift / tau)
    w_adaptive = exp_s.div(exp_s.sum(axis=1), axis=0)
    # Shrink toward equal-weight
    n = R.shape[1]
    w_eq = pd.DataFrame(1.0 / n, index=R.index, columns=R.columns)
    w_final = (1 - lambda_shrink) * w_adaptive + lambda_shrink * w_eq
    # Cap per-sleeve weight at 30%
    w_final = w_final.clip(upper=0.30, lower=0.0)
    w_final = w_final.div(w_final.sum(axis=1), axis=0)
    return w_final


def run_v16(cp, sw, use_adaptive=True, static_weights=None,
            crypto_w=0.45, target_vol=0.22, dd_floor=-0.10,
            dbs_override="strict", use_multi_crypto=True):
    letf_cap = 1.0 - crypto_w

    # Compute sleeve returns for adaptive weights
    sr_dict = {name: PX._weights_to_ret(W, cp) for name, W in sw.items()}
    R = pd.DataFrame(sr_dict).fillna(0.0)

    if use_adaptive:
        w_df = adaptive_weights(R)
    else:
        if static_weights is None:
            static_weights = {k: 1.0 / len(sw) for k in sw}
        w_df = pd.DataFrame(index=R.index, columns=R.columns)
        for k, v in static_weights.items():
            w_df[k] = v

    # Apply (1-crypto_w) to LETF portion
    w_df = w_df * (1 - crypto_w)

    # Build portfolio weights
    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw.items():
        # Each sleeve's weight varies daily per w_df
        sleeve_wt = w_df[name].fillna(0.0)
        W_weighted = W.mul(sleeve_wt, axis=0)
        P = P + W_weighted.fillna(0.0)
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

    print("Building v16 sleeves (10 LETF sleeves)...")
    sw = build_v16(cp)
    print(f"Sleeves: {list(sw.keys())}")

    # Sleeve metrics
    print(f"\n{'Sleeve':18s}  {'SR':>5}  {'CAGR':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sw.items():
        r = PX._weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(r, "2022-01-01", "2022-12-31"))
        m08 = util.metrics(util.regime_slice(r, "2008-01-01", "2008-12-31"))
        print(f"  {name:18s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{om.get('sharpe',0):>5.2f}  {m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")

    # Test multiple configs
    results = []
    for use_adaptive in [True, False]:
        for cw in [0.30, 0.40, 0.45, 0.50]:
            for tv in [0.18, 0.22, 0.28]:
                net, _ = run_v16(cp, sw, use_adaptive=use_adaptive, crypto_w=cw, target_vol=tv)
                m = util.metrics(net)
                om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
                m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
                m08 = util.metrics(util.regime_slice(net, "2008-01-01", "2008-12-31"))
                results.append({
                    "adaptive": use_adaptive, "cw": cw, "tv": tv,
                    "full_sr": m["sharpe"], "full_cagr": m["cagr"], "full_mdd": m["mdd"],
                    "oos_sr": om.get("sharpe", 0), "oos_cagr": om.get("cagr", 0),
                    "y22_sr": m22.get("sharpe", 0),
                    "y08_sr": m08.get("sharpe", 0),
                    "net": net,
                })

    results.sort(key=lambda r: -r["oos_sr"])
    print(f"\n{'adapt':>5} {'cw':>5} {'tv':>5}  {'FULL':>5}  {'OOS':>5}  {'CAGR_F':>7}  {'CAGR_O':>7}  {'2008':>6}  {'2022':>6}  {'MDD':>6}")
    for r in results[:12]:
        print(f"  {str(r['adaptive'])[:5]:>5s}  {r['cw']:.2f}  {r['tv']:.2f}  "
              f"{r['full_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['full_cagr']*100:>6.1f}%  {r['oos_cagr']*100:>6.1f}%  "
              f"{r['y08_sr']:>6.2f}  {r['y22_sr']:>6.2f}  {r['full_mdd']*100:>5.1f}%")

    best = results[0]
    net = best["net"]
    print(f"\nBEST: adaptive={best['adaptive']} cw={best['cw']} tv={best['tv']}")
    print(f"OOS SR = {best['oos_sr']:.2f}, Full SR = {best['full_sr']:.2f}")

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

    net.to_frame("apex_v16_ret").to_csv(OUT / "apex_v16_returns.csv")
    (OUT / "apex_v16_meta.json").write_text(json.dumps({
        "best": {k: v for k, v in best.items() if k != "net"}
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
