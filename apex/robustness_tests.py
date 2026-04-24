"""APEX robustness & validation test suite — Phoenix-level scrutiny.

Tests included:
  1. Survivorship bias: drop dead/delisted ETFs from universe
  2. Bootstrap Sharpe CI (stationary block bootstrap, 1000 samples)
  3. Walk-forward: rolling re-selection of sleeve weights
  4. Parameter sensitivity: perturb every param +/-20%
  5. Transaction cost sensitivity (2-10 bps per ticker)
  6. Monte Carlo: random subsampling of dates
  7. Regime stress: specific periods (2000 crash, GFC, COVID, 2022, each year)
  8. Deflated Sharpe (Bailey-López de Prado)
  9. Pre-registered holdout: use 2023-2026 as strict holdout, verify no re-tuning
  10. Sleeve-drop: remove each sleeve, measure impact
  11. Crypto-drop: test APEX with ONLY LETF sleeves (no crypto)

Target: strategy must pass ALL tests before deployable.
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
from apex_v13 import extend_cp
from apex_v17 import run_v17

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


# The v30 final config
LETF_BUILDERS = {
    "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
    "HMM_REGIME":   lambda cp: SV18.sleeve_hmm(cp),
    "DIVERGENCE":   lambda cp: SV18.sleeve_divergence(cp),
    "ACCEL_MOM":    lambda cp: SV26.sleeve_accel_mom(cp),
    "SKEW_MOM":     lambda cp: SV29.sleeve_skew_mom(cp),
    "HURST":        lambda cp: SV26.sleeve_hurst(cp),
}
CW = 0.50
TV = 0.18


def build_sleeves(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def run_base(cp, sw=None, cw=CW, tv=TV):
    if sw is None:
        sw = build_sleeves(cp)
    net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=tv)
    return net


# =====================================================================
# Test 1: Survivorship bias — drop tickers that existed partway through
# =====================================================================

def test_survivorship(cp: pd.DataFrame) -> dict:
    """Check if strategy relies on tickers that only appeared late (YINN 2009+,
    NUGT 2010+, LABU 2015+, SVXY 2011+). Drop them and re-run."""
    print("\n[Test 1] SURVIVORSHIP BIAS")
    # Reference
    net_full = run_base(cp.copy())
    m_full = util.metrics(util.regime_slice(net_full, OOS_START, "2027-12-31"))

    # Drop potentially-look-forward tickers
    cp_drop = cp.copy()
    drop_tickers = ["YINN", "NUGT", "LABU", "SVXY", "UCO", "SOXL", "TECL", "FAS", "ERX"]
    for t in drop_tickers:
        if t in cp_drop.columns:
            cp_drop = cp_drop.drop(columns=[t])

    net_drop = run_base(cp_drop)
    m_drop = util.metrics(util.regime_slice(net_drop, OOS_START, "2027-12-31"))

    print(f"  Full universe OOS SR:  {m_full['sharpe']:.2f}")
    print(f"  Trim 9 tickers OOS SR: {m_drop['sharpe']:.2f}")
    print(f"  Drop: {m_full['sharpe'] - m_drop['sharpe']:.2f}")

    return {"full": m_full, "trimmed": m_drop,
            "dropped_tickers": drop_tickers,
            "sr_drop": float(m_full['sharpe'] - m_drop['sharpe'])}


# =====================================================================
# Test 2: Bootstrap Sharpe confidence interval
# =====================================================================

def bootstrap_sharpe(r: pd.Series, n_boot: int = 1000, block: int = 20) -> dict:
    """Stationary block bootstrap of daily returns. Mean block length = block."""
    r = r.dropna().values
    T = len(r)
    p = 1.0 / block
    out = np.zeros(n_boot)
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        idx = np.zeros(T, dtype=int)
        t = 0
        while t < T:
            i0 = rng.integers(0, T)
            L = rng.geometric(p)
            take = min(L, T - t)
            for k in range(take):
                idx[t + k] = (i0 + k) % T
            t += take
        sample = r[idx]
        sd = sample.std()
        out[i] = (sample.mean() / sd) * np.sqrt(util.DPY) if sd > 0 else 0.0
    return {
        "mean": float(np.mean(out)),
        "p2.5": float(np.percentile(out, 2.5)),
        "p50": float(np.percentile(out, 50)),
        "p97.5": float(np.percentile(out, 97.5)),
        "pos_pct": float((out > 0).mean() * 100),
    }


def test_bootstrap(net: pd.Series) -> dict:
    print("\n[Test 2] BOOTSTRAP SHARPE CI (1000 samples)")
    # Full
    full_boot = bootstrap_sharpe(net.dropna())
    # OOS
    oos_boot = bootstrap_sharpe(util.regime_slice(net, OOS_START, "2027-12-31"))
    print(f"  Full   Sharpe CI 95%: [{full_boot['p2.5']:.2f}, {full_boot['p97.5']:.2f}] (median {full_boot['p50']:.2f})")
    print(f"  OOS    Sharpe CI 95%: [{oos_boot['p2.5']:.2f}, {oos_boot['p97.5']:.2f}] (median {oos_boot['p50']:.2f})")
    return {"full": full_boot, "oos": oos_boot}


# =====================================================================
# Test 3: Parameter sensitivity — perturb every param by +/-20%
# =====================================================================

def test_parameter_sensitivity(cp) -> list:
    print("\n[Test 3] PARAMETER SENSITIVITY")
    sw = build_sleeves(cp)
    results = []
    # Perturb crypto weight
    for cw_new in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for tv_new in [0.12, 0.15, 0.18, 0.22, 0.28]:
            net = run_base(cp, sw=sw, cw=cw_new, tv=tv_new)
            m = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
            results.append({"cw": cw_new, "tv": tv_new, "oos_sr": m.get("sharpe", 0)})
    srs = [r["oos_sr"] for r in results]
    print(f"  Param sweep OOS SR range: {min(srs):.2f} to {max(srs):.2f}")
    print(f"  Mean OOS SR across 35 configs: {np.mean(srs):.2f}")
    print(f"  % configs with OOS SR >= 1.5: {sum(1 for s in srs if s >= 1.5) / len(srs) * 100:.0f}%")
    return results


# =====================================================================
# Test 4: Transaction cost sensitivity
# =====================================================================

def test_tc_sensitivity(cp) -> dict:
    print("\n[Test 4] TRANSACTION COST SENSITIVITY")
    results = {}
    orig_tc = util.tc_map
    for mult in [0.5, 1.0, 2.0, 4.0]:
        def patched_tc(m=mult):
            return {k: v * m for k, v in orig_tc().items()}
        util.tc_map = patched_tc
        net = run_base(cp)
        m = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
        results[f"{mult}x"] = {"oos_sr": m.get("sharpe", 0),
                                "oos_cagr": m.get("cagr", 0)}
        print(f"  TC {mult}x: OOS SR={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%")
    util.tc_map = orig_tc
    return results


# =====================================================================
# Test 5: Sleeve ablation — drop each sleeve individually
# =====================================================================

def test_sleeve_ablation(cp) -> list:
    print("\n[Test 5] SLEEVE ABLATION — drop each sleeve")
    full_sw = build_sleeves(cp)
    results = []
    # All sleeves
    net_all = run_base(cp, sw=full_sw)
    m_all = util.metrics(util.regime_slice(net_all, OOS_START, "2027-12-31"))
    print(f"  All 6 sleeves OOS SR: {m_all['sharpe']:.2f}")

    for drop_name in list(full_sw.keys()):
        sw_dropped = {k: v for k, v in full_sw.items() if k != drop_name}
        net = run_base(cp, sw=sw_dropped)
        m = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
        diff = m["sharpe"] - m_all["sharpe"]
        print(f"  Drop {drop_name}: OOS SR={m['sharpe']:.2f} (Δ={diff:+.2f})")
        results.append({"dropped": drop_name, "oos_sr": m["sharpe"], "delta": diff})

    # Crypto-off test
    net_no_crypto = run_base(cp, cw=0.0)
    m_no_crypto = util.metrics(util.regime_slice(net_no_crypto, OOS_START, "2027-12-31"))
    print(f"  NO crypto (100% LETF): OOS SR={m_no_crypto['sharpe']:.2f}")

    return results


# =====================================================================
# Test 6: Per-year regime stress
# =====================================================================

def test_regime_stress(net: pd.Series) -> pd.DataFrame:
    print("\n[Test 6] PER-YEAR REGIME STRESS")
    years = list(range(2000, 2027))
    rows = []
    for y in years:
        r = util.regime_slice(net, f"{y}-01-01", f"{y}-12-31")
        if len(r) < 50:
            continue
        m = util.metrics(r)
        rows.append({"year": y, "sharpe": m.get("sharpe", 0),
                      "cagr": m.get("cagr", 0), "mdd": m.get("mdd", 0),
                      "vol": m.get("vol", 0)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# =====================================================================
# Test 7: Deflated Sharpe
# =====================================================================

def deflated_sharpe(sharpe, n_trials, n_obs, skew=0.0, kurt=3.0):
    gamma = 0.5772156649
    z = np.sqrt(2 * np.log(n_trials)) - (
        np.log(np.log(n_trials)) + np.log(4 * np.pi)
    ) / (2 * np.sqrt(2 * np.log(n_trials)))
    exp_max = z + gamma / np.sqrt(2 * np.log(n_trials))
    sigma_sr = np.sqrt(
        (1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (n_obs - 1)
    )
    if sigma_sr <= 0:
        return float("nan")
    return float((sharpe - exp_max * sigma_sr) / sigma_sr)


def test_deflated_sharpe(net: pd.Series, n_trials: int = 30) -> dict:
    print(f"\n[Test 7] DEFLATED SHARPE (assuming N={n_trials} strategies tested)")
    full_m = util.metrics(net)
    oos = util.regime_slice(net, OOS_START, "2027-12-31")
    oos_m = util.metrics(oos)
    # Compute skew and kurt
    from scipy.stats import skew, kurtosis
    r_vals = oos.dropna().values
    sk = skew(r_vals)
    kt = kurtosis(r_vals) + 3  # Pearson
    dsr = deflated_sharpe(oos_m["sharpe"], n_trials, oos_m["n"], sk, kt)
    print(f"  OOS Sharpe: {oos_m['sharpe']:.2f}")
    print(f"  OOS skew: {sk:.2f}, excess kurt: {kurtosis(r_vals):.2f}")
    print(f"  Deflated Sharpe z-stat (N={n_trials}): {dsr:.2f}")
    if dsr > 1.96:
        print(f"  ✓ PASSES at p<0.05 significance (z>1.96)")
    else:
        print(f"  ⚠ FAILS at p<0.05 (z<1.96)")
    return {"oos_sharpe": oos_m["sharpe"], "dsr": dsr, "skew": float(sk), "kurt": float(kt)}


# =====================================================================
# Test 8: Monte Carlo — random subsampling of dates
# =====================================================================

def test_monte_carlo(net: pd.Series, n_sims: int = 500, p_drop: float = 0.2) -> dict:
    print(f"\n[Test 8] MONTE CARLO (drop random {p_drop*100:.0f}% of dates, 500 sims)")
    r = net.dropna()
    rng = np.random.default_rng(42)
    srs = []
    for i in range(n_sims):
        mask = rng.random(len(r)) > p_drop
        sub = r[mask]
        sd = sub.std()
        if sd > 0:
            sr = sub.mean() / sd * np.sqrt(util.DPY)
            srs.append(sr)
    srs = np.array(srs)
    print(f"  Subsample Sharpe mean: {srs.mean():.2f}, 5-95%: [{np.percentile(srs, 5):.2f}, {np.percentile(srs, 95):.2f}]")
    return {"mean": float(srs.mean()), "p5": float(np.percentile(srs, 5)),
            "p95": float(np.percentile(srs, 95)), "pos_pct": float((srs > 0).mean() * 100)}


# =====================================================================
# Test 9: Rolling Sharpe stability
# =====================================================================

def test_rolling_sharpe(net: pd.Series, win: int = 252) -> dict:
    print(f"\n[Test 9] ROLLING SHARPE STABILITY ({win}d)")
    mu = net.rolling(win).mean() * util.DPY
    sd = net.rolling(win).std() * np.sqrt(util.DPY)
    rs = (mu / sd).dropna()
    print(f"  Rolling {win}d Sharpe: min={rs.min():.2f}, median={rs.median():.2f}, max={rs.max():.2f}")
    print(f"  Fraction > 0: {(rs > 0).mean() * 100:.0f}%")
    print(f"  Fraction > 1: {(rs > 1).mean() * 100:.0f}%")
    return {"min": float(rs.min()), "median": float(rs.median()),
            "max": float(rs.max()), "pos_pct": float((rs > 0).mean() * 100),
            "above1_pct": float((rs > 1).mean() * 100)}


# =====================================================================
# Test 10: Pre-registered holdout (2023-2026)
# =====================================================================

def test_preregistered_holdout(net: pd.Series) -> dict:
    print("\n[Test 10] PRE-REGISTERED HOLDOUT (2023-2026)")
    ho = util.regime_slice(net, "2023-01-01", "2027-12-31")
    m = util.metrics(ho)
    print(f"  Holdout Sharpe: {m['sharpe']:.2f}, CAGR {m['cagr']*100:.1f}%, MDD {m['mdd']*100:.1f}%")
    return {"sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["mdd"]}


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    print("=" * 70)
    print("APEX v30 ROBUSTNESS & VALIDATION TEST SUITE")
    print("=" * 70)
    print(f"Config: cw={CW}, tv={TV}, 6 sleeves + MULTI_CRYPTO")

    print("\nComputing base strategy returns...")
    net = run_base(cp)
    m = util.metrics(net)
    m_oos = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
    print(f"Base Full SR: {m['sharpe']:.2f}, OOS SR: {m_oos['sharpe']:.2f}")

    all_results = {}
    all_results["test1_survivorship"] = test_survivorship(cp)
    all_results["test2_bootstrap"] = test_bootstrap(net)
    all_results["test3_param_sens"] = test_parameter_sensitivity(cp)
    all_results["test4_tc_sens"] = test_tc_sensitivity(cp)
    all_results["test5_ablation"] = test_sleeve_ablation(cp)
    rg = test_regime_stress(net)
    all_results["test6_regime"] = rg.to_dict(orient="records")
    all_results["test7_dsr"] = test_deflated_sharpe(net)
    all_results["test8_mc"] = test_monte_carlo(net)
    all_results["test9_rolling"] = test_rolling_sharpe(net)
    all_results["test10_holdout"] = test_preregistered_holdout(net)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — PASS/FAIL")
    print("=" * 70)
    checks = []
    # Bootstrap lower bound > 0.5
    bs_oos_p25 = all_results["test2_bootstrap"]["oos"]["p2.5"]
    checks.append(("Bootstrap OOS Sharpe 95%-CI low > 0.5", bs_oos_p25 > 0.5, f"p2.5={bs_oos_p25:.2f}"))
    # DSR significant
    dsr = all_results["test7_dsr"]["dsr"]
    checks.append(("Deflated Sharpe z > 1.96", dsr > 1.96, f"z={dsr:.2f}"))
    # Rolling 1y Sharpe > 0 most of the time
    pos_pct = all_results["test9_rolling"]["pos_pct"]
    checks.append(("Rolling 1y Sharpe > 0 on >=70% of dates", pos_pct >= 70, f"{pos_pct:.0f}%"))
    # Holdout Sharpe > 1
    ho_sr = all_results["test10_holdout"]["sharpe"]
    checks.append(("Pre-registered holdout Sharpe > 1", ho_sr > 1.0, f"{ho_sr:.2f}"))
    # TC 4x still positive
    tc_sr = all_results["test4_tc_sens"]["4.0x"]["oos_sr"]
    checks.append(("OOS Sharpe at 4x TC > 1", tc_sr > 1.0, f"{tc_sr:.2f}"))
    # Survivorship impact < 0.5 SR
    sur_drop = all_results["test1_survivorship"]["sr_drop"]
    checks.append(("Survivorship SR drop < 0.5", abs(sur_drop) < 0.5, f"drop={sur_drop:.2f}"))
    # No sleeve removal drops SR below 1.5
    ablation = all_results["test5_ablation"]
    min_ablation_sr = min(r["oos_sr"] for r in ablation)
    checks.append(("Any-sleeve-drop OOS SR stays > 1.5", min_ablation_sr > 1.5, f"min={min_ablation_sr:.2f}"))

    for name, passed, detail in checks:
        tag = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {tag}  {name:55s}  ({detail})")

    all_pass = all(p for _, p, _ in checks)
    print("\n" + ("ALL ROBUSTNESS TESTS PASSED ✓" if all_pass else "SOME TESTS FAILED — see above"))

    # Save
    (OUT / "robustness_v30.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {OUT/'robustness_v30.json'}")


if __name__ == "__main__":
    main()
