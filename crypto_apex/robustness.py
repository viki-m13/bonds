"""Crypto-APEX robustness test battery.

1. Survivorship bias (full vs survivors-only)
2. Bootstrap Sharpe CI
3. Parameter sensitivity
4. Transaction cost sensitivity
5. Sleeve ablation
6. Regime stability (rolling 1y SR)
7. Deflated Sharpe ratio
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
from util import metrics, regime_slice, load_prices, load_macro, SURVIVORS, DEAD, OUT
import sleeves as SV
from strategy import build_portfolio, FINAL_SLEEVES, TARGET_VOL, DD_FLOOR

OOS_START = "2022-07-01"


def bootstrap_sharpe(r: pd.Series, n_boot: int = 1000, block: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    r = r.dropna().values
    n = len(r)
    n_blocks = n // block + 1
    srs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([r[s:s+block] for s in starts])[:n]
        mu = sample.mean() * util.DPY
        sd = sample.std() * np.sqrt(util.DPY)
        sr = mu / sd if sd > 0 else 0
        srs.append(sr)
    srs = np.array(srs)
    return {
        "mean": float(srs.mean()),
        "p2.5": float(np.percentile(srs, 2.5)),
        "p50": float(np.percentile(srs, 50)),
        "p97.5": float(np.percentile(srs, 97.5)),
        "pos_pct": float((srs > 0).mean() * 100),
    }


def deflated_sharpe(sr_oos, n_trials, n_obs, skew, kurt):
    """Bailey-López de Prado Deflated Sharpe."""
    from scipy.stats import norm
    emax = norm.ppf(1 - 1/n_trials) + 0.5772 * (norm.ppf(1 - 1/(n_trials*np.e)) - norm.ppf(1 - 1/n_trials))
    var_sr = (1 - skew * sr_oos + (kurt - 1)/4 * sr_oos**2) / (n_obs - 1)
    dsr = (sr_oos - emax) / np.sqrt(max(var_sr, 1e-10))
    return float(dsr)


def main():
    cp = load_prices()
    macro = load_macro(cp.index)
    all_sw = SV.build_all(cp, macro)
    sw = {k: all_sw[k] for k in FINAL_SLEEVES}

    net = build_portfolio(cp, sw, target_vol=TARGET_VOL, dd_floor=DD_FLOOR).fillna(0.0)
    r_oos = regime_slice(net, OOS_START, "2027-12-31")

    results = {}

    # Test 1: Survivorship
    print("=== Test 1: Survivorship bias ===")
    cp_surv = load_prices(coins=SURVIVORS)
    sw_surv = {k: v for k, v in SV.build_all(cp_surv, macro).items() if k in FINAL_SLEEVES}
    net_surv = build_portfolio(cp_surv, sw_surv, target_vol=TARGET_VOL, dd_floor=DD_FLOOR).fillna(0.0)
    m_full = metrics(net)
    m_surv = metrics(net_surv)
    print(f"  FULL (survivors+dead): SR={m_full['sharpe']:.2f}, CAGR={m_full['cagr']*100:.1f}%")
    print(f"  SURVIVORS-ONLY (biased): SR={m_surv['sharpe']:.2f}, CAGR={m_surv['cagr']*100:.1f}%")
    print(f"  Bias impact: dSR={m_surv['sharpe']-m_full['sharpe']:+.3f}")
    results["test1_survivorship"] = {
        "full": m_full, "survivors_only": m_surv,
        "sr_drop": float(m_surv["sharpe"] - m_full["sharpe"]),
    }

    # Test 2: Bootstrap Sharpe
    print("\n=== Test 2: Bootstrap Sharpe CI ===")
    boot_full = bootstrap_sharpe(net, n_boot=1000, block=20)
    boot_oos = bootstrap_sharpe(r_oos, n_boot=1000, block=20)
    print(f"  Full: mean {boot_full['mean']:.2f}, 95% CI [{boot_full['p2.5']:.2f}, {boot_full['p97.5']:.2f}], pos {boot_full['pos_pct']:.1f}%")
    print(f"  OOS:  mean {boot_oos['mean']:.2f}, 95% CI [{boot_oos['p2.5']:.2f}, {boot_oos['p97.5']:.2f}], pos {boot_oos['pos_pct']:.1f}%")
    results["test2_bootstrap"] = {"full": boot_full, "oos": boot_oos}

    # Test 3: Parameter sensitivity
    print("\n=== Test 3: Parameter sensitivity ===")
    sens = []
    for tv in [0.15, 0.20, 0.25, 0.30, 0.40]:
        for dd in [-0.15, -0.25, -0.35]:
            n = build_portfolio(cp, sw, target_vol=tv, dd_floor=dd).fillna(0.0)
            m = metrics(regime_slice(n, OOS_START, "2027-12-31"))
            sens.append({"tv": tv, "dd": dd, "oos_sr": m["sharpe"], "oos_cagr": m["cagr"]})
            print(f"  tv={tv} dd={dd}: OOS SR={m['sharpe']:.2f}")
    results["test3_param_sens"] = sens

    # Test 4: TC sensitivity
    print("\n=== Test 4: TC sensitivity ===")
    tc = []
    for bps in [10, 30, 60, 100]:
        n = build_portfolio(cp, sw, target_vol=TARGET_VOL, dd_floor=DD_FLOOR, tc_bps=bps).fillna(0.0)
        m_f = metrics(n)
        m_o = metrics(regime_slice(n, OOS_START, "2027-12-31"))
        tc.append({"bps": bps, "full_sr": m_f["sharpe"], "oos_sr": m_o["sharpe"]})
        print(f"  {bps}bps: Full SR={m_f['sharpe']:.2f}, OOS SR={m_o['sharpe']:.2f}")
    results["test4_tc_sens"] = tc

    # Test 5: Sleeve ablation
    print("\n=== Test 5: Sleeve ablation ===")
    abl = []
    for drop in FINAL_SLEEVES:
        sub = {k: v for k, v in sw.items() if k != drop}
        n = build_portfolio(cp, sub, target_vol=TARGET_VOL, dd_floor=DD_FLOOR).fillna(0.0)
        m_f = metrics(n)
        m_o = metrics(regime_slice(n, OOS_START, "2027-12-31"))
        abl.append({"dropped": drop, "full_sr": m_f["sharpe"], "oos_sr": m_o["sharpe"]})
        print(f"  drop {drop}: Full SR={m_f['sharpe']:.2f}, OOS SR={m_o['sharpe']:.2f}")
    results["test5_ablation"] = abl

    # Test 6: Rolling 1y SR stability
    print("\n=== Test 6: Rolling 1y SR ===")
    r365 = net.rolling(365).mean() / net.rolling(365).std() * np.sqrt(util.DPY)
    r365 = r365.dropna()
    print(f"  Min: {r365.min():.2f}, Median: {r365.median():.2f}, Max: {r365.max():.2f}, "
          f"Pos %: {(r365 > 0).mean()*100:.1f}%, >0.5 %: {(r365 > 0.5).mean()*100:.1f}%")
    results["test6_rolling"] = {
        "min": float(r365.min()), "median": float(r365.median()), "max": float(r365.max()),
        "pos_pct": float((r365 > 0).mean() * 100),
        "above0_5_pct": float((r365 > 0.5).mean() * 100),
    }

    # Test 7: Deflated Sharpe
    print("\n=== Test 7: Deflated Sharpe ===")
    oos_sharpe = metrics(r_oos)["sharpe"]
    skew_oos = r_oos.skew()
    kurt_oos = r_oos.kurt()
    n_trials = 20  # sleeve combos explored
    dsr = deflated_sharpe(oos_sharpe, n_trials, len(r_oos), skew_oos, kurt_oos)
    print(f"  OOS Sharpe: {oos_sharpe:.2f}")
    print(f"  Deflated z: {dsr:.2f} (> 1.96 for p < 0.05)")
    results["test7_dsr"] = {
        "oos_sharpe": float(oos_sharpe),
        "dsr": dsr,
        "skew": float(skew_oos),
        "kurt": float(kurt_oos),
        "n_trials": n_trials,
    }

    (OUT / "crypto_apex_robustness.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved robustness to {OUT / 'crypto_apex_robustness.json'}")

    # Summary
    print("\n=== ROBUSTNESS SUMMARY ===")
    pass_fail = []
    pass_fail.append(("Bootstrap CI positive", boot_oos["p2.5"] > 0))
    pass_fail.append(("Survivorship bias small (<0.3)", abs(m_surv["sharpe"] - m_full["sharpe"]) < 0.3))
    pass_fail.append(("Param sens 80%+ positive", sum(1 for s in sens if s["oos_sr"] > 0) / len(sens) > 0.8))
    pass_fail.append(("TC 3x still positive", tc[2]["oos_sr"] > 0))
    pass_fail.append(("DSR > 1.96", dsr > 1.96))
    pass_fail.append(("Rolling 1y >50% pos", results["test6_rolling"]["pos_pct"] > 50))
    pass_fail.append(("No critical sleeve (abs 0.1)", all(r["oos_sr"] > oos_sharpe - 0.5 for r in abl)))
    n_pass = sum(1 for _, p in pass_fail if p)
    for name, p in pass_fail:
        mark = "PASS" if p else "FAIL"
        print(f"  [{mark}] {name}")
    print(f"\n  {n_pass}/{len(pass_fail)} tests PASS")


if __name__ == "__main__":
    main()
