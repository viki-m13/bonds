"""S4 transfer test — issuer-curve cheapness inside GRANITE-XL / BEDROCK-V.
Pre-registered in BEDROCK_RESEARCH.md §8h.

S4 (frozen from BULWARK): at the signal row, >=2 CUSIP6 siblings with finite
positive cs within 10 days, and cs_own >= median(cs_siblings) + 0.005.

The primary control is the SIBLING-ELIGIBLE baseline (>=2 siblings present,
S4 condition ignored) — comparing S4 against the full book would confound the
gate with a large-issuer selection effect.

  python corps/research/s4_transfer.py [oos]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bedrock_v import (cl_fills, real_coupons, exit_lagged, build_cs_median,  # noqa: E402
                       gate_value, gate_issuer_curve, limit_control_real)
from combos import depth_of  # noqa: E402

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)
RNG = np.random.default_rng(4747)


def sib_cs(bonds, issuers, six, sig_i, maxlag=10):
    """Sibling spreads at the signal row (point-in-time)."""
    b = bonds[six]
    sd = int(b["day"][sig_i])
    vals = []
    for s6 in issuers.get(six[:6], []):
        if s6 == six:
            continue
        sb = bonds[s6]
        j = np.searchsorted(sb["day"], sd, side="right") - 1
        if j < 0 or sd - sb["day"][j] > maxlag:
            continue
        c = sb["cs"][j]
        if np.isfinite(c) and c > 0:
            vals.append(float(c))
    return vals


def split_s4(bonds, issuers, fills, min_gap=0.005):
    """-> (sibling_eligible, s4_pass, s4_fail)"""
    elig, ok, bad = [], [], []
    for f in fills:
        b = bonds[f.six]
        i = np.searchsorted(b["day"], f.entry_day, side="left") - 1
        if i < 0 or not (np.isfinite(b["cs"][i]) and b["cs"][i] > 0):
            continue
        vals = sib_cs(bonds, issuers, f.six, i)
        if len(vals) < 2:
            continue
        elig.append(f)
        (ok if float(b["cs"][i]) >= float(np.median(vals)) + min_gap else bad).append(f)
    return elig, ok, bad


def book(bonds, fills, label, lo, hi, ctl=True, base=None):
    """Audited pipeline stats: real coupons, lagged recovery exit, depth wts."""
    fl = real_coupons(bonds, exit_lagged(bonds, fills))
    if not fl:
        print(f"  {label:34} n=0", flush=True)
        return {"n": 0}
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fl]
    days, nav, daily = e2.mtm_nav(bonds, fl, weights=w)
    ps = e2.perf_stats(days, nav, daily)
    r = np.array([f.ret for f in fl])
    o = {"n": len(fl), "mean": float(r.mean()), "win": float((r > 0).mean()),
         "hold": float(np.mean([f.hold for f in fl])),
         "issuers": len({f.six[:6] for f in fl}), **ps}
    # Excess is measured HOLD-MATCHED (XL_AUDIT §6b): the control exits at
    # 365-455d, so the strategy leg must too. Comparing the ~250d recovery-exit
    # book against a ~380d control is the documented hold-mismatch defect and
    # makes every excess column spuriously negative.
    if ctl:
        f1y = real_coupons(bonds, fills)
        r1 = np.array([f.ret for f in f1y])
        o["mean_1y"] = float(r1.mean())
        o["hold_1y"] = float(np.mean([f.hold for f in f1y]))
        c = limit_control_real(bonds, f1y, lo, hi)
        if len(c):
            by = {}
            for f in f1y:
                by.setdefault(f.six[:6], []).append(f.ret)
            keys = list(by); boots = []
            for _ in range(2000):
                ks = RNG.choice(len(keys), size=len(keys), replace=True)
                sm = np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean()
                boots.append(sm - RNG.choice(c, size=len(c), replace=True).mean())
            o["ctl"] = float(c.mean())
            o["excess"] = float(r1.mean() - c.mean())
            o["excess_p"] = float((np.array(boots) <= 0).mean())
    if base:
        o["T1"] = bool(o["mean"] > base["mean"] and o["sharpe_m"] > base["sharpe_m"])
        o["T1_1y"] = bool(o.get("mean_1y", -9) > base.get("mean_1y", -9))
        o["T2"] = bool(o.get("excess_p", 1) < 0.01
                       and o.get("excess", -9) > base.get("excess", -9))
        o["T3"] = bool(o["maxdd"] >= base["maxdd"] - 0.05)
        o["admit"] = bool(o["T1"] and o["T2"] and o["T3"])
    print(f"  {label:34} n={o['n']:5} iss={o['issuers']:4} mean={o['mean']*100:+6.2f}% "
          f"win={o['win']*100:3.0f}% cagr={o['cagr']*100:+6.2f}% "
          f"sh={o['sharpe_m']:5.2f} dd={o['maxdd']*100:6.1f}%"
          + (f" | 1y-hold mean={o.get('mean_1y',float('nan'))*100:+6.2f}% "
             f"exc={o.get('excess',float('nan'))*100:+5.2f}% p={o.get('excess_p',float('nan')):.3f}"
             if ctl and "excess" in o else "")
          + (f"  T1={int(o['T1'])} T2={int(o['T2'])} T3={int(o['T3'])} "
             f"[{'ADMIT' if o['admit'] else 'reject'}]" if base else ""), flush=True)
    return o


def main():
    oos_mode = len(sys.argv) > 1 and sys.argv[1] == "oos"
    lo, hi = OOS if oos_mode else IS
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    base_f = cl_fills(bonds, lo, hi)
    print(f"GRANITE-XL entries: {len(base_f)}", flush=True)
    out = {}
    tag = "OOS 2016+" if oos_mode else "IS 2003-2015"

    print(f"\n[{tag}] GRANITE-XL", flush=True)
    out["A_granite_full"] = book(bonds, base_f, "A GRANITE-XL full", lo, hi)
    g_elig, g_ok, g_bad = split_s4(bonds, issuers, base_f)
    out["B_granite_sibelig"] = book(bonds, g_elig, "B  sibling-eligible (control)", lo, hi)
    out["C_granite_s4"] = book(bonds, g_ok, "C  + S4 pass", lo, hi,
                               base=out["B_granite_sibelig"])
    out["D_granite_s4fail"] = book(bonds, g_bad, "D  + S4 fail (complement)", lo, hi)

    print(f"\n[{tag}] BEDROCK-V", flush=True)
    bv = gate_issuer_curve(bonds, issuers, gate_value(bonds, base_f, med))
    out["E_bedrock_full"] = book(bonds, bv, "E BEDROCK-V full", lo, hi)
    b_elig, b_ok, b_bad = split_s4(bonds, issuers, bv)
    out["F_bedrock_sibelig"] = book(bonds, b_elig, "F  sibling-eligible (control)", lo, hi)
    out["G_bedrock_s4"] = book(bonds, b_ok, "G  + S4 pass", lo, hi,
                               base=out["F_bedrock_sibelig"])
    out["H_bedrock_s4fail"] = book(bonds, b_bad, "H  + S4 fail (complement)", lo, hi)

    if not oos_mode:
        print(f"\n[{tag}] locating S4 against the existing gates", flush=True)
        out["I_granite_g4"] = book(bonds, gate_issuer_curve(bonds, issuers, base_f),
                                   "I GRANITE + G4 only", lo, hi)
        out["J_granite_g1"] = book(bonds, gate_value(bonds, base_f, med),
                                   "J GRANITE + G1 only", lo, hi)

    p = ROOT / "research" / (f"s4_transfer2_{'oos' if oos_mode else 'is'}.json")
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
