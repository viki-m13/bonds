"""BEDROCK — the ONE-SHOT OOS batch. Run once, report as printed.

Admitted by the IS screens (see BEDROCK_RESEARCH.md §8):
  V2 (corp): GRANITE-XL entries + G1 spread-value + G4 issuer-curve.
      Decision metric = the repo's established XL admission comparison:
      paired MTM (Sharpe_m AND CAGR) vs the ungated baseline on identical
      conventions (real coupons, lagged recovery exits, depth weights),
      plus the hold-matched 1y entry-excess vs a real-coupon cap-matched
      control (p<0.01).
  A  (corp): fallen-angel proxy detector (drop<=-8%/20 rows, vol>=3x qv90,
      cs 0.03-crossing), ~1y holds, real coupons, vs real-coupon matched
      control.

Window: 2016-01-01 .. 2025-03-31 - 455d (the GRANITE OOS convention).

  python corps/research/bedrock_oos.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bedrock_v import (cl_fills, real_coupons, exit_lagged, build_cs_median,  # noqa: E402
                       gate_value, gate_issuer_curve, limit_control_real, RNG)
from bedrock_a import detect  # noqa: E402
from combos import depth_of  # noqa: E402

MAXH = 455
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)


def matched_control_real(bonds, fills, n_draws=15, seed=7):
    """engine2.matched_control with real coupons (coupon_inv)."""
    rng = np.random.default_rng(seed)
    per_bond = {}
    for f in fills:
        per_bond.setdefault(f.six, []).append(f)
    rets = []
    for six, fl in per_bond.items():
        b = bonds[six]
        gate = b["elig"]
        day = b["day"]
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        cp = float(b.get("coupon_inv", b["coupon"]))
        lo = min(f.entry_day for f in fl); hi = max(f.entry_day for f in fl)
        cand = np.flatnonzero(gate & (day >= lo - 30) & (day <= hi + 30))
        if not len(cand):
            continue
        picks = rng.choice(cand, size=n_draws * len(fl), replace=True)
        for i in picks:
            sd = day[i]
            j = np.searchsorted(s_day, sd, side="right")
            if j >= len(s_day) or s_day[j] - sd > 7:
                continue
            ed = int(s_day[j]); ep = float(s_px[j])
            k = np.searchsorted(p_day, ed + 365, side="left")
            if k < len(p_day) and p_day[k] <= ed + MAXH:
                xd, xp = int(p_day[k]), float(p_px[k])
            else:
                k2 = np.searchsorted(p_day, ed + MAXH, side="right") - 1
                if k2 < 0 or p_day[k2] <= ed:
                    continue
                xd, xp = ed + MAXH, float(p_px[k2])
            acc = cp / 100.0 / 365.0 * (xd - ed) * 100.0
            rets.append((xp - ep + acc) / ep)
    return np.array(rets)


def boot_p(fills, ctl):
    by = {}
    for f in fills:
        by.setdefault(f.six, []).append(f.ret)
    keys = list(by); boots = []
    cm_pool = np.asarray(ctl)
    for _ in range(2000):
        ks = RNG.choice(len(keys), size=len(keys), replace=True)
        sm = np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean()
        boots.append(sm - RNG.choice(cm_pool, size=len(cm_pool), replace=True).mean())
    return float((np.array(boots) <= 0).mean())


def mtm(bonds, fills, label):
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    days, nav, daily = e2.mtm_nav(bonds, fills, weights=w)
    ps = e2.perf_stats(days, nav, daily)
    r = np.array([f.ret for f in fills])
    print(f"  {label:22} n={len(fills):5} mean={r.mean()*100:+6.2f}% "
          f"win={(r>0).mean()*100:3.0f}% cagr={ps['cagr']*100:+6.2f}% "
          f"sharpe_m={ps['sharpe_m']:5.2f} dd={ps['maxdd']*100:6.1f}%", flush=True)
    ps.update({"n": len(fills), "mean": float(r.mean()), "win": float((r > 0).mean())})
    return ps


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    print("\n" + "=" * 68, flush=True)
    print("BEDROCK ONE-SHOT OOS — 2016-2024 — reported as printed", flush=True)
    print("=" * 68, flush=True)

    med = build_cs_median(bonds).to_dict()
    base = cl_fills(bonds, *OOS)
    g14 = gate_issuer_curve(bonds, issuers, gate_value(bonds, base, med))
    print(f"\n[V2] entries: baseline {len(base)}, G1+G4 {len(g14)}", flush=True)

    print("  paired MTM (real coupons, lagged recovery exits):", flush=True)
    out["v2_baseline"] = mtm(bonds, real_coupons(bonds, exit_lagged(bonds, base)), "baseline")
    out["v2_g1g4"] = mtm(bonds, real_coupons(bonds, exit_lagged(bonds, g14)), "G1+G4")

    print("  hold-matched 1y entry-excess (real-coupon control):", flush=True)
    for tag, fl0 in [("baseline", base), ("g1g4", g14)]:
        fl = real_coupons(bonds, fl0)
        r = np.array([f.ret for f in fl])
        c = limit_control_real(bonds, fl, *OOS)
        p = boot_p(fl, c)
        out[f"v2_{tag}_excess"] = {"n": len(fl), "mean": float(r.mean()),
                                   "ctl": float(c.mean()),
                                   "excess": float(r.mean() - c.mean()), "p": p}
        print(f"    {tag:9} n={len(fl):5} mean={r.mean()*100:+.2f}% ctl={c.mean()*100:+.2f}% "
              f"excess={100*(r.mean()-c.mean()):+.2f}% p={p:.4f}", flush=True)

    print("\n[A] fallen-angel proxy, OOS entries:", flush=True)
    ev = detect(bonds)
    fills = []
    for six, idx in ev.items():
        b = bonds[six]
        day = b["day"]
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        if not len(s_day) or not len(p_day):
            continue
        for i in idx:
            sd = day[i]
            j = np.searchsorted(s_day, sd, side="right")
            if j >= len(s_day) or s_day[j] - sd > 7:
                continue
            ed = int(s_day[j]); ep = float(s_px[j])
            if not (OOS[0] <= ed <= OOS[1]):
                continue
            k = np.searchsorted(p_day, ed + 365, side="left")
            if k < len(p_day) and p_day[k] <= ed + MAXH:
                xd, xp, st = int(p_day[k]), float(p_px[k]), False
            else:
                k2 = np.searchsorted(p_day, ed + MAXH, side="right") - 1
                if k2 < 0 or p_day[k2] <= ed:
                    continue
                xd, xp, st = ed + MAXH, float(p_px[k2]), True
            cp = float(b.get("coupon_inv", b["coupon"]))
            fills.append(e2.Fill(six, ed, ep, xd, xp, cp, st))
    if fills:
        r = np.array([f.ret for f in fills])
        c = matched_control_real(bonds, fills)
        p = boot_p(fills, c)
        out["a_oos"] = {"n": len(fills), "mean": float(r.mean()),
                        "win": float((r > 0).mean()),
                        "stale": float(np.mean([f.stale for f in fills])),
                        "ctl": float(c.mean()),
                        "excess": float(r.mean() - c.mean()), "p": p}
        print(f"  n={len(fills)} mean={r.mean()*100:+.2f}% win={(r>0).mean()*100:.0f}% "
              f"stale={100*np.mean([f.stale for f in fills]):.0f}% "
              f"ctl={c.mean()*100:+.2f}% excess={100*(r.mean()-c.mean()):+.2f}% p={p:.4f}", flush=True)
        out["a_mtm"] = mtm(bonds, fills, "A book (equal depth-wt)")

    p = ROOT / "research" / "bedrock_oos_results.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
