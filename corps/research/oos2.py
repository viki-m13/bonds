"""SECOND one-shot OOS batch (disclosed as such) — sole surviving candidate:

  GRANITE-CL = GRANITE-C (>=3pt dislocation, <=5y maturity, 1 concurrent
  position per issuer) + LIMIT-ENTRY DISCIPLINE: accept the post-signal ask
  only if it is <= the latest prior mid + 0.25 (never chase a bounced price).

Spec frozen from the IS screen (variants.py). The matched random-entry control
faces the SAME fill discipline (the FLOWBACK-S red-team rule: any fill cap
must bind the control identically, or the excess is inflated).

This is the program's second OOS look overall (first: oos_validate.py). Each
candidate family gets exactly one look; GRANITE-CL has not been OOS-tested
before. Reported as printed.

  python corps/research/oos2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from granite_experiments import sig_disc, gate_mat5, issuer_cap_filter  # noqa: E402

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)
RNG = np.random.default_rng(41)


def limit_filter(bonds, fills, cap=0.25):
    kept = []
    for f in fills:
        b = bonds[f.six]
        i = np.searchsorted(b["day"], f.entry_day, side="left") - 1
        if i >= 0 and np.isfinite(b["mid"][i]) and f.entry_px <= b["mid"][i] + cap:
            kept.append(f)
    return kept


def limit_control(bonds, fills, lo, hi, cap=0.25, k=10):
    """Random gated days on the same bonds; identical fill cap and exit."""
    rets = []
    for f in fills:
        b = bonds[f.six]
        day = b["day"]
        g = b["elig"] & (b["mat"] <= 5)
        pool = np.flatnonzero(g)
        if not len(pool):
            continue
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        for _ in range(k):
            i = int(RNG.choice(pool))
            t = day[i]
            j = np.searchsorted(s_day, t, side="right")
            if j >= len(s_day) or s_day[j] - t > 7:
                continue
            ed = int(s_day[j]); ep = float(s_px[j])
            if ed < lo or ed > hi:
                continue
            ii = np.searchsorted(day, ed, side="left") - 1
            if ii < 0 or not np.isfinite(b["mid"][ii]) or ep > b["mid"][ii] + cap:
                continue
            kk = np.searchsorted(p_day, ed + 365, side="left")
            if kk < len(p_day) and p_day[kk] <= ed + MAXH:
                xd, xp = int(p_day[kk]), float(p_px[kk])
            else:
                k2 = np.searchsorted(p_day, ed + MAXH, side="right") - 1
                if k2 < 0 or p_day[k2] <= ed:
                    continue
                xd, xp = ed + MAXH, float(p_px[k2])
            acc = b["coupon"] / 100 / 365 * (xd - ed) * 100
            rets.append((xp - ep + acc) / ep)
    return np.array(rets)


def evaluate(bonds, lo, hi, label):
    fills = e2.run_events(bonds, sig_disc(3.0), min_hold=365, max_hold=MAXH,
                          date_lo=lo, date_hi=hi, extra_gate=gate_mat5)
    fills = issuer_cap_filter(fills, cap=1)
    fills = limit_filter(bonds, fills)
    ctl = limit_control(bonds, fills, lo, hi)
    s = e2.summarize(fills, control=ctl)
    print(f"  {label:14} n={s.get('n',0):5} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+6.2f}% "
          f"excess={s.get('excess_vs_control',0)*100:+5.2f}% "
          f"p={s.get('excess_p_boot',1):.3f}", flush=True)
    if fills:
        r = e2.mtm_nav(bonds, fills)
        if r:
            ps = e2.perf_stats(*r)
            s["mtm"] = ps
            print(f"    MTM cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:5.2f} "
                  f"maxdd={ps['maxdd']*100:6.1f}% vol={ps['vol_m_ann']*100:5.2f}%",
                  flush=True)
    from combine import save_fills
    save_fills(fills, ROOT / "research" / f"fills_granitecl_{label.strip().lower()}.json")
    return s


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    print("\n" + "=" * 68, flush=True)
    print("GRANITE-CL — limit-entry discipline — frozen spec, second OOS batch",
          flush=True)
    print("=" * 68, flush=True)
    out = {}
    print("\n[IS re-run, v3 coupons] 2003-2015:", flush=True)
    out["is"] = evaluate(bonds, *IS, "IS")
    print("\n[ONE-SHOT OOS] 2016-2025:", flush=True)
    out["oos"] = evaluate(bonds, *OOS, "OOS")
    print("\n[FULL SAMPLE] 2003-2025:", flush=True)
    out["full"] = evaluate(bonds, IS[0], OOS[1], "FULL")
    (ROOT / "research" / "oos2_results.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/oos2_results.json", flush=True)


if __name__ == "__main__":
    main()
