"""R3 — IS-only combos on GRANITE-CL (frozen rules, paired comparisons):

  [S] depth-proportional sizing: weight each position by its effective entry
      depth, w = clip(depth_pts / 3.0, 0.5, 2.0), renormalized daily over the
      open book. Frozen rule (3.0 = the signal threshold, not a fitted knob).
      Same fills as equal-weight => pure accounting comparison.
  [X] dynamic recovery exit x limit entry: same limit-entry fills, exit at the
      first bid (>=21d) once the day's mid has recovered to the ENTRY-day
      trailing-60d median, else the 455d hard stop. Paired exit-policy test on
      identical entries (no control needed for the pairing; MTM decides).

Gate for OOS batch 3 (pre-registered): a combo goes to OOS only if it improves
BOTH IS Sharpe and IS CAGR over plain GRANITE-CL (0.99 / +11.45%).

  python corps/research/combos.py
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
from oos2 import limit_filter  # noqa: E402

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))


def cl_fills(bonds, lo, hi):
    f = e2.run_events(bonds, sig_disc(3.0), min_hold=365, max_hold=MAXH,
                      date_lo=lo, date_hi=hi, extra_gate=gate_mat5)
    f = issuer_cap_filter(f, cap=1)
    return limit_filter(bonds, f, cap=0.25)


def depth_of(bonds, f):
    b = bonds[f.six]
    i = np.searchsorted(b["day"], f.entry_day, side="left") - 1
    if i >= 0 and np.isfinite(b["med60"][i]):
        return float(b["med60"][i] - f.entry_px)
    return 3.0


def dynamic_exit(bonds, fills):
    out = []
    for f in fills:
        b = bonds[f.six]
        day = b["day"]; mid = b["mid"]; med60 = b["med60"]
        i0 = np.searchsorted(day, f.entry_day, side="left") - 1
        tgt = float(med60[i0]) if i0 >= 0 and np.isfinite(med60[i0]) else None
        if tgt is None:
            continue
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        xd = xp = None
        k0 = np.searchsorted(p_day, f.entry_day + 21, side="left")
        for k in range(k0, len(p_day)):
            if p_day[k] > f.entry_day + MAXH:
                break
            di = np.searchsorted(day, p_day[k], side="right") - 1
            if di >= 0 and np.isfinite(mid[di]) and mid[di] >= tgt:
                xd, xp, st = int(p_day[k]), float(p_px[k]), False
                break
        if xd is None:
            k2 = np.searchsorted(p_day, f.entry_day + MAXH, side="right") - 1
            if k2 < 0 or p_day[k2] <= f.entry_day:
                continue
            xd, xp, st = f.entry_day + MAXH, float(p_px[k2]), True
        out.append(e2.Fill(f.six, f.entry_day, f.entry_px, xd, xp, f.coupon, st))
    return out


def show(bonds, fills, label, weights=None):
    r = e2.mtm_nav(bonds, fills, weights=weights)
    ps = e2.perf_stats(*r)
    rr = np.array([f.ret for f in fills])
    hold = np.mean([f.hold for f in fills])
    print(f"  {label:26} n={len(fills):5} mean={rr.mean()*100:+6.2f}% "
          f"hold={hold:4.0f}d cagr={ps['cagr']*100:+6.2f}% "
          f"sharpe_m={ps['sharpe_m']:5.2f} maxdd={ps['maxdd']*100:6.1f}%", flush=True)
    ps["n"] = len(fills); ps["mean_ret"] = float(rr.mean()); ps["mean_hold"] = float(hold)
    return ps


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    fills = cl_fills(bonds, *IS)
    print(f"\nGRANITE-CL IS fills: {len(fills)}", flush=True)
    out["base"] = show(bonds, fills, "equal-weight, 1y hold")

    print("\n[S] depth-proportional sizing (paired, same fills):", flush=True)
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    out["sized"] = show(bonds, fills, "depth-weighted", weights=w)

    print("\n[X] dynamic recovery exit (paired, same entries):", flush=True)
    dfills = dynamic_exit(bonds, fills)
    out["dynamic"] = show(bonds, dfills, "recovery exit (21d+)")
    print("\n[S+X] both:", flush=True)
    wd = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in dfills]
    out["sized_dynamic"] = show(bonds, dfills, "depth-weighted + recovery", weights=wd)

    gate = out["base"]
    print("\n[GATE] OOS batch 3 admission (must beat IS Sharpe 0.99 AND CAGR +11.45%):",
          flush=True)
    for k in ("sized", "dynamic", "sized_dynamic"):
        ok = (out[k]["sharpe_m"] or 0) > (gate["sharpe_m"] or 0) and \
             out[k]["cagr"] > gate["cagr"]
        print(f"  {k:16} -> {'ADMIT' if ok else 'reject'}", flush=True)
        out[k]["admit_oos"] = bool(ok)

    (ROOT / "research" / "combos_is.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/combos_is.json", flush=True)


if __name__ == "__main__":
    main()
