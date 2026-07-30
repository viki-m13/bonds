"""Downside-limiting experiments for GRANITE-XL — IS-only (2003-2015).

Pre-registered gate for OOS admission (fixed before running):
  keep >= 90% of IS CAGR  AND  cut IS maxDD by >= 20% relative
  AND not gut the crisis vintages (2008-09 mean/trade must stay positive).

  E1  entry-price floor: skip entries with ask below {70, 75, 80}
  E2  bond-specific-only: skip entries where a same-issuer sibling also
      printed >=3pt below ITS median within the prior 5 days
  E3  thesis-break exit: after entry, if any sibling collapses (mid >=5pts
      below its own med60), exit at the next bid (>=21d min hold kept)
  E4  mark stop-loss: exit at next bid once mid drops {10, 15, 20} pts
      below entry (expected to fail — sells crisis bottoms; tested honestly)

  python corps/research/downside.py
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
from combos import dynamic_exit, depth_of  # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")
MAXH = 455


def build_imap(bonds):
    m = {}
    for six in bonds:
        m.setdefault(six[:6], []).append(six)
    return m


def cl_entries(bonds, lo, hi):
    f = e2.run_events(bonds, sig_disc(3.0), min_hold=365, max_hold=MAXH,
                      date_lo=lo, date_hi=hi, extra_gate=gate_mat5)
    f = issuer_cap_filter(f, cap=1)
    return limit_filter(bonds, f, cap=0.25)


def sib_dislocated_at(bonds, imap, six, day_i, window=5):
    for p in imap.get(six[:6], []):
        if p == six:
            continue
        bp = bonds[p]
        med = bp.get("med60")
        if med is None:
            continue
        j1 = np.searchsorted(bp["day"], day_i, side="right")
        j0 = np.searchsorted(bp["day"], day_i - window)
        for j in range(j0, j1):
            if np.isfinite(bp["s_px"][j]) and np.isfinite(med[j]) \
               and (bp["s_px"][j] - med[j]) <= -3.0:
                return True
    return False


def thesis_break_exit(bonds, imap, fills):
    """Recovery exit, but also exit at the next bid if any sibling's mid falls
    >=5pts below its own median after our entry (new adverse information)."""
    out = []
    for f in fills:
        b = bonds[f.six]
        day = b["day"]; mid = b["mid"]; med60 = b["med60"]
        i0 = np.searchsorted(day, f.entry_day, side="left") - 1
        tgt = float(med60[i0]) if i0 >= 0 and np.isfinite(med60[i0]) else None
        if tgt is None:
            continue
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        # build sibling-break day (first day after entry when a sib collapses)
        brk = None
        for p in imap.get(f.six[:6], []):
            if p == f.six:
                continue
            bp = bonds[p]
            medp = bp.get("med60")
            if medp is None:
                continue
            j0 = np.searchsorted(bp["day"], f.entry_day, side="right")
            j1 = np.searchsorted(bp["day"], f.entry_day + MAXH, side="right")
            for j in range(j0, j1):
                if np.isfinite(bp["mid"][j]) and np.isfinite(medp[j]) \
                   and (bp["mid"][j] - medp[j]) <= -5.0:
                    d = int(bp["day"][j])
                    brk = d if brk is None else min(brk, d)
                    break
        xd = xp = None
        k0 = np.searchsorted(p_day, f.entry_day + 21, side="left")
        for k in range(k0, len(p_day)):
            if p_day[k] > f.entry_day + MAXH:
                break
            di = np.searchsorted(day, p_day[k], side="right") - 1
            recovered = di >= 0 and np.isfinite(mid[di]) and mid[di] >= tgt
            broke = brk is not None and p_day[k] >= brk
            if recovered or broke:
                xd, xp, st = int(p_day[k]), float(p_px[k]), False
                break
        if xd is None:
            k2 = np.searchsorted(p_day, f.entry_day + MAXH, side="right") - 1
            if k2 < 0 or p_day[k2] <= f.entry_day:
                continue
            xd, xp, st = f.entry_day + MAXH, float(p_px[k2]), True
        out.append(e2.Fill(f.six, f.entry_day, f.entry_px, xd, xp, f.coupon, st))
    return out


def stop_loss_exit(bonds, fills, stop):
    """Recovery exit + stop: exit at next bid once mid <= entry_px - stop."""
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
            if di < 0 or not np.isfinite(mid[di]):
                continue
            if mid[di] >= tgt or mid[di] <= f.entry_px - stop:
                xd, xp, st = int(p_day[k]), float(p_px[k]), False
                break
        if xd is None:
            k2 = np.searchsorted(p_day, f.entry_day + MAXH, side="right") - 1
            if k2 < 0 or p_day[k2] <= f.entry_day:
                continue
            xd, xp, st = f.entry_day + MAXH, float(p_px[k2]), True
        out.append(e2.Fill(f.six, f.entry_day, f.entry_px, xd, xp, f.coupon, st))
    return out


def book(bonds, fills, label, base=None):
    if not fills:
        print(f"  {label}: no fills", flush=True)
        return None
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    r = e2.mtm_nav(bonds, fills, weights=w)
    ps = e2.perf_stats(*r)
    rr = np.array([f.ret for f in fills])
    gfc = [f.ret for f in fills if e2.D("2008-01-01") <= f.entry_day <= e2.D("2009-12-31")]
    ps.update({"n": len(fills), "mean_ret": float(rr.mean()),
               "win": float((rr > 0).mean()),
               "p5": float(np.percentile(rr, 5)),
               "gfc_mean": float(np.mean(gfc)) if gfc else None,
               "gfc_n": len(gfc)})
    gate = ""
    if base:
        keep = ps["cagr"] / base["cagr"]
        ddcut = 1 - ps["maxdd"] / base["maxdd"]
        ok = keep >= 0.90 and ddcut >= 0.20 and (ps["gfc_mean"] or 0) > 0
        ps["admit"] = bool(ok)
        gate = f" | keepCAGR={keep*100:.0f}% ddCut={ddcut*100:+.0f}% -> {'ADMIT' if ok else 'reject'}"
    print(f"  {label:26} n={len(fills):5} mean={rr.mean()*100:+6.2f}% p5={ps['p5']*100:+6.1f}% "
          f"gfc={ps['gfc_mean']*100 if gfc else 0:+5.1f}% cagr={ps['cagr']*100:+6.2f}% "
          f"sharpe={ps['sharpe_m']:.2f} maxdd={ps['maxdd']*100:6.1f}%{gate}", flush=True)
    return ps


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    imap = build_imap(bonds)
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    entries = cl_entries(bonds, IS_LO, IS_HI)
    xl = dynamic_exit(bonds, entries)
    print(f"\n[BASE] XL IS book:", flush=True)
    base = book(bonds, xl, "XL base")
    out["base"] = base

    print("\n[E1] entry-price floor:", flush=True)
    for floor in (70.0, 75.0, 80.0):
        fl = [f for f in entries if f.entry_px >= floor]
        out[f"e1_{floor:.0f}"] = book(bonds, dynamic_exit(bonds, fl),
                                      f"floor>={floor:.0f}", base)

    print("\n[E2] bond-specific-only entries:", flush=True)
    solo = [f for f in entries
            if not sib_dislocated_at(bonds, imap, f.six, f.entry_day)]
    out["e2"] = book(bonds, dynamic_exit(bonds, solo), "no sib-dislocation", base)

    print("\n[E3] thesis-break exit:", flush=True)
    out["e3"] = book(bonds, thesis_break_exit(bonds, imap, entries),
                     "sibling-break exit", base)

    print("\n[E4] mark stop-loss:", flush=True)
    for stop in (10.0, 15.0, 20.0):
        out[f"e4_{stop:.0f}"] = book(bonds, stop_loss_exit(bonds, entries, stop),
                                     f"stop -{stop:.0f}pt", base)

    (ROOT / "research" / "downside_is.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/downside_is.json", flush=True)


if __name__ == "__main__":
    main()
