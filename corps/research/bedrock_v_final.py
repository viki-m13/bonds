"""BEDROCK-V finishing pass (XL_AUDIT standard) — everything a live desk
needs before capital:

  [R] LIVE-PROTOCOL REPLAY: chronological admission, issuer capacity checked
      against ACTUAL open positions (lagged recovery exits), 30d bond
      cooldown from actual exit, limit filter BEFORE capacity (no phantom
      blocking), G1+G4 gates at signal time, real coupons. Both lockout
      variants: position-based (capacity freed at actual exit) and
      tight time-based (~13mo from entry).
  [S] SLIPPAGE GRID: h in {0, 1/8, 1/4, 1/2} pt on entry ask / exit bid of
      the pipeline book.
  [E] ERA DECOMPOSITION of the pipeline book (per-trade, entry vintage).
  [P] GATE PERTURBATIONS (robustness, not tuning — report monotonicity):
      G1 cutoff at cross-sectional {40th, 50th, 60th} percentile;
      G4 min_gap in {1, 2, 3} pts. Full-window pipeline book stats.

  python corps/research/bedrock_v_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bedrock_v import (cl_fills, real_coupons, exit_lagged, build_cs_median,  # noqa: E402
                       gate_value, gate_issuer_curve)
from combos import depth_of  # noqa: E402

MAXH = 455
FULL = (e2.D("2003-01-01"), e2.D("2025-03-31") - MAXH)
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)
ERAS = [("2004-2007", "2004-01-01", "2007-12-31"),
        ("2008-2009", "2008-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2021-2023", "2021-01-01", "2023-12-31")]


def mtm(bonds, fills, label):
    if not fills:
        print(f"  {label:30} n=0", flush=True)
        return {"n": 0}
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    days, nav, daily = e2.mtm_nav(bonds, fills, weights=w)
    ps = e2.perf_stats(days, nav, daily)
    r = np.array([f.ret for f in fills])
    ps.update({"n": len(fills), "mean": float(r.mean()), "win": float((r > 0).mean()),
               "hold": float(np.mean([f.hold for f in fills]))})
    print(f"  {label:30} n={ps['n']:5} mean={ps['mean']*100:+6.2f}% win={ps['win']*100:3.0f}% "
          f"cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:5.2f} "
          f"dd={ps['maxdd']*100:6.1f}%", flush=True)
    return ps


def gates_pass(bonds, issuers, med, six, sig_i, g1_off=0.0, g4_gap=2.0):
    """G1+G4 evaluated at a signal row (point-in-time)."""
    b = bonds[six]
    cs = b["cs"][sig_i]
    if not (np.isfinite(cs) and cs > 0):
        return False
    key = (int(b["day"][sig_i]), int(np.clip(b["mat"][sig_i], 0, 10)))
    m = med.get(key)
    if m is None or not np.isfinite(m) or np.log(float(cs)) < m + g1_off:
        return False
    if not (np.isfinite(b["mid"][sig_i]) and np.isfinite(b["med60"][sig_i])):
        return True
    own = float(b["mid"][sig_i] - b["med60"][sig_i])
    sd = int(b["day"][sig_i])
    sibs = []
    for s6 in issuers.get(six[:6], []):
        if s6 == six:
            continue
        sb = bonds[s6]
        j = np.searchsorted(sb["day"], sd, side="right") - 1
        if j < 0 or sd - sb["day"][j] > 10:
            continue
        if np.isfinite(sb["mid"][j]) and np.isfinite(sb["med60"][j]):
            sibs.append(float(sb["mid"][j] - sb["med60"][j]))
    if len(sibs) < 2:
        return True
    return own <= float(np.median(sibs)) - g4_gap


def live_replay(bonds, issuers, med, lo, hi, lock="position"):
    """Chronological live protocol with G1+G4 at signal time."""
    cands = []
    for six, b in bonds.items():
        m60 = b.get("med60")
        if m60 is None:
            continue
        gate = b["elig"] & (b["mat"] <= 5)
        sig = (b["s_px"] - m60) <= -3.0
        idx = np.flatnonzero(sig & gate)
        if not len(idx):
            continue
        day = b["day"]
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        seen = set()
        for i in idx:
            sd = day[i]
            j = np.searchsorted(s_day, sd, side="right")
            if j >= len(s_day) or s_day[j] - sd > 7:
                continue
            ed = int(s_day[j]); ep = float(s_px[j])
            if ed < lo or ed > hi or ed in seen:
                continue
            seen.add(ed)
            ii = np.searchsorted(day, ed, side="left") - 1
            if ii < 0 or not np.isfinite(b["mid"][ii]) or ep > b["mid"][ii] + 0.25:
                continue
            if not gates_pass(bonds, issuers, med, six, ii):
                continue
            cands.append((ed, six, ep))
    cands.sort()
    print(f"  gated candidates: {len(cands)}", flush=True)

    def exit_l(b, ed):
        day = b["day"]; mid = b["mid"]; med60 = b["med60"]
        i0 = np.searchsorted(day, ed, side="left") - 1
        tgt = float(med60[i0]) if i0 >= 0 and np.isfinite(med60[i0]) else None
        if tgt is None:
            return None
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        k0 = np.searchsorted(p_day, ed + 21, side="left")
        for k in range(k0, len(p_day)):
            if p_day[k] > ed + MAXH:
                break
            di = np.searchsorted(day, p_day[k], side="left") - 1
            if di >= 0 and np.isfinite(mid[di]) and mid[di] >= tgt:
                return int(p_day[k]), float(p_px[k]), False
        k2 = np.searchsorted(p_day, ed + MAXH, side="right") - 1
        if k2 < 0 or p_day[k2] <= ed:
            return None
        return ed + MAXH, float(p_px[k2]), True

    open_by_iss, last_exit_bond, fills = {}, {}, []
    for ed, six, ep in cands:
        if ed - last_exit_bond.get(six, -10**9) < 30:
            continue
        iss = six[:6]
        cur = [x for x in open_by_iss.get(iss, []) if x > ed]
        if len(cur) >= 1:
            open_by_iss[iss] = cur
            continue
        b = bonds[six]
        ex = exit_l(b, ed)
        if ex is None:
            open_by_iss[iss] = cur
            continue
        xd, xp, st = ex
        fills.append(e2.Fill(six, ed, ep, xd, xp,
                             float(b.get("coupon_inv", b["coupon"])), st))
        lockout = xd if lock == "position" else min(ed + 395, FULL[1] + MAXH)
        cur.append(lockout)
        open_by_iss[iss] = cur
        last_exit_bond[six] = xd if lock == "position" else lockout
    return fills


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    out = {}

    print("\n[PIPELINE reference] full window:", flush=True)
    base_full = cl_fills(bonds, *FULL)
    v_full = gate_issuer_curve(bonds, issuers, gate_value(bonds, base_full, med))
    pipe = real_coupons(bonds, exit_lagged(bonds, v_full))
    out["pipeline_full"] = mtm(bonds, pipe, "pipeline BEDROCK-V full")
    poos = [f for f in pipe if OOS[0] <= f.entry_day <= OOS[1]]
    out["pipeline_oos"] = mtm(bonds, poos, "pipeline BEDROCK-V OOS-slice")

    print("\n[R] live-protocol replays (full window):", flush=True)
    for lock in ("position", "tight"):
        fl = live_replay(bonds, issuers, med, *FULL, lock=lock)
        out[f"replay_{lock}_full"] = mtm(bonds, fl, f"replay {lock}-lock full")
        oos_fl = [f for f in fl if OOS[0] <= f.entry_day <= OOS[1]]
        out[f"replay_{lock}_oos"] = mtm(bonds, oos_fl, f"replay {lock}-lock OOS")

    print("\n[S] slippage grid on the pipeline book (full window):", flush=True)
    for h in (0.0, 0.125, 0.25, 0.5):
        fh = [e2.Fill(f.six, f.entry_day, f.entry_px + h, f.exit_day,
                      max(f.exit_px - h, 1.0), f.coupon, f.stale) for f in pipe]
        out[f"slip_{h}"] = mtm(bonds, fh, f"h={h:5.3f}")

    print("\n[E] era decomposition (pipeline, per-trade, entry vintage):", flush=True)
    out["era"] = {}
    for lab, lo, hi in ERAS:
        lo_d, hi_d = e2.D(lo), e2.D(hi)
        fl = [f for f in pipe if lo_d <= f.entry_day <= hi_d]
        if len(fl) < 15:
            continue
        r = np.array([f.ret for f in fl])
        out["era"][lab] = {"n": len(fl), "mean": float(r.mean()),
                           "win": float((r > 0).mean())}
        print(f"  {lab:10} n={len(fl):5} mean={r.mean()*100:+6.2f}% win={(r>0).mean()*100:3.0f}%", flush=True)

    print("\n[P] gate perturbations (full-window pipeline stats; monotonicity check):", flush=True)
    # G1 percentile shift: implemented as an offset on log-median; the
    # cross-sectional dispersion of log cs is ~0.6-0.8, so +/-0.15 in log
    # space approximates the 40th/60th percentile shift.
    out["perturb"] = {}
    for g1_off, g4_gap, tag in [(-0.15, 2.0, "G1@~40pct"), (0.0, 2.0, "G1@median (spec)"),
                                (+0.15, 2.0, "G1@~60pct"), (0.0, 1.0, "G4 gap=1"),
                                (0.0, 3.0, "G4 gap=3")]:
        kept = []
        for f in base_full:
            b = bonds[f.six]
            i = np.searchsorted(b["day"], f.entry_day, side="left") - 1
            if i >= 0 and gates_pass(bonds, issuers, med, f.six, i, g1_off, g4_gap):
                kept.append(f)
        fl = real_coupons(bonds, exit_lagged(bonds, kept))
        out["perturb"][tag] = mtm(bonds, fl, tag)

    p = ROOT / "research" / "bedrock_v_final.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
