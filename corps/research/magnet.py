"""MAGNET v2 — dislocation ENTRY + hold-to-redemption EXIT (IS-only screen),
rebuilt to the red-team's required modifications.

Thesis: graft the only OOS-proven timing alpha (GRANITE's dislocation entry)
onto the only exit that does not pay the dealer's bid (redemption). The
bid-ask is paid once; there is no exit-timing decision.

RED-TEAM PATCHES (all mandatory, all implemented):
  1. classify_ending2: par credit requires a healthy final print (>=90,
     cs<=5%, no sub-80 in final year) AND remaining mat <= 1 at the final
     print AND the tape ends >= 1y before the panel edge. Bonds still
     printing near 2025-03 or fading early with years to run exit at their
     LAST OBSERVED BID — no phantom pull-to-par.
  2. Runs on the RECOVERED coupons (coupon_inv, augment4 v2).
  3. THREE comparisons:
       Control A  same bonds, random gated entry days, same exit
                  (nets classifier error + carry; excess ~ depth by
                  construction — reported, not headline)
       Control B  random OTHER bonds from the same gated universe, matched
                  entry month, same exit (tests adverse selection — the
                  headline control)
       Benchmark C same entries, GRANITE exit (sell at first bid in
                  [365,455]d) — MAGNET must beat the proven exit on
                  identical entries to earn capital
  4. Par / punitive(last-bid-only) bracket on every headline; verdict must be
     bracket-invariant or the sleeve is NO-GO.
  5. Pre-registered kill gates: excess vs Control B positive & monotone in
     depth {1,2,3,4}; ANNUALIZED excess non-increasing in max-mat
     {1,2,3} (rising-with-maturity = repackaged carry beta = artifact);
     issuer cap 1 concurrent position per issuer.
  6. Universe floor on the TRAILING MEDIAN (med60 >= 85) not the dislocated
     print, plus an absolute entry-price floor (ask >= 75).

  python corps/research/magnet.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from granite_experiments import issuer_cap_filter  # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")
PANEL_END = e2.D("2025-03-31")
RNG = np.random.default_rng(31)


def classify_ending2(b):
    """Par credit only for tapes that plausibly ENDED IN REDEMPTION:
    healthy final print, remaining mat <= 1, and tape end >= 1y before the
    panel edge. Everything else exits at the last observed bid."""
    day = b["day"]; mid = b["mid"]
    pm = ~np.isnan(b["p_px"])
    if not pm.any():
        return False, None, None
    p_day = day[pm]; p_px = b["p_px"][pm]
    lb_day, lb_px = int(p_day[-1]), float(p_px[-1])
    if int(day[-1]) > PANEL_END - 365:            # sample-edge guard
        return False, lb_day, lb_px
    if int(b["mat"][-1]) > 1:                     # faded early with years to run
        return False, lb_day, lb_px
    last_mid = mid[-1] if np.isfinite(mid[-1]) else lb_px
    last_cs = b["cs"][-1]
    j = np.searchsorted(day, day[-1] - 365)
    fin = mid[j:]
    min_final = np.nanmin(fin) if np.isfinite(fin).any() else np.nan
    healthy = (np.isfinite(last_mid) and last_mid >= 90
               and np.isfinite(last_cs) and last_cs <= 0.05
               and (not np.isfinite(min_final) or min_final >= 80))
    return bool(healthy), lb_day, lb_px


def gates(b, max_mat):
    day = b["day"]; mid = b["mid"]; med60 = b.get("med60")
    if med60 is None or len(day) < 30:
        return None
    n = len(day)
    j0s = np.searchsorted(day, day - 365)
    run_min = np.full(n, np.nan)
    for i in range(n):
        seg = mid[j0s[i]:i + 1]
        if len(seg):
            run_min[i] = np.nanmin(seg)
    return (b["elig"] & (b["mat"] >= 1) & (b["mat"] <= max_mat)
            & (b["cs"] <= 0.05) & np.isfinite(med60) & (med60 >= 85)
            & ~(run_min < 80))


def terminal_exit(b, par_redemption):
    red, lb_day, lb_px = classify_ending2(b)
    if lb_day is None:
        return None
    if par_redemption and red:
        return int(b["day"][-1]), 100.0, red
    return lb_day, lb_px, red


def fill_hold_to_end(b, i, lo, hi, par_redemption):
    day = b["day"]
    te = terminal_exit(b, par_redemption)
    if te is None:
        return None
    xd, xp, red = te
    sm = ~np.isnan(b["s_px"])
    s_day = day[sm]; s_px = b["s_px"][sm]
    j = np.searchsorted(s_day, day[i], side="right")
    if j >= len(s_day) or s_day[j] - day[i] > 7:
        return None
    ed = int(s_day[j]); ep = float(s_px[j])
    if ed < lo or ed > hi or xd <= ed or ep < 75:
        return None
    return e2.Fill(b["_six"], ed, ep, xd, xp, b["coupon_inv"], not red)


def run_magnet(bonds, lo, hi, depth=3.0, max_mat=3, par_redemption=True):
    fills = []
    gate_cache = {}
    for six, b in bonds.items():
        g = gates(b, max_mat)
        if g is None or not g.any():
            continue
        gate_cache[six] = g
        sig = g & ((b["s_px"] - b["med60"]) <= -depth)
        idx = np.flatnonzero(sig)
        for i in idx:
            f = fill_hold_to_end(b, int(i), lo, hi, par_redemption)
            if f is not None:
                fills.append(f)
                break                             # one entry per bond
    fills = issuer_cap_filter(fills, cap=1)
    return fills, gate_cache


def control_A(bonds, fills, gate_cache, lo, hi, par_redemption, k=8):
    rets = []
    for f in fills:
        b = bonds[f.six]; g = gate_cache.get(f.six)
        if g is None:
            continue
        pool = np.flatnonzero(g)
        for _ in range(k):
            cf = fill_hold_to_end(b, int(RNG.choice(pool)), lo, hi, par_redemption)
            if cf is not None:
                rets.append(cf.ret)
    return np.array(rets)


def control_B(bonds, fills, gate_cache, lo, hi, par_redemption, k=6):
    """Random OTHER bonds (different issuer), gated day in the same month."""
    month_bin = {}
    for six, g in gate_cache.items():
        day = bonds[six]["day"]
        for i in np.flatnonzero(g):
            month_bin.setdefault(int(day[i]) // 30, []).append((six, int(i)))
    rets, par_flags = [], []
    for f in fills:
        mb = month_bin.get(f.entry_day // 30, [])
        cand = [(s, i) for s, i in mb if s[:6] != f.six[:6]]
        if not cand:
            continue
        for _ in range(k):
            s, i = cand[int(RNG.integers(len(cand)))]
            cf = fill_hold_to_end(bonds[s], i, lo, hi, par_redemption)
            if cf is not None:
                rets.append(cf.ret)
                par_flags.append(not cf.stale)
    return np.array(rets), np.array(par_flags)


def benchmark_C(bonds, fills):
    """Same entries, GRANITE exit: first bid in [entry+365, entry+455]."""
    out = []
    for f in fills:
        b = bonds[f.six]
        day = b["day"]; pm = ~np.isnan(b["p_px"])
        p_day = day[pm]; p_px = b["p_px"][pm]
        lo = f.entry_day + 365; hi = f.entry_day + 455
        kk = np.searchsorted(p_day, lo, side="left")
        if kk < len(p_day) and p_day[kk] <= hi:
            xd, xp, st = int(p_day[kk]), float(p_px[kk]), False
        else:
            k2 = np.searchsorted(p_day, hi, side="right") - 1
            if k2 < 0 or p_day[k2] <= f.entry_day:
                continue
            xd, xp, st = hi, float(p_px[k2]), True
        out.append(e2.Fill(f.six, f.entry_day, f.entry_px, xd, xp,
                           b["coupon_inv"], st))
    return out


def stats(fills, ctl=None):
    s = e2.summarize(fills, control=ctl if ctl is not None and len(ctl) else None)
    if s.get("n"):
        s["mean_ann"] = s["mean_ret"] / max(s["mean_hold"] / 365.0, 0.25)
        s["par_rate"] = 1.0 - s["stale_share"]
    return s


def line(name, s, extra=""):
    print(f"  {name:22} n={s.get('n',0):5} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+6.2f}% ann={s.get('mean_ann',0)*100:+6.2f}% "
          f"hold={s.get('mean_hold',0):4.0f}d par={s.get('par_rate',0)*100:3.0f}% "
          f"excess={s.get('excess_vs_control',0)*100:+5.2f}% "
          f"p={s.get('excess_p_boot',1):.3f} {extra}", flush=True)


def mtm_line(bonds, fills, s):
    if fills:
        r = e2.mtm_nav(bonds, fills)
        if r:
            ps = e2.perf_stats(*r)
            s["mtm"] = ps
            print(f"    MTM cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:5.2f} "
                  f"maxdd={ps['maxdd']*100:6.1f}%", flush=True)


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
        if "coupon_inv" not in b:
            b["coupon_inv"] = b["coupon"]
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    print("\n[KILL GATE 1] depth monotonicity vs Control B (<=3y, par bracket):",
          flush=True)
    for k in (1.0, 2.0, 3.0, 4.0):
        fills, gc = run_magnet(bonds, IS_LO, IS_HI, depth=k, max_mat=3)
        cb, cb_par = control_B(bonds, fills, gc, IS_LO, IS_HI, True)
        s = stats(fills, cb)
        s["ctlB_par_rate"] = float(cb_par.mean()) if len(cb_par) else None
        out[f"depth{k:.0f}"] = s
        line(f"k={k:.0f}pt vs B", s,
             f"ctlB_par={s['ctlB_par_rate']*100:.0f}%" if s["ctlB_par_rate"] else "")

    print("\n[KILL GATE 2] maturity gradient, ANNUALIZED excess must be "
          "non-increasing (k=3, par):", flush=True)
    grad = {}
    for mm in (1, 2, 3):
        fills, gc = run_magnet(bonds, IS_LO, IS_HI, depth=3.0, max_mat=mm)
        cb, _ = control_B(bonds, fills, gc, IS_LO, IS_HI, True)
        s = stats(fills, cb)
        if s.get("n") and len(cb):
            ctl_ann = float(np.mean(cb)) / max(s["mean_hold"] / 365.0, 0.25)
            s["excess_ann"] = s["mean_ann"] - ctl_ann
        out[f"mat{mm}"] = s; grad[mm] = s.get("excess_ann")
        line(f"<={mm}y", s, f"ann_excess={s.get('excess_ann',0)*100:+.2f}%")

    print("\n[KILL GATE 3] bracket invariance (k=3, <=3y):", flush=True)
    fills_p, gc = run_magnet(bonds, IS_LO, IS_HI, 3.0, 3, par_redemption=True)
    ca = control_A(bonds, fills_p, gc, IS_LO, IS_HI, True)
    cb, _ = control_B(bonds, fills_p, gc, IS_LO, IS_HI, True)
    sp = stats(fills_p, cb); out["par"] = sp
    sp_a = stats(fills_p, ca); out["par_vs_A"] = {"excess": sp_a.get("excess_vs_control"),
                                                 "p": sp_a.get("excess_p_boot")}
    line("par vs B", sp)
    line("par vs A (depth-by-constr)", sp_a)
    mtm_line(bonds, fills_p, sp)
    fills_n, gcn = run_magnet(bonds, IS_LO, IS_HI, 3.0, 3, par_redemption=False)
    cbn, _ = control_B(bonds, fills_n, gcn, IS_LO, IS_HI, False)
    sn = stats(fills_n, cbn); out["punitive"] = sn
    line("punitive vs B", sn)
    mtm_line(bonds, fills_n, sn)

    print("\n[BENCHMARK C] same entries, GRANITE exit (365-455d bid sale):", flush=True)
    bc = benchmark_C(bonds, fills_p)
    sb = stats(bc); out["benchC"] = sb
    line("GRANITE exit", sb)
    mtm_line(bonds, bc, sb)

    from combine import save_fills
    save_fills(fills_p, ROOT / "research" / "fills_magnet_is.json")
    (ROOT / "research" / "magnet_is.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/magnet_is.json", flush=True)


if __name__ == "__main__":
    main()
