"""IS-ONLY screens of the shortlisted event sleeves (wave 1).

  COILSPRING  mild spread-widening reversion, GRANITE-complement band
  FLOWBACK-S  volume-confirmed fire-sale reversal w/ sibling news control
  ENDGAME     pull-to-par cheapening after the mat 2->1 index-exclusion flip

All rules point-in-time; every sleeve's control replays the SAME filter chain
with random timing. Design window 2003-2015 ONLY (the OOS wall).

  python corps/research/sleeves_events.py [coil flow end]
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")
RNG = np.random.default_rng(17)


def load_global():
    with open(ROOT / "data" / "cache_global.pkl", "rb") as f:
        return pickle.load(f)


def report(name, fills, ctl_rets, bonds, extra=None):
    s = e2.summarize(fills, control=ctl_rets if len(ctl_rets) else None)
    line = (f"  {name:12} n={s.get('n',0):5} win={s.get('win_rate',0)*100:3.0f}% "
            f"mean={s.get('mean_ret',0)*100:+6.2f}% hold={s.get('mean_hold',0):4.0f}d "
            f"excess={s.get('excess_vs_control',0)*100:+5.2f}% "
            f"p={s.get('excess_p_boot',1):.3f}")
    print(line, flush=True)
    if fills:
        r = e2.mtm_nav(bonds, fills)
        if r:
            ps = e2.perf_stats(*r)
            s["mtm"] = ps
            print(f"    MTM cagr={ps['cagr']*100:+.2f}% sharpe_m={ps['sharpe_m']} "
                  f"maxdd={ps['maxdd']*100:.1f}% trades/yr={len(fills)/13.0:.0f}", flush=True)
    if extra:
        s.update(extra)
    return s


# ------------------------------------------------------------------ COILSPRING

def run_coilspring(bonds, glob, lo, hi, widen_extra=0.0):
    xs_days, xs_med = glob["xs_days"], glob["csmat_med"]
    fills, ctl = [], []
    for six, b in bonds.items():
        day = b["day"]; cs = b["cs"]; mid = b["mid"]; mat = b["mat"]
        cs60 = b.get("cs60"); med60 = b.get("med60")
        if cs60 is None or len(day) < 20:
            continue
        thr = np.maximum(0.0075, 0.004 + 0.004 / np.maximum(mat, 0.5)) + widen_extra
        widened = (cs - cs60) >= thr
        base = (b["elig"] & (mat >= 1) & (mat <= 5) & (cs >= 0.015) & (cs <= 0.05)
                & (mid <= 102) & np.isfinite(cs60)
                & ~((b["s_px"] - med60) <= -3.0))          # NOT a GRANITE trigger
        # persistence: widened on >=2 distinct print days in [t-3, t]
        pers = np.zeros(len(day), bool)
        wi = np.flatnonzero(widened)
        for i in wi:
            j0 = np.searchsorted(day, day[i] - 3)
            if widened[j0:i + 1].sum() >= 2:
                pers[i] = True
        # cross-sectional: cs/mat above daily median
        xi = np.clip(np.searchsorted(xs_days, day, side="right") - 1, 0, len(xs_med) - 1)
        xs_ok = (cs / np.maximum(mat, 0.5)) > xs_med[xi]
        sig = base & widened & pers & xs_ok
        idx = np.flatnonzero(sig)
        if not len(idx):
            continue
        smask = ~np.isnan(b["s_px"]); s_day = day[smask]; s_px = b["s_px"][smask]
        pmask = ~np.isnan(b["p_px"]); p_day = day[pmask]; p_px = b["p_px"][pmask]
        cand = np.flatnonzero(base & np.isfinite(cs60))    # control pool: same gates, no timing
        last_exit = -10**9
        for i in idx:
            f = _coil_fill(b, i, day, cs, cs60, mat, s_day, s_px, p_day, p_px, lo, hi, last_exit)
            if f is None:
                continue
            fills.append(f); last_exit = f.exit_day
            for _ in range(10):                             # matched random entries
                j = int(RNG.choice(cand))
                cf = _coil_fill(b, j, day, cs, cs60, mat, s_day, s_px, p_day, p_px, lo, hi, -10**9)
                if cf is not None:
                    ctl.append(cf.ret)
    return fills, np.array(ctl)


def _coil_fill(b, i, day, cs, cs60, mat, s_day, s_px, p_day, p_px, lo, hi, last_exit):
    sd = day[i]
    j = np.searchsorted(s_day, sd, side="right")
    if j >= len(s_day) or s_day[j] - sd > 7:
        return None
    ed = int(s_day[j]); ep = float(s_px[j])
    if ed < lo or ed > hi or ed - last_exit < 30:
        return None
    # exit: min 21d; first bid once cs (at that day) <= cs60(now)+25bp; mat<0.75; 240d stop
    k0 = np.searchsorted(p_day, ed + 21, side="left")
    xd = xp = None
    for k in range(k0, len(p_day)):
        if p_day[k] > ed + 240:
            break
        di = np.searchsorted(day, p_day[k], side="right") - 1
        if di >= 0 and (
            (np.isfinite(cs[di]) and np.isfinite(cs60[di]) and cs[di] <= cs60[di] + 0.0025)
            or mat[di] < 0.75):
            xd, xp = int(p_day[k]), float(p_px[k]); break
    if xd is None:
        kk = np.searchsorted(p_day, ed + 240, side="right") - 1
        if kk < 0 or p_day[kk] <= ed:
            return None
        xd, xp = int(min(ed + 240, int(p_day[kk]) if p_day[kk] > ed + 240 else ed + 240)), float(p_px[kk])
        xd = ed + 240
    return e2.Fill(b["_six"], ed, ep, xd, xp, b["coupon"], False)


# ------------------------------------------------------------------ FLOWBACK-S

def build_issuer_map(bonds):
    m = {}
    for six in bonds:
        m.setdefault(six[:6], []).append(six)
    return m


def run_flowback(bonds, imap, lo, hi, vol_mult=4.0, drop=1.25):
    fills, ctl = [], []
    QV_ABS = 1000.0   # ~$1mm par in qvolume units ($000s)
    for six, b in bonds.items():
        day = b["day"]; mid = b["mid"]; qv = b["qv"]
        med15 = b.get("med15"); qvmed = b.get("qvmed90"); spr60 = b.get("spr60")
        if med15 is None or len(day) < 20:
            continue
        base = (b["elig"] & (b["act90"] >= 15) & (b["mat"] >= 1) & (b["mat"] <= 7)
                & (b["cs"] >= 0.008) & (b["cs"] <= 0.05) & (mid >= 70)
                & np.isfinite(spr60) & (spr60 <= 1.0) & np.isfinite(med15))
        vol_ok = (np.nan_to_num(qv) >= vol_mult * np.maximum(np.nan_to_num(qvmed), 1e-9)) \
                 & (np.nan_to_num(qv) >= QV_ABS)
        dropped = (med15 - mid) >= drop
        sig = base & vol_ok & dropped
        idx = np.flatnonzero(sig)
        if not len(idx):
            continue
        smask = ~np.isnan(b["s_px"]); s_day = day[smask]; s_px = b["s_px"][smask]
        pmask = ~np.isnan(b["p_px"]); p_day = day[pmask]; p_px = b["p_px"][pmask]
        cand = np.flatnonzero(base)
        last_exit = -10**9
        for i in idx:
            # sibling news check
            veto = False
            sibs = imap.get(six[:6], [])
            for p in sibs:
                if p == six:
                    continue
                bp = bonds[p]
                jj = np.searchsorted(bp["day"], day[i], side="right") - 1
                if jj < 0 or day[i] - bp["day"][jj] > 5:
                    continue
                m15 = bp.get("med15")
                if m15 is None or not np.isfinite(m15[jj]):
                    continue
                if (m15[jj] - bp["mid"][jj]) > 0.5:
                    veto = True; break
            if veto:
                continue
            f = _flow_fill(b, i, day, mid, med15, s_day, s_px, p_day, p_px, lo, hi, last_exit)
            if f is None:
                continue
            fills.append(f); last_exit = f.exit_day
            for _ in range(10):
                j = int(RNG.choice(cand))
                cf = _flow_fill(b, j, day, mid, med15, s_day, s_px, p_day, p_px, lo, hi, -10**9)
                if cf is not None:
                    ctl.append(cf.ret)
    return fills, np.array(ctl)


def _flow_fill(b, i, day, mid, med15, s_day, s_px, p_day, p_px, lo, hi, last_exit):
    sd = day[i]
    tgt = float(med15[i]) if np.isfinite(med15[i]) else None
    if tgt is None:
        return None
    j = np.searchsorted(s_day, sd, side="right")
    if j >= len(s_day) or s_day[j] - sd > 7:
        return None
    ed = int(s_day[j]); ep = float(s_px[j])
    if ep > mid[i] + 0.25:            # fill cap (applied to control identically)
        return None
    if ed < lo or ed > hi or ed - last_exit < 30:
        return None
    k0 = np.searchsorted(p_day, ed + 5, side="left")
    xd = xp = None
    for k in range(k0, len(p_day)):
        if p_day[k] > ed + 35:
            break
        di = np.searchsorted(day, p_day[k], side="right") - 1
        rec = di >= 0 and np.isfinite(mid[di]) and mid[di] >= tgt - 0.25
        prof = float(p_px[k]) >= ep + 1.5
        if rec or prof:
            xd, xp = int(p_day[k]), float(p_px[k]); break
    if xd is None:
        k = np.searchsorted(p_day, ed + 35, side="left")   # time stop: first bid after +35
        if k < len(p_day) and p_day[k] <= ed + 60:
            xd, xp = int(p_day[k]), float(p_px[k])
        else:
            kk = np.searchsorted(p_day, ed + 60, side="right") - 1
            if kk < 0 or p_day[kk] <= ed:
                return None
            xd, xp = ed + 60, float(p_px[kk])
    return e2.Fill(b["_six"], ed, ep, xd, xp, b["coupon"], False)


# -------------------------------------------------------------------- ENDGAME

def run_endgame(bonds, lo, hi):
    fills, ctl = [], []
    rf_cache = {}
    for six, b in bonds.items():
        day = b["day"]; mat = b["mat"]; ytw = b["ytw"]; mid = b["mid"]
        spr60 = b.get("spr60")
        # first 2->1 flip observed in panel
        flips = np.flatnonzero((mat[1:] == 1) & (mat[:-1] == 2)) + 1
        if not len(flips):
            continue
        t1i = int(flips[0]); t1 = int(day[t1i])
        # money-good gates evaluated at flip
        pre = (day >= t1 - 120) & (day <= t1 - 10) & np.isfinite(ytw)
        if pre.sum() < 8:
            continue
        rf = e2.load_rf(day) * 100.0
        base_spread = np.median((ytw - rf)[pre])
        smask = ~np.isnan(b["s_px"]); s_day = day[smask]; s_px = b["s_px"][smask]
        pmask = ~np.isnan(b["p_px"]); p_day = day[pmask]; p_px = b["p_px"][pmask]
        win = np.flatnonzero((day >= t1) & (day <= t1 + 60))
        good = lambda i: (mid[i] >= 92 and b["cs"][i] <= 0.035 and ytw[i] <= 8
                          and np.isfinite(spr60[i]) and spr60[i] <= 0.5
                          and not np.any(mid[np.searchsorted(day, day[i] - 250):i + 1] < 80))
        took = False
        for i in win:
            if not b["elig"][i] or not good(i):
                continue
            if (ytw[i] - rf[i]) < base_spread + 0.40:
                continue
            f = _end_fill(b, i, day, ytw, rf, base_spread, s_day, s_px, p_day, p_px, lo, hi)
            if f is not None:
                fills.append(f); took = True
                break
        if took:
            # control: random timing days in the SAME [t1, t1+60] window w/ gates
            pool = [i for i in win if b["elig"][i] and good(i)]
            for _ in range(10):
                if not pool:
                    break
                j = int(RNG.choice(pool))
                cf = _end_fill(b, j, day, ytw, rf, base_spread, s_day, s_px, p_day, p_px, lo, hi)
                if cf is not None:
                    ctl.append(cf.ret)
    return fills, np.array(ctl)


def _end_fill(b, i, day, ytw, rf, base_spread, s_day, s_px, p_day, p_px, lo, hi):
    sd = day[i]
    j = np.searchsorted(s_day, sd, side="right")
    if j >= len(s_day) or s_day[j] - sd > 7:
        return None
    ed = int(s_day[j]); ep = float(s_px[j])
    if ed < lo or ed > hi:
        return None
    k0 = np.searchsorted(p_day, ed + 30, side="left")
    xd = xp = None
    for k in range(k0, len(p_day)):
        if p_day[k] > ed + 240:
            break
        di = np.searchsorted(day, p_day[k], side="right") - 1
        j5 = np.searchsorted(day, p_day[k] - 5)
        seg = (ytw - rf)[j5:di + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) and np.median(seg) <= base_spread:
            xd, xp = int(p_day[k]), float(p_px[k]); break
        if p_day[k] >= ed + 150:
            xd, xp = int(p_day[k]), float(p_px[k]); break
    if xd is None:
        kk = np.searchsorted(p_day, ed + 270, side="right") - 1
        if kk < 0 or p_day[kk] <= ed:
            return None
        xd, xp = ed + 270, float(p_px[kk])
    return e2.Fill(b["_six"], ed, ep, xd, xp, b["coupon"], False)


def main():
    from combine import save_fills
    want = set(sys.argv[1:]) or {"coil", "flow", "end"}
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    glob = load_global()
    imap = build_issuer_map(bonds)
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}
    if "coil" in want:
        print("\n[COILSPRING] IS 2003-2015 + widening monotonicity:", flush=True)
        f, c = run_coilspring(bonds, glob, IS_LO, IS_HI)
        out["coilspring"] = report("base", f, c, bonds)
        save_fills(f, ROOT / "research" / "fills_coilspring_is.json")
        for extra, tag in [(0.005, "+50bp deeper"), (0.010, "+100bp deeper")]:
            f2, c2 = run_coilspring(bonds, glob, IS_LO, IS_HI, widen_extra=extra)
            out[f"coilspring_{tag}"] = report(tag, f2, c2, bonds)
    if "flow" in want:
        print("\n[FLOWBACK-S] IS 2003-2015 + knob monotonicity:", flush=True)
        f, c = run_flowback(bonds, imap, IS_LO, IS_HI)
        out["flowback"] = report("base 4x/1.25", f, c, bonds)
        save_fills(f, ROOT / "research" / "fills_flowback_is.json")
        for vm, dr, tag in [(3.0, 1.0, "3x/1.0"), (6.0, 2.0, "6x/2.0")]:
            f2, c2 = run_flowback(bonds, imap, IS_LO, IS_HI, vol_mult=vm, drop=dr)
            out[f"flowback_{tag}"] = report(tag, f2, c2, bonds)
    if "end" in want:
        print("\n[ENDGAME] IS 2003-2015:", flush=True)
        f, c = run_endgame(bonds, IS_LO, IS_HI)
        out["endgame"] = report("base", f, c, bonds)
        save_fills(f, ROOT / "research" / "fills_endgame_is.json")
    p = ROOT / "research" / "sleeves_is.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
