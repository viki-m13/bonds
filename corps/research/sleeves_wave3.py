"""IS-ONLY screens of the remaining test-first sleeves (wave 3).

  TWINS-R     bond wide vs same-issuer maturity-matched siblings (change-based)
  DEBUT       new-issue concession vs maturity-matched sibling yield
  ANGELFALL-M fallen-angel proxy: idiosyncratic cs crossing of 3% w/ forced flow

Design window 2003-2015 ONLY. Controls replay each sleeve's full filter chain
with random timing where the spec defines a timing window.

  python corps/research/sleeves_wave3.py [twins debut angel]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from sleeves_events import report, build_issuer_map  # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")
RNG = np.random.default_rng(23)


def _prints(b):
    day = b["day"]
    sm = ~np.isnan(b["s_px"]); pm = ~np.isnan(b["p_px"])
    return day[sm], b["s_px"][sm], day[pm], b["p_px"][pm]


def _exit_first_bid(p_day, p_px, lo, hi, stale_hi):
    k = np.searchsorted(p_day, lo, side="left")
    if k < len(p_day) and p_day[k] <= hi:
        return int(p_day[k]), float(p_px[k]), False
    kk = np.searchsorted(p_day, stale_hi, side="right") - 1
    if kk >= 0:
        return int(stale_hi), float(p_px[kk]), True
    return None


# -------------------------------------------------------------------- TWINS-R

def run_twins(bonds, imap, lo, hi):
    fills, ctl = [], []
    open_issuer = {}
    for iss, members in imap.items():
        if len(members) < 3:
            continue
        # build per-member last-known cs on a merged day grid (10d freshness)
        mem = [(six, bonds[six]) for six in members]
        for six, b in mem:
            day = b["day"]; cs = b["cs"]; mat = b["mat"]
            if len(day) < 30:
                continue
            gap = np.full(len(day), np.nan)
            nsib = np.zeros(len(day), np.int16)
            for i in range(len(day)):
                if not b["elig"][i] or not np.isfinite(cs[i]):
                    continue
                vals = []
                for psix, bp in mem:
                    if psix == six:
                        continue
                    j = np.searchsorted(bp["day"], day[i], side="right") - 1
                    if j < 0 or day[i] - bp["day"][j] > 10 or not bp["elig"][j]:
                        continue
                    if abs(int(bp["mat"][j]) - int(mat[i])) > 2:
                        continue
                    c = bp["cs"][j]
                    if np.isfinite(c):
                        vals.append(float(c))
                if len(vals) >= 2:
                    gap[i] = cs[i] - np.median(vals)
                    nsib[i] = len(vals)
            if not np.isfinite(gap).any():
                continue
            # trailing-60d median gap, shift-1, >=20 obs
            import pandas as pd
            idx = pd.to_datetime(day, unit="D")
            g60 = (pd.Series(gap, index=idx).rolling("60D", min_periods=20)
                   .median().shift(1).to_numpy())
            gstd = (pd.Series(gap, index=idx).rolling("120D", min_periods=20)
                    .std().shift(1).to_numpy())
            dev = gap - g60
            z = dev / np.where(gstd > 1e-6, gstd, np.nan)
            base = (b["elig"] & (mat >= 1) & (mat <= 7) & (b["cs"] >= 0.01)
                    & (b["cs"] <= 0.06) & (b["mid"] >= 70) & np.isfinite(gap)
                    & np.isfinite(g60))
            sig = base & (dev >= 0.0075) & (z >= 2.0)
            idx_sig = np.flatnonzero(sig)
            if not len(idx_sig):
                continue
            s_day, s_px, p_day, p_px = _prints(b)
            for i in idx_sig:
                # persistence: dev >= 50bp on >=3 print days in [t-10, t]
                j0 = np.searchsorted(day, day[i] - 10)
                if np.sum(np.nan_to_num(dev[j0:i + 1]) >= 0.005) < 3:
                    continue
                t = day[i]
                held = open_issuer.get(iss, -1)
                if held >= t:
                    continue
                j = np.searchsorted(s_day, t, side="right")
                if j >= len(s_day) or s_day[j] - t > 7:
                    continue
                ed = int(s_day[j]); ep = float(s_px[j])
                if ed < lo or ed > hi:
                    continue
                # exit: min 10d, first bid once gap - frozen entry median <= 25bp; 180d stop
                frozen = g60[i]
                xd = xp = None
                k0 = np.searchsorted(p_day, ed + 10, side="left")
                for k in range(k0, len(p_day)):
                    if p_day[k] > ed + 180:
                        break
                    di = np.searchsorted(day, p_day[k], side="right") - 1
                    if di >= 0 and np.isfinite(gap[di]) and gap[di] - frozen <= 0.0025:
                        xd, xp = int(p_day[k]), float(p_px[k]); break
                if xd is None:
                    r = _exit_first_bid(p_day, p_px, ed + 180, ed + 210, ed + 210)
                    if r is None:
                        continue
                    xd, xp, _ = r
                fills.append(e2.Fill(six, ed, ep, xd, xp, b["coupon"], False))
                open_issuer[iss] = xd
                # control: random base days, same mechanics
                cand = np.flatnonzero(base)
                for _ in range(8):
                    ii = int(RNG.choice(cand))
                    tt = day[ii]
                    jj = np.searchsorted(s_day, tt, side="right")
                    if jj >= len(s_day) or s_day[jj] - tt > 7:
                        continue
                    ced = int(s_day[jj]); cep = float(s_px[jj])
                    if ced < lo or ced > hi:
                        continue
                    frozen_c = g60[ii]
                    cxd = cxp = None
                    kk0 = np.searchsorted(p_day, ced + 10, side="left")
                    for k in range(kk0, len(p_day)):
                        if p_day[k] > ced + 180:
                            break
                        di = np.searchsorted(day, p_day[k], side="right") - 1
                        if di >= 0 and np.isfinite(gap[di]) and np.isfinite(frozen_c) \
                                and gap[di] - frozen_c <= 0.0025:
                            cxd, cxp = int(p_day[k]), float(p_px[k]); break
                    if cxd is None:
                        r = _exit_first_bid(p_day, p_px, ced + 180, ced + 210, ced + 210)
                        if r is None:
                            continue
                        cxd, cxp, _ = r
                    acc = b["coupon"] / 100 / 365 * (cxd - ced) * 100
                    ctl.append((cxp - cep + acc) / cep)
    return fills, np.array(ctl)


# ---------------------------------------------------------------------- DEBUT

def run_debut(bonds, imap, lo, hi, sample_start=None):
    if sample_start is None:
        sample_start = e2.D("2002-07-01")
    fills, ctl = [], []
    for six, b in bonds.items():
        day = b["day"]
        if len(day) < 10:
            continue
        t0 = int(day[0])
        if t0 < sample_start + 200 or t0 < e2.D("2007-01-01"):
            continue        # artifact quarantine: calibrate on debuts after 2007
        mid0 = b["mid"][0]
        if not (np.isfinite(mid0) and 97 <= mid0 <= 103):
            continue
        if int(b["mat"][0]) < 2:
            continue
        # >=3 print days in [t0, t0+5] and >= $25mm cumulative volume proxy
        j5 = np.searchsorted(day, t0 + 5, side="right")
        if j5 < 3:
            continue
        if np.nansum(b["qv"][:j5]) < 25.0:      # >=$25mm cumulative ($MM units)
            continue
        s_day, s_px, p_day, p_px = _prints(b)
        # concession gate on t in [t0+3, t0+15]
        cand_win = np.flatnonzero((day >= t0 + 3) & (day <= t0 + 15))
        took = None
        pool = []
        for i in cand_win:
            y = b["ytw"][i]
            if not np.isfinite(y):
                continue
            # maturity-matched sibling 30d median ytw
            sib = []
            for p in imap.get(six[:6], []):
                if p == six:
                    continue
                bp = bonds[p]
                jj1 = np.searchsorted(bp["day"], day[i], side="right")
                jj0 = np.searchsorted(bp["day"], day[i] - 30)
                if jj1 - jj0 < 3:
                    continue
                mm = bp["mat"][jj0:jj1]; yy = bp["ytw"][jj0:jj1]
                ok = np.abs(mm.astype(int) - int(b["mat"][i])) <= 2
                yy = yy[ok & np.isfinite(yy)]
                if len(yy) >= 3:
                    sib.append(float(np.median(yy)))
            if not sib:
                continue
            pool.append(i)
            if y >= np.median(sib) + 0.20 and took is None:
                took = i
        if took is None:
            continue
        def fill_at(i):
            t = day[i]
            j = np.searchsorted(s_day, t, side="right")
            if j >= len(s_day) or s_day[j] - t > 7:
                return None
            ed = int(s_day[j]); ep = float(s_px[j])
            if ed < lo or ed > hi:
                return None
            r = _exit_first_bid(p_day, p_px, ed + 45, ed + 90, ed + 120)
            if r is None:
                return None
            xd, xp, st = r
            return e2.Fill(six, ed, ep, xd, xp, b["coupon"], st)
        f = fill_at(took)
        if f is None:
            continue
        fills.append(f)
        for _ in range(10):     # control: random timing INSIDE the debut window
            if not pool:
                break
            cf = fill_at(int(RNG.choice(pool)))
            if cf is not None:
                ctl.append(cf.ret)
    return fills, np.array(ctl)


# ----------------------------------------------------------------- ANGELFALL-M

def run_angel(bonds, lo, hi, use_cap=True):
    import pandas as pd
    # first pass: detect events + build daily crossing-count for the cap
    events = {}
    day_cross = {}
    for six, b in bonds.items():
        day = b["day"]; cs = b["cs"]
        if len(day) < 60:
            continue
        idx = pd.to_datetime(day, unit="D")
        cs20 = pd.Series(cs, index=idx).rolling("20D", min_periods=4).median().to_numpy()
        cs90 = pd.Series(cs, index=idx).rolling("90D", min_periods=8).median().to_numpy()
        crossed = cs20 > 0.03
        first = None
        for i in np.flatnonzero(crossed):
            j60 = np.searchsorted(day, day[i] - 60, side="right") - 1
            if j60 < 0 or not np.isfinite(cs90[j60]) or cs90[j60] > 0.02:
                continue
            if not b["elig"][i]:
                continue
            first = i
            break
        if first is None:
            continue
        e = int(day[first])
        # forced-flow confirmation
        j0 = np.searchsorted(day, e - 5); j1 = np.searchsorted(day, e + 15, side="right")
        jp0 = np.searchsorted(day, e - 26); jp1 = np.searchsorted(day, e - 5)
        v_ev = np.nansum(b["qv"][j0:j1]); v_pre = np.nanmean(b["qv"][jp0:jp1]) * 21 if jp1 > jp0 else np.nan
        if not (np.isfinite(v_pre) and v_ev >= 4 * max(v_pre, 1e-9)):
            continue
        m60 = np.nanmedian(b["mid"][np.searchsorted(day, e - 60):np.searchsorted(day, e - 10)])
        if not (np.isfinite(m60) and np.isfinite(b["mid"][first]) and b["mid"][first] <= m60 - 4):
            continue
        events[six] = (first, e)
        day_cross[e] = day_cross.get(e, 0) + 1
    # cap: idiosyncratic only — skip events on days when >=1.5% of universe crossing
    n_gated = 12000   # approx gated universe size; cap threshold
    cap_thresh = 0.015 * n_gated
    fills, ctl = [], []
    for six, (first, e) in events.items():
        if use_cap:
            wave = sum(c for d, c in day_cross.items() if abs(d - e) <= 10)
            if wave > cap_thresh:
                continue
        b = bonds[six]
        day = b["day"]; cs = b["cs"]
        s_day, s_px, p_day, p_px = _prints(b)
        # entry t in [e+10, e+40]: 5d volume < 1.5x pre-event avg, cs<=5.5%, mid>=70
        jp0 = np.searchsorted(day, e - 26); jp1 = np.searchsorted(day, e - 5)
        v_pre_d = np.nanmean(b["qv"][jp0:jp1]) if jp1 > jp0 else np.nan
        win = np.flatnonzero((day >= e + 10) & (day <= e + 40))
        pool = []
        took = None
        for i in win:
            if not (2 <= int(b["mat"][i]) <= 10) or not b["elig"][i]:
                continue
            if not (np.isfinite(cs[i]) and cs[i] <= 0.055 and b["mid"][i] >= 70):
                continue
            j5 = np.searchsorted(day, day[i] - 5)
            v5 = np.nanmean(b["qv"][j5:i + 1])
            if np.isfinite(v_pre_d) and v5 >= 1.5 * v_pre_d:
                continue
            pool.append(i)
            if took is None:
                took = i
        if took is None:
            continue
        def fill_at(i):
            t = day[i]
            j = np.searchsorted(s_day, t, side="right")
            if j >= len(s_day) or s_day[j] - t > 7:
                return None
            ed = int(s_day[j]); ep = float(s_px[j])
            if ed < lo or ed > hi:
                return None
            # early exit from +45 if 10d median cs <= 3%
            xd = xp = None
            k0 = np.searchsorted(p_day, ed + 45, side="left")
            for k in range(k0, len(p_day)):
                if p_day[k] > ed + 240:
                    break
                di = np.searchsorted(day, p_day[k], side="right") - 1
                j10 = np.searchsorted(day, p_day[k] - 10)
                seg = cs[j10:di + 1]; seg = seg[np.isfinite(seg)]
                early = len(seg) and np.median(seg) <= 0.03
                if early or p_day[k] >= ed + 120:
                    xd, xp = int(p_day[k]), float(p_px[k]); break
            if xd is None:
                r = _exit_first_bid(p_day, p_px, ed + 120, ed + 240, ed + 300)
                if r is None:
                    return None
                xd, xp, _ = r
            return e2.Fill(six, ed, ep, xd, xp, b["coupon"], False)
        f = fill_at(took)
        if f is None:
            continue
        fills.append(f)
        for _ in range(10):
            cf = fill_at(int(RNG.choice(pool)))
            if cf is not None:
                ctl.append(cf.ret)
    return fills, np.array(ctl)


def main():
    from combine import save_fills
    want = set(sys.argv[1:]) or {"twins", "debut", "angel"}
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    imap = build_issuer_map(bonds)
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}
    if "twins" in want:
        print("\n[TWINS-R] IS 2003-2015:", flush=True)
        f, c = run_twins(bonds, imap, IS_LO, IS_HI)
        out["twins_r"] = report("base", f, c, bonds)
        save_fills(f, ROOT / "research" / "fills_twins_is.json")
    if "debut" in want:
        print("\n[DEBUT] IS 2007-2015 (calibration quarantine):", flush=True)
        f, c = run_debut(bonds, imap, e2.D("2007-01-01"), IS_HI)
        out["debut"] = report("base", f, c, bonds)
        save_fills(f, ROOT / "research" / "fills_debut_is.json")
    if "angel" in want:
        print("\n[ANGELFALL-M] IS 2003-2015 (with and without wave cap):", flush=True)
        f, c = run_angel(bonds, IS_LO, IS_HI, use_cap=True)
        out["angelfall_cap"] = report("with cap", f, c, bonds)
        save_fills(f, ROOT / "research" / "fills_angelfall_is.json")
        f2, c2 = run_angel(bonds, IS_LO, IS_HI, use_cap=False)
        out["angelfall_nocap"] = report("no cap", f2, c2, bonds)
    p = ROOT / "research" / "sleeves_is.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update({k: v for k, v in out.items() if v})
    p.write_text(json.dumps(old, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
