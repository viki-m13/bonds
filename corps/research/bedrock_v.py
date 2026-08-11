"""BEDROCK Sleeve V — GRANITE-XL entries + the replication-crisis-surviving
gates. IN-SAMPLE ONLY (2003-2015). Pre-registered spec (BEDROCK_RESEARCH.md):

  G1 spread-value : signal-day log(cs) >= same-day maturity-bucket median
                    (bond cheap vs peers; cross-section known end of signal
                    day, entry is the NEXT ask print - no lookahead)
  G2 spread-change: cs widened over trailing ~20 prints (cs_i - cs_{i-20} > 0)
  G3 stress       : signal-day volume >= 2x trailing-90d mean volume
  G4 issuer-curve : bond's own dislocation (mid - med60) at least 2 pts deeper
                    than the median contemporaneous dislocation of >=2 sibling
                    CUSIP6 bonds (pass when <2 siblings - no information)

All returns at RECOVERED REAL COUPONS with the LAGGED-mid recovery exit and
depth weights (the audited honest GRANITE-XL conventions). Baseline = same
entries without the new gates. Kill gate for OOS admission: ALL-gates book
must beat baseline IS on BOTH Sharpe(m) and CAGR (paired, same conventions),
with per-trade excess vs the cap-matched control positive at p<0.01.

  python corps/research/bedrock_v.py
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
from granite_experiments import sig_disc, gate_mat5, issuer_cap_filter  # noqa: E402
from oos2 import limit_filter, limit_control  # noqa: E402
from combos import depth_of  # noqa: E402

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
RNG = np.random.default_rng(97)


def cl_fills(bonds, lo, hi):
    f = e2.run_events(bonds, sig_disc(3.0), min_hold=365, max_hold=MAXH,
                      date_lo=lo, date_hi=hi, extra_gate=gate_mat5)
    f = issuer_cap_filter(f, cap=1)
    return limit_filter(bonds, f, cap=0.25)


def real_coupons(bonds, fills):
    return [e2.Fill(f.six, f.entry_day, f.entry_px, f.exit_day, f.exit_px,
                    float(bonds[f.six].get("coupon_inv", f.coupon)), f.stale)
            for f in fills]


def exit_lagged(bonds, fills):
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
            di = np.searchsorted(day, p_day[k], side="left") - 1
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


def build_cs_median(bonds):
    """(day, mat-bucket) -> median log(cs) across bonds printing that day."""
    print("building cross-sectional spread medians ...", flush=True)
    days, mats, lcs = [], [], []
    for b in bonds.values():
        cs = b["cs"]
        ok = np.isfinite(cs) & (cs > 0)
        if not ok.any():
            continue
        days.append(b["day"][ok])
        mats.append(b["mat"][ok])
        lcs.append(np.log(cs[ok].astype(np.float64)))
    df = pd.DataFrame({"day": np.concatenate(days),
                       "mat": np.concatenate(mats),
                       "lcs": np.concatenate(lcs)})
    df["mb"] = df["mat"].clip(0, 10)
    med = df.groupby(["day", "mb"])["lcs"].median()
    print(f"  {len(med):,} (day,bucket) cells from {len(df):,} prints", flush=True)
    return med


def sig_row(b, f):
    return np.searchsorted(b["day"], f.entry_day, side="left") - 1


def gate_value(bonds, fills, med):
    kept = []
    for f in fills:
        b = bonds[f.six]
        i = sig_row(b, f)
        if i < 0 or not np.isfinite(b["cs"][i]) or b["cs"][i] <= 0:
            continue
        key = (int(b["day"][i]), int(np.clip(b["mat"][i], 0, 10)))
        m = med.get(key)
        if m is None or not np.isfinite(m):
            continue
        if np.log(float(b["cs"][i])) >= m:
            kept.append(f)
    return kept


def gate_schange(bonds, fills, lag=20):
    kept = []
    for f in fills:
        b = bonds[f.six]
        i = sig_row(b, f)
        j = i - lag
        if i < 0 or j < 0:
            continue
        c1, c0 = b["cs"][i], b["cs"][j]
        if np.isfinite(c1) and np.isfinite(c0) and c1 > c0:
            kept.append(f)
    return kept


def gate_stress(bonds, fills, k=2.0):
    kept = []
    for f in fills:
        b = bonds[f.six]
        i = sig_row(b, f)
        if i < 0:
            continue
        v, v90 = b["qv"][i], b["qv90"][i]
        if np.isfinite(v) and np.isfinite(v90) and v90 > 0 and v >= k * v90:
            kept.append(f)
    return kept


def gate_issuer_curve(bonds, issuers, fills, min_gap=2.0):
    kept = []
    for f in fills:
        b = bonds[f.six]
        i = sig_row(b, f)
        if i < 0 or not (np.isfinite(b["mid"][i]) and np.isfinite(b["med60"][i])):
            kept.append(f); continue
        own = float(b["mid"][i] - b["med60"][i])
        sd = int(b["day"][i])
        sibs = []
        for s6 in issuers.get(f.six[:6], []):
            if s6 == f.six:
                continue
            sb = bonds[s6]
            j = np.searchsorted(sb["day"], sd, side="right") - 1
            if j < 0 or sd - sb["day"][j] > 10:
                continue
            if np.isfinite(sb["mid"][j]) and np.isfinite(sb["med60"][j]):
                sibs.append(float(sb["mid"][j] - sb["med60"][j]))
        if len(sibs) < 2:
            kept.append(f)                      # no information -> pass
        elif own <= float(np.median(sibs)) - min_gap:
            kept.append(f)
    return kept


def book_stats(bonds, fills, label, lo, hi, ctl=True):
    fl = real_coupons(bonds, exit_lagged(bonds, fills))
    if not fl:
        print(f"  {label:24} n=0", flush=True)
        return {"n": 0}
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fl]
    days, nav, daily = e2.mtm_nav(bonds, fl, weights=w)
    ps = e2.perf_stats(days, nav, daily)
    r = np.array([f.ret for f in fl])
    out = {"n": len(fl), "mean": float(r.mean()), "win": float((r > 0).mean()),
           "hold": float(np.mean([f.hold for f in fl])), **ps}
    if ctl:
        c = limit_control(bonds, fl, lo, hi)
        if len(c):
            out["excess"] = float(r.mean() - c.mean())
            by = {}
            for f in fl:
                by.setdefault(f.six, []).append(f.ret)
            keys = list(by); boots = []
            for _ in range(2000):
                ks = RNG.choice(len(keys), size=len(keys), replace=True)
                sm = np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean()
                boots.append(sm - RNG.choice(c, size=len(c), replace=True).mean())
            out["excess_p"] = float((np.array(boots) <= 0).mean())
    print(f"  {label:24} n={out['n']:5} mean={out['mean']*100:+6.2f}% "
          f"win={out['win']*100:3.0f}% hold={out['hold']:4.0f}d "
          f"cagr={out['cagr']*100:+6.2f}% sharpe_m={out['sharpe_m']:5.2f} "
          f"dd={out['maxdd']*100:6.1f}%"
          + (f" excess={out.get('excess',float('nan'))*100:+5.2f}% p={out.get('excess_p',float('nan')):.3f}"
             if ctl and "excess" in out else ""), flush=True)
    return out


def limit_control_real(bonds, fills, lo, hi, cap=0.25, k=10):
    """oos2.limit_control with REAL coupons (coupon_inv) so the control's
    accrual convention matches the strategy leg (fixes the mismatch that
    biased the v1 excess column down)."""
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
        cp = float(b.get("coupon_inv", b["coupon"]))
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
            acc = cp / 100 / 365 * (xd - ed) * 100
            rets.append((xp - ep + acc) / ep)
    return np.array(rets)


def v2():
    """DISCLOSED second IS iteration (selection acknowledged): the two gates
    that individually beat baseline in the v1 ablations (G1 spread-value, G4
    issuer-curve), combined. Its own kill gate; one OOS look only if passed.
    The v1 4-gate stack remains REJECTED."""
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    base = cl_fills(bonds, *IS)
    out = {}
    print("\n[V2 IS] baseline vs G1+G4 (real-coupon control):", flush=True)

    def stats_rc(fills, label):
        fl = real_coupons(bonds, exit_lagged(bonds, fills))
        w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fl]
        days, nav, daily = e2.mtm_nav(bonds, fl, weights=w)
        ps = e2.perf_stats(days, nav, daily)
        r = np.array([f.ret for f in fl])
        c = limit_control_real(bonds, fl, *IS)
        by = {}
        for f in fl:
            by.setdefault(f.six, []).append(f.ret)
        keys = list(by); boots = []
        for _ in range(2000):
            ks = RNG.choice(len(keys), size=len(keys), replace=True)
            sm = np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean()
            boots.append(sm - RNG.choice(c, size=len(c), replace=True).mean())
        o = {"n": len(fl), "mean": float(r.mean()), "win": float((r > 0).mean()),
             "excess": float(r.mean() - c.mean()),
             "excess_p": float((np.array(boots) <= 0).mean()), **ps}
        print(f"  {label:22} n={o['n']:5} mean={o['mean']*100:+6.2f}% "
              f"cagr={o['cagr']*100:+6.2f}% sharpe_m={o['sharpe_m']:5.2f} "
              f"dd={o['maxdd']*100:6.1f}% excess={o['excess']*100:+5.2f}% p={o['excess_p']:.3f}", flush=True)
        return o

    out["baseline"] = stats_rc(base, "baseline")
    g14 = gate_issuer_curve(bonds, issuers, gate_value(bonds, base, med))
    out["v2_g1g4"] = stats_rc(g14, "G1+G4")
    ok = out["v2_g1g4"]["sharpe_m"] > out["baseline"]["sharpe_m"] \
        and out["v2_g1g4"]["cagr"] > out["baseline"]["cagr"] \
        and out["v2_g1g4"]["excess_p"] < 0.01
    out["admit_oos"] = bool(ok)
    print(f"\n[V2 KILL GATE] {'ADMIT to one-shot OOS' if ok else 'REJECT'}", flush=True)
    p = ROOT / "research" / "bedrock_v2_is.json"
    p.write_text(json.dumps(out, default=float))
    print(f"wrote {p}", flush=True)


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)

    med = build_cs_median(bonds)
    med = med.to_dict()

    base = cl_fills(bonds, *IS)
    print(f"\nCL entries IS: {len(base)}", flush=True)
    out = {}
    print("\n[IS 2003-2015] real coupons + lagged recovery exits:", flush=True)
    out["baseline"] = book_stats(bonds, base, "baseline (no new gates)", *IS)

    g1 = gate_value(bonds, base, med)
    out["g1_value"] = book_stats(bonds, g1, "+G1 spread-value", *IS)
    g2 = gate_schange(bonds, g1)
    out["g2_schange"] = book_stats(bonds, g2, "+G2 spread-change", *IS)
    g3 = gate_stress(bonds, g2)
    out["g3_stress"] = book_stats(bonds, g3, "+G3 stress", *IS)
    g4 = gate_issuer_curve(bonds, issuers, g3)
    out["g4_issuercurve"] = book_stats(bonds, g4, "+G4 issuer-curve (ALL)", *IS)

    # individual-gate ablations on the baseline (diagnostic, pre-registered)
    print("\n[ablations] each gate alone on baseline:", flush=True)
    out["only_value"] = book_stats(bonds, gate_value(bonds, base, med), "value only", *IS, ctl=False)
    out["only_schange"] = book_stats(bonds, gate_schange(bonds, base), "schange only", *IS, ctl=False)
    out["only_stress"] = book_stats(bonds, gate_stress(bonds, base), "stress only", *IS, ctl=False)
    out["only_issuer"] = book_stats(bonds, gate_issuer_curve(bonds, issuers, base), "issuer-curve only", *IS, ctl=False)

    ok = (out["g4_issuercurve"].get("sharpe_m") or 0) > (out["baseline"].get("sharpe_m") or 0) \
        and out["g4_issuercurve"].get("cagr", -9) > out["baseline"].get("cagr", -9) \
        and out["g4_issuercurve"].get("excess_p", 1) < 0.01
    out["admit_oos"] = bool(ok)
    print(f"\n[KILL GATE] ALL-gates beats baseline Sharpe AND CAGR, excess p<0.01: "
          f"{'ADMIT to OOS' if ok else 'REJECT'}", flush=True)

    p = ROOT / "research" / "bedrock_v_is.json"
    p.write_text(json.dumps(out, default=float))
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "v2":
        v2()
    else:
        main()
