"""BULWARK v2 — IS screen on the HONEST EXIT convention, plus the
pre-registered screen combinations. (BULWARK_RESEARCH.md §2 + §3 addendum.)

Convention change (DISCLOSED, made before the OOS look, from the audit in
bulwark_censor_audit.py): exit = first customer-bid print >=365d after entry;
else the LAST REAL bid strictly after entry, booked AT ITS OWN DATE with
coupon accrued only to that date; else (no bid at all after entry) the
position is booked as a credit loss at min(last mid after entry, 40). The
engine default instead dated such exits at entry+455 using a price from a
median 249 days earlier while accruing 455 days of coupon.

  python corps/research/bulwark_v2.py [oos]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from oos2 import limit_filter  # noqa: E402
from granite_experiments import issuer_cap_filter  # noqa: E402
from bulwark_v1 import (u_mask, build_xsec, sig_factory, post_filters,  # noqa: E402
                        real_coupons, IS, OOS, MAXH, RNG)
from bulwark_censor_audit import honest_exit  # noqa: E402

# --------------------------------------------------- control (honest exits)
def matched_control_honest(bonds, fills, lo, hi, k=10, recovery=40.0):
    """Random entry DAYS inside U on the same bonds, exited under the SAME
    honest convention as the strategy leg. Fixes the convention mismatch that
    made every v2 excess column negative (strategy on honest exits vs a
    control still using the engine's inflated stale-exit fallback)."""
    rets = []
    for f in fills:
        b = bonds[f.six]
        day = b["day"]
        pool = np.flatnonzero(u_mask(b))
        if not len(pool):
            continue
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        mm = ~np.isnan(b["mid"]); m_day = day[mm]; m_px = b["mid"][mm]
        if not len(s_day) or not len(p_day):
            continue
        cp = float(b.get("coupon_inv", b["coupon"]))
        for _ in range(k):
            i = int(RNG.choice(pool)); t = day[i]
            j = np.searchsorted(s_day, t, side="right")
            if j >= len(s_day) or s_day[j] - t > 7:
                continue
            ed = int(s_day[j]); ep = float(s_px[j])
            if ed < lo or ed > hi:
                continue
            ii = np.searchsorted(day, ed, side="left") - 1
            if ii < 0 or not np.isfinite(b["mid"][ii]) or ep > b["mid"][ii] + 0.25:
                continue
            kk = np.searchsorted(p_day, ed + 365, side="left")
            if kk < len(p_day) and p_day[kk] <= ed + MAXH:
                xd, xp = int(p_day[kk]), float(p_px[kk])
            else:
                msk = (p_day > ed) & (p_day <= ed + MAXH)
                if msk.any():
                    q = np.flatnonzero(msk)[-1]
                    xd, xp = int(p_day[q]), float(p_px[q])
                else:
                    sel = (m_day > ed) & (m_day <= ed + MAXH)
                    if sel.any():
                        xd, xp = int(m_day[sel][-1]), min(float(m_px[sel][-1]), recovery)
                    else:
                        xd, xp = ed + MAXH, recovery
            acc = cp / 100 / 365 * (xd - ed) * 100
            rets.append((xp - ep + acc) / ep)
    return np.array(rets)


def stats_h(bonds, lab, fills, label, lo, hi, base=None):
    """stats() with the honest-exit control."""
    if not fills:
        print(f"  {label:26} n=0", flush=True)
        return {"n": 0}
    r = np.array([f.ret for f in fills])
    days, nav, daily = e2.mtm_nav(bonds, fills)
    ps = e2.perf_stats(days, nav, daily)
    nb = sum(1 for f in fills
             if (lab.get(f.six) and lab[f.six][0]
                 and f.entry_day <= lab[f.six][1] <= f.exit_day + 30))
    o = {"n": len(fills), "mean": float(r.mean()), "win": float((r > 0).mean()),
         "hold": float(np.mean([f.hold for f in fills])),
         "bust_n": nb, "bust_rate": nb / len(fills),
         "loss20": float((r < -0.20).mean()),
         "worst_decile": float(np.sort(r)[:max(1, len(r) // 10)].mean()),
         "issuers": len({f.six[:6] for f in fills}), **ps}
    c = matched_control_honest(bonds, fills, lo, hi)
    if len(c):
        by = {}
        for f in fills:
            by.setdefault(f.six[:6], []).append(f.ret)
        keys = list(by); boots = []
        for _ in range(2000):
            ks = RNG.choice(len(keys), size=len(keys), replace=True)
            sm = np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean()
            boots.append(sm - RNG.choice(c, size=len(c), replace=True).mean())
        o["ctl"] = float(c.mean()); o["excess"] = float(r.mean() - c.mean())
        o["excess_p"] = float((np.array(boots) <= 0).mean())
    if base:
        o["gA"] = bool(o["bust_rate"] <= 0.25 * base["bust_rate"])
        o["gB"] = bool(o["mean"] >= base["mean"] and o["sharpe_m"] >= base["sharpe_m"])
        o["gC"] = bool(o.get("excess_p", 1) < 0.01)
        o["admit"] = bool(o["gA"] and o["gB"] and o["gC"])
    print(f"  {label:26} n={o['n']:5} iss={o['issuers']:4} mean={o['mean']*100:+6.2f}% "
          f"win={o['win']*100:3.0f}% bust={nb:3}({o['bust_rate']*100:4.1f}%) "
          f"l20={o['loss20']*100:4.1f}% cagr={o['cagr']*100:+6.2f}% "
          f"sh={o['sharpe_m']:5.2f} dd={o['maxdd']*100:6.1f}% "
          f"exc={o.get('excess',float('nan'))*100:+5.2f}% p={o.get('excess_p',float('nan')):.3f}"
          + (f" [{'ADMIT' if o.get('admit') else 'reject'}]" if base else ""), flush=True)
    return o


COMBOS = {
    "S1+S2": dict(s1=True, s2=True),
    "S2+S4": dict(s2=True, s4=True),
    "S1+S4": dict(s1=True, s4=True),
    "S4+S6": dict(s4=True, s6=True),
    "S1+S2+S6": dict(s1=True, s2=True, s6=True),
    "S2+S4+S6": dict(s2=True, s4=True, s6=True),
    "S1+S2+S4+S6": dict(s1=True, s2=True, s4=True, s6=True),
}


def pipeline(bonds, issuers, q97, amed, lo, hi, honest=True, **screens):
    f = e2.run_events(bonds, sig_factory(bonds, q97, amed,
                                         screens.get("s1", False),
                                         screens.get("s2", False),
                                         screens.get("s3", False),
                                         screens.get("s5", False)),
                      min_hold=365, max_hold=MAXH, date_lo=lo, date_hi=hi)
    if screens.get("s4") or screens.get("s6"):
        f = post_filters(bonds, issuers, f, screens.get("s4", False),
                         screens.get("s6", False))
    f = limit_filter(bonds, f, cap=0.25)
    f = issuer_cap_filter(f, cap=1)
    if honest:
        f, _, _ = honest_exit(bonds, f)
    return real_coupons(bonds, f)


def main():
    oos_mode = len(sys.argv) > 1 and sys.argv[1] == "oos"
    lo, hi = OOS if oos_mode else IS
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    ldf = pd.read_parquet(ROOT / "research" / "bulwark_labels.parquet")
    lab = {r.six: (bool(r.bust), int(r.last_day)) for r in ldf.itertuples()}
    q97, amed = build_xsec(bonds)
    out = {}
    tag = "OOS 2016-2024" if oos_mode else "IS 2003-2015"
    print(f"\n[{tag}] HONEST EXITS, cheap junk, equal weight, real coupons:",
          flush=True)
    base = stats_h(bonds, lab, pipeline(bonds, issuers, q97, amed, lo, hi),
                 "U unscreened (baseline)", lo, hi)
    out["baseline"] = base

    if oos_mode:
        specs = {"BULWARK-S4": dict(s4=True),
                 "BULWARK-Z": dict(s1=True, s2=True, s4=True, s6=True)}
        print("\n[ONE-SHOT OOS of the frozen specs (BULWARK-U = baseline above)]",
              flush=True)
        for name, sc in specs.items():
            out[name] = stats_h(bonds, lab,
                                pipeline(bonds, issuers, q97, amed, lo, hi, **sc),
                                name, lo, hi, base=base)
        p = ROOT / "research" / "bulwark_oos3.json"
    else:
        print("\n[single screens, honest exits]", flush=True)
        for k in ("s1", "s2", "s3", "s4", "s5", "s6"):
            out[k] = stats_h(bonds, lab,
                           pipeline(bonds, issuers, q97, amed, lo, hi, **{k: True}),
                           f"+{k.upper()} only", lo, hi, base=base)
        print("\n[pre-registered combinations]", flush=True)
        for name, sc in COMBOS.items():
            out[name] = stats_h(bonds, lab,
                              pipeline(bonds, issuers, q97, amed, lo, hi, **sc),
                              name, lo, hi, base=base)
        p = ROOT / "research" / "bulwark_is3.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(out)
    p.write_text(json.dumps(prev, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
