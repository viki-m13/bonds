"""BULWARK — IS screen (2003-2015). Pre-registered in BULWARK_RESEARCH.md §2.

Universe U: engine-eligible, mat<=5, cs>=4%, mid>=40 ("super cheap junk").
Screens S1..S6 tested alone and stacked. Mechanics identical to the audited
GRANITE/BEDROCK engine (ask entry <=7d after signal, limit ask<=prior mid+0.25,
bid exit 365-455d, real coupons, 30d cooldown, 1 position per CUSIP6);
EQUAL weights (there is no dislocation depth to weight by).

Order of operations (audit-safe): run_events -> issuer post-filters ->
limit_filter -> issuer capacity. Capacity is consumed only by trades that
pass every screen (no phantom blocking).

  python corps/research/bulwark_v1.py [oos]
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

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
OOS = (e2.D("2016-01-01"), e2.D("2024-01-01"))     # censor-safe: end - 455d
CS_MIN, MAT_MAX, PX_MIN = 0.04, 5, 40.0
RNG = np.random.default_rng(20260906)


# ---------------------------------------------------------------- universe
def u_mask(b):
    """Universe U membership per row (point-in-time only)."""
    cs, mid = b["cs"], b["mid"]
    return (b["elig"] & (b["mat"] <= MAT_MAX) & np.isfinite(cs) & (cs >= CS_MIN)
            & ~np.isnan(mid) & (mid >= PX_MIN))


def active90(b):
    d = b["day"]
    return (np.searchsorted(d, d) - np.searchsorted(d, d - 90)).astype(np.int32)


def build_xsec(bonds):
    """day -> (97th pct of cs in U, median trailing-90d activity in U)."""
    print("building cross-sections ...", flush=True)
    days, css, acts = [], [], []
    for b in bonds.values():
        m = u_mask(b)
        if not m.any():
            continue
        days.append(b["day"][m]); css.append(b["cs"][m]); acts.append(active90(b)[m])
    df = pd.DataFrame({"day": np.concatenate(days), "cs": np.concatenate(css),
                       "act": np.concatenate(acts)})
    q97 = df.groupby("day")["cs"].quantile(0.97).to_dict()
    amed = df.groupby("day")["act"].median().to_dict()
    print(f"  {len(q97):,} days; {len(df):,} U bond-days", flush=True)
    return q97, amed


# ---------------------------------------------------------------- screens
def sig_factory(bonds, q97, amed, s1=False, s2=False, s3=False, s5=False):
    def fn(b):
        m = u_mask(b)
        if not m.any():
            return None
        day, cs, mid = b["day"], b["cs"], b["mid"]
        if s1:
            q = np.array([q97.get(int(d), np.inf) for d in day])
            m = m & (cs <= q) & (mid >= 70.0)
        if s2:
            m = m & (b["mat"] <= 3)
        if s3:
            n = len(day)
            ok = np.zeros(n, bool)
            if n > 60:
                prev_cs, prev_mid = cs[:-60], mid[:-60]
                cur_cs, cur_mid = cs[60:], mid[60:]
                good = (np.isfinite(prev_cs) & np.isfinite(cur_cs)
                        & ~np.isnan(prev_mid) & ~np.isnan(cur_mid)
                        & (cur_cs <= 1.25 * prev_cs) & (cur_mid >= prev_mid - 5.0))
                ok[60:] = good
            m = m & ok
        if s5:
            a = active90(b)
            med = np.array([amed.get(int(d), 1e9) for d in day])
            m = m & (a > med)
        return m
    return fn


def sib_rows(bonds, issuers, six, sd, maxlag=10):
    """Latest sibling rows within `maxlag` days of the signal day."""
    out = []
    for s6 in issuers.get(six[:6], []):
        if s6 == six:
            continue
        sb = bonds[s6]
        j = np.searchsorted(sb["day"], sd, side="right") - 1
        if j < 0 or sd - sb["day"][j] > maxlag:
            continue
        out.append((sb, j))
    return out


def post_filters(bonds, issuers, fills, s4=False, s6=False):
    kept = []
    for f in fills:
        b = bonds[f.six]
        i = np.searchsorted(b["day"], f.entry_day, side="left") - 1
        if i < 0:
            continue
        sd = int(b["day"][i])
        sibs = sib_rows(bonds, issuers, f.six, sd)
        if s4:
            vals = [float(sb["cs"][j]) for sb, j in sibs
                    if np.isfinite(sb["cs"][j]) and sb["cs"][j] > 0]
            if len(vals) < 2 or not np.isfinite(b["cs"][i]):
                continue                       # evidence required
            if float(b["cs"][i]) < float(np.median(vals)) + 0.005:
                continue
        if s6:
            bad = False
            for sb, j in sibs:
                if (np.isfinite(sb["cs"][j]) and sb["cs"][j] > 0.20
                        and not np.isnan(sb["mid"][j]) and sb["mid"][j] < 70):
                    bad = True; break
            if bad:
                continue
        kept.append(f)
    return kept


def real_coupons(bonds, fills):
    return [e2.Fill(f.six, f.entry_day, f.entry_px, f.exit_day, f.exit_px,
                    float(bonds[f.six].get("coupon_inv", f.coupon)), f.stale)
            for f in fills]


# ---------------------------------------------------------------- control
def matched_control_U(bonds, fills, lo, hi, k=10):
    """Random ENTRY DAYS inside U on the same bonds — same limit rule, same
    365-455d exit, real coupons. Null = 'the screens picked no better day
    than chance within the same cheap-junk universe'."""
    rets = []
    for f in fills:
        b = bonds[f.six]
        day = b["day"]
        pool = np.flatnonzero(u_mask(b))
        if not len(pool):
            continue
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
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
                k2 = np.searchsorted(p_day, ed + MAXH, side="right") - 1
                if k2 < 0 or p_day[k2] <= ed:
                    continue
                xd, xp = ed + MAXH, float(p_px[k2])
            acc = cp / 100 / 365 * (xd - ed) * 100
            rets.append((xp - ep + acc) / ep)
    return np.array(rets)


# ---------------------------------------------------------------- stats
def stats(bonds, lab, fills, label, lo, hi, ctl=True, base=None):
    if not fills:
        print(f"  {label:26} n=0", flush=True)
        return {"n": 0}
    r = np.array([f.ret for f in fills])
    days, nav, daily = e2.mtm_nav(bonds, fills)
    ps = e2.perf_stats(days, nav, daily)
    nb = 0
    for f in fills:
        row = lab.get(f.six)
        if row and row[0] and f.entry_day <= row[1] <= f.exit_day + 30:
            nb += 1
    o = {"n": len(fills), "mean": float(r.mean()), "win": float((r > 0).mean()),
         "hold": float(np.mean([f.hold for f in fills])),
         "bust_n": nb, "bust_rate": nb / len(fills),
         "loss20": float((r < -0.20).mean()),
         "worst_decile": float(np.sort(r)[:max(1, len(r) // 10)].mean()),
         "issuers": len({f.six[:6] for f in fills}), **ps}
    if ctl:
        c = matched_control_U(bonds, fills, lo, hi)
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
          f"sh={o['sharpe_m']:5.2f} dd={o['maxdd']*100:6.1f}%"
          + (f" exc={o.get('excess',float('nan'))*100:+5.2f}% p={o.get('excess_p',float('nan')):.3f}"
             if ctl and 'excess' in o else "")
          + (f" [{'ADMIT' if o.get('admit') else 'reject'}]" if base else ""), flush=True)
    return o


def pipeline(bonds, issuers, q97, amed, lo, hi, s1=False, s2=False, s3=False,
             s4=False, s5=False, s6=False):
    f = e2.run_events(bonds, sig_factory(bonds, q97, amed, s1, s2, s3, s5),
                      min_hold=365, max_hold=MAXH, date_lo=lo, date_hi=hi)
    if s4 or s6:
        f = post_filters(bonds, issuers, f, s4, s6)
    f = limit_filter(bonds, f, cap=0.25)
    f = issuer_cap_filter(f, cap=1)
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
    print(f"\n[{tag}] cheap-junk universe, equal weight, real coupons:", flush=True)
    base = stats(bonds, lab, pipeline(bonds, issuers, q97, amed, lo, hi),
                 "U unscreened (baseline)", lo, hi)
    out["baseline"] = base
    if oos_mode:
        spec = json.loads((ROOT / "research" / "bulwark_is.json").read_text())["spec"]
        print(f"\n[one-shot OOS of the frozen spec: {spec}]", flush=True)
        out["spec"] = spec
        out["bulwark"] = stats(bonds, lab,
                               pipeline(bonds, issuers, q97, amed, lo, hi, **spec),
                               "BULWARK (frozen spec)", lo, hi, base=base)
    else:
        print("\n[single screens]", flush=True)
        for k in ("s1", "s2", "s3", "s4", "s5", "s6"):
            out[k] = stats(bonds, lab,
                           pipeline(bonds, issuers, q97, amed, lo, hi, **{k: True}),
                           f"+{k.upper()} only", lo, hi, base=base)
    p = ROOT / "research" / (f"bulwark_{'oos' if oos_mode else 'is'}.json")
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(out)
    p.write_text(json.dumps(prev, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
