"""GRANITE follow-up experiments on the fast engine:

[0] Honest mark-to-market baselines (monthly-marked Sharpe vs T-bill) for the
    published operating points: full >=3pt, focused <=5y >=3pt, >=4pt.
[A] Stacking the two accepted levers: depth >=4pt x duration <=5y.
[B] Liquidity-tiered capacity: focused book split by trailing-90d traded
    volume (point-in-time) — where does the alpha sit, and what does a
    tradable-size book look like?
[C] Issuer concentration cap: at most one concurrent position per issuer
    (6-char CUSIP prefix), greedy chronological admission — kills issuer
    clustering; does per-trade excess survive?

Engine parity with the published numbers is asserted first (same signal on the
fast engine must reproduce the >=3pt full-universe result within tolerance).

  python corps/research/granite_experiments.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402

MAXH = 455
LO_ALL = e2.D("2002-01-01")
HI_ALL = e2.D("2025-03-31") - MAXH
IS = (e2.D("2002-01-01"), e2.D("2015-12-31"))
OOS = (e2.D("2016-01-01"), HI_ALL)


def sig_disc(disc):
    def fn(b):
        med = b.get("med60")
        if med is None:
            return None
        return (b["s_px"] - med) <= -disc
    return fn


def gate_mat5(b):
    return b["mat"] <= 5


def ev(bonds, disc, lo, hi, label, gate=None, mtm=False):
    fills = e2.run_events(bonds, sig_disc(disc), min_hold=365, max_hold=MAXH,
                          date_lo=lo, date_hi=hi, extra_gate=gate)
    ctl = e2.matched_control(bonds, fills, min_hold=365, max_hold=MAXH,
                            extra_gate=gate)
    s = e2.summarize(fills, control=ctl)
    print(f"  {label:32} n={s.get('n',0):6} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+6.2f}% excess={s.get('excess_vs_control',0)*100:+5.2f}% "
          f"p={s.get('excess_p_boot',1):.3f}", flush=True)
    if mtm and fills:
        days, nav, daily = e2.mtm_nav(bonds, fills)
        ps = e2.perf_stats(days, nav, daily)
        s["mtm"] = ps
        print(f"    MTM: cagr={ps['cagr']*100:+.2f}% total={ps['total']*100:+.1f}% "
              f"maxdd={ps['maxdd']*100:.1f}% sharpe_m={ps['sharpe_m']:.2f} "
              f"sharpe_d={ps['sharpe_d']:.2f} sharpe_a={ps['sharpe_a']:.2f} "
              f"vol_m={ps['vol_m_ann']*100:.1f}%", flush=True)
    return s, fills


def issuer_cap_filter(fills, cap=1):
    """Greedy chronological admission: reject entries while `cap` positions in
    the same issuer (6-char prefix) are open. Uses only entry-time book state."""
    fills = sorted(fills, key=lambda f: f.entry_day)
    open_by = {}
    kept = []
    for f in fills:
        iss = f.six[:6]
        cur = open_by.get(iss, [])
        cur = [x for x in cur if x > f.entry_day]
        if len(cur) < cap:
            kept.append(f)
            cur.append(f.exit_day)
        open_by[iss] = cur
    return kept


def main():
    bonds = e2.load_cache()
    print(f"loaded cache: {len(bonds)} bonds", flush=True)
    out = {}

    print("\n[PARITY] fast engine vs published (>=3pt full universe):", flush=True)
    s, fills3 = ev(bonds, 3.0, LO_ALL, HI_ALL, "full >=3pt (expect ~+2.06%)", mtm=True)
    out["parity_full3"] = s
    ref = 0.02063
    d = abs(s.get("excess_vs_control", 0) - ref)
    print(f"  parity delta vs published: {d*100:.2f}pp "
          f"({'OK' if d < 0.005 else 'CHECK'})", flush=True)

    print("\n[0] Honest MTM baselines:", flush=True)
    s5, fills5 = ev(bonds, 3.0, LO_ALL, HI_ALL, "focused <=5y >=3pt", gate=gate_mat5, mtm=True)
    out["base_5y"] = s5
    s4, _ = ev(bonds, 4.0, LO_ALL, HI_ALL, "full >=4pt", mtm=True)
    out["base_4pt"] = s4

    print("\n[A] Stack: >=4pt x <=5y:", flush=True)
    for tag, lo, hi in [("full", LO_ALL, HI_ALL), ("IS", *IS), ("OOS", *OOS)]:
        s, f = ev(bonds, 4.0, lo, hi, f"{tag} >=4pt & <=5y", gate=gate_mat5,
                  mtm=(tag == "full"))
        out[f"stack_{tag}"] = s

    print("\n[B] Liquidity tiers within focused <=5y >=3pt (trailing qv90 tier):", flush=True)
    # per-entry trailing volume: tier by qv90 at the signal day
    qv_cuts = [0, 50, 250, 1000, 10**9]   # $000s/day avg
    tier_names = ["<50k", "50-250k", "250k-1M", ">1M"]
    tiers = {t: [] for t in tier_names}
    for f in fills5:
        b = bonds[f.six]
        i = np.searchsorted(b["day"], f.entry_day)
        i = min(max(i, 0), len(b["day"]) - 1)
        q = float(b["qv90"][i])
        for t, lo_c, hi_c in zip(tier_names, qv_cuts[:-1], qv_cuts[1:]):
            if lo_c <= q < hi_c:
                tiers[t].append(f)
                break
    out["liquidity"] = {}
    for t in tier_names:
        fl = tiers[t]
        if not fl:
            continue
        ctl = e2.matched_control(bonds, fl, min_hold=365, max_hold=MAXH,
                                 extra_gate=gate_mat5)
        s = e2.summarize(fl, control=ctl)
        avg_qv = np.mean([float(bonds[f.six]["qv90"][min(np.searchsorted(bonds[f.six]['day'], f.entry_day), len(bonds[f.six]['day'])-1)]) for f in fl])
        s["avg_qv90_k"] = float(avg_qv)
        out["liquidity"][t] = s
        print(f"  tier {t:9} n={s['n']:6} win={s['win_rate']*100:3.0f}% "
              f"mean={s['mean_ret']*100:+6.2f}% excess={s.get('excess_vs_control',0)*100:+5.2f}% "
              f"p={s.get('excess_p_boot',1):.3f} avg_dailyvol={avg_qv:,.0f}k", flush=True)

    print("\n[C] Issuer concentration cap (max 1 concurrent per issuer), <=5y >=3pt:", flush=True)
    capped = issuer_cap_filter(fills5, cap=1)
    ctl = e2.matched_control(bonds, capped, min_hold=365, max_hold=MAXH,
                             extra_gate=gate_mat5)
    s = e2.summarize(capped, control=ctl)
    out["issuer_cap"] = s
    print(f"  capped: n={s['n']} (from {len(fills5)}) win={s['win_rate']*100:.0f}% "
          f"mean={s['mean_ret']*100:+.2f}% excess={s.get('excess_vs_control',0)*100:+.2f}% "
          f"p={s.get('excess_p_boot',1):.3f}", flush=True)
    if capped:
        days, nav, daily = e2.mtm_nav(bonds, capped)
        ps = e2.perf_stats(days, nav, daily)
        s["mtm"] = ps
        print(f"    MTM: cagr={ps['cagr']*100:+.2f}% maxdd={ps['maxdd']*100:.1f}% "
              f"sharpe_m={ps['sharpe_m']:.2f}", flush=True)

    (ROOT / "research" / "granite_experiments.json").write_text(
        json.dumps(out, default=float))
    print("\nwrote corps/research/granite_experiments.json", flush=True)


if __name__ == "__main__":
    main()
