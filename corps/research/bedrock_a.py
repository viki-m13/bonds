"""BEDROCK Sleeve A — ratings-free fallen-angel proxy: detector precision +
IS event returns. IN-SAMPLE ONLY (2003-2015 for returns; detector counts
shown through 2024 for episode-clustering diagnostics only — no return stats
outside IS are computed).

Pre-registered detector (BEDROCK_RESEARCH.md): on bond b, event at row i if
  (1) 20-row cumulative mid return <= -8%
  (2) volume qv[i] >= 3x trailing qv90[i]
  (3) spread migration: cs[i-20] < 0.03 (IG-proxy) and cs[i] >= 0.03 (HY-proxy; cs is DECIMAL in the cache)
30-day cooldown between events per bond. Diagnostics: events/year (should
cluster in 2005 autos, 2008-09, 2020 COVID); IS forward ~1y returns at the
honest engine conventions vs a matched random control in the same bonds.

  python corps/research/bedrock_a.py
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

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
RNG = np.random.default_rng(173)


def detect(bonds):
    events = {}   # six -> list of row idx
    for six, b in bonds.items():
        mid, cs, qv, qv90, day = b["mid"], b["cs"], b["qv"], b["qv90"], b["day"]
        n = len(day)
        if n < 40:
            continue
        idx = []
        last = -10**9
        for i in range(20, n):
            if day[i] - last < 30:
                continue
            m0, m1 = mid[i - 20], mid[i]
            c0, c1 = cs[i - 20], cs[i]
            v, v90 = qv[i], qv90[i]
            if not (np.isfinite(m0) and np.isfinite(m1) and np.isfinite(c0)
                    and np.isfinite(c1) and np.isfinite(v) and np.isfinite(v90)):
                continue
            if m1 / m0 - 1 <= -0.08 and v >= 3 * v90 and v90 > 0 \
                    and c0 < 0.03 and c1 >= 0.03:
                idx.append(i)
                last = day[i]
        if idx:
            events[six] = idx
    return events


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    ev = detect(bonds)
    n_ev = sum(len(v) for v in ev.values())
    print(f"events: {n_ev} across {len(ev)} bonds", flush=True)

    yrs = []
    for six, idx in ev.items():
        d = bonds[six]["day"]
        yrs += [pd.Timestamp(int(d[i]), unit="D").year for i in idx]
    byyr = pd.Series(yrs).value_counts().sort_index()
    print("events per year (clustering diagnostic):", flush=True)
    print("  " + byyr.to_string().replace("\n", "\n  "), flush=True)

    # IS forward returns at engine conventions, real coupons
    fills = []
    for six, idx in ev.items():
        b = bonds[six]
        day = b["day"]
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        if not len(s_day) or not len(p_day):
            continue
        for i in idx:
            sd = day[i]
            j = np.searchsorted(s_day, sd, side="right")
            if j >= len(s_day) or s_day[j] - sd > 7:
                continue
            ed = int(s_day[j]); ep = float(s_px[j])
            if not (IS[0] <= ed <= IS[1]):
                continue
            k = np.searchsorted(p_day, ed + 365, side="left")
            if k < len(p_day) and p_day[k] <= ed + MAXH:
                xd, xp, st = int(p_day[k]), float(p_px[k]), False
            else:
                k2 = np.searchsorted(p_day, ed + MAXH, side="right") - 1
                if k2 < 0 or p_day[k2] <= ed:
                    continue
                xd, xp, st = ed + MAXH, float(p_px[k2]), True
            cp = float(b.get("coupon_inv", b["coupon"]))
            fills.append(e2.Fill(six, ed, ep, xd, xp, cp, st))
    out = {"n_events": int(n_ev), "by_year": {int(k): int(v) for k, v in byyr.items()}}
    print(f"\n[IS returns] fills={len(fills)}", flush=True)
    if fills:
        ctl = e2.matched_control(bonds, fills, min_hold=365, max_hold=MAXH)
        s = e2.summarize(fills, control=ctl)
        out["is"] = s
        print(f"  n={s['n']} mean={s['mean_ret']*100:+.2f}% win={s['win_rate']*100:.0f}% "
              f"stale={s['stale_share']*100:.0f}% excess={s.get('excess_vs_control', float('nan'))*100:+.2f}% "
              f"p={s.get('excess_p_boot', float('nan')):.3f}", flush=True)
    p = ROOT / "research" / "bedrock_a_is.json"
    p.write_text(json.dumps(out, default=float))
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
