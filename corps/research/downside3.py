"""F1 + F3 — the red-team's structurally sound downside levers (IS-only).
Both INCREASE crisis participation instead of selling it. One frozen variant
each (multiplicity discipline).

F1  DEPLOYMENT PACING, depth-keyed release: explicit capital simulation.
    New deployment capped at 4% of NAV per week, reserve earns T-bills;
    cap doubles when the day's signal queue has median depth >= 4pts
    (crisis = spend the reserve into the decline). Entries lost for lack
    of capital are lost (no queueing). Gate: same as E1-E5 plus the reserve
    must actually be spent inside crisis windows.

F3  DRAWDOWN-CONDITIONED EXIT LOOSENING: while the base book sits >15%
    below its high-water mark, the recovery-exit target relaxes by 1.5pts
    (sell partially-recovered strength faster, recycle into the deeper
    dislocations on offer). Conditioning state = the base book's own DD
    windows (one-pass approximation, documented).

  python corps/research/downside3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from downside import cl_entries, book, IS_LO, IS_HI  # noqa: E402
from combos import dynamic_exit, depth_of  # noqa: E402

MAXH = 455


# ------------------------------------------------------------------------ F3

def dd_windows(bonds, fills):
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    days, nav, daily = e2.mtm_nav(bonds, fills, weights=w)
    hwm = np.maximum.accumulate(nav)
    under = nav / hwm - 1 <= -0.15
    return days, under


def loosened_exit(bonds, entries, days, under, m=1.5):
    d0 = int(days[0])
    out = []
    for f in entries:
        b = bonds[f.six]
        day = b["day"]; mid = b["mid"]; med60 = b["med60"]
        i0 = np.searchsorted(day, f.entry_day, side="left") - 1
        tgt0 = float(med60[i0]) if i0 >= 0 and np.isfinite(med60[i0]) else None
        if tgt0 is None:
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
            ui = int(p_day[k]) - d0
            loose = 0 <= ui < len(under) and under[ui]
            tgt = tgt0 - (m if loose else 0.0)
            if mid[di] >= tgt:
                xd, xp, st = int(p_day[k]), float(p_px[k]), False
                break
        if xd is None:
            k2 = np.searchsorted(p_day, f.entry_day + MAXH, side="right") - 1
            if k2 < 0 or p_day[k2] <= f.entry_day:
                continue
            xd, xp, st = f.entry_day + MAXH, float(p_px[k2]), True
        out.append(e2.Fill(f.six, f.entry_day, f.entry_px, xd, xp, f.coupon, st))
    return out


# ------------------------------------------------------------------------ F1

def paced_capital_sim(bonds, fills, cap_frac=0.04, crisis_depth=4.0,
                      crisis_mult=2.0):
    """Explicit capital NAV: positions carry dollar weights set at entry from
    available cash under the weekly deployment cap; reserve earns T-bills.
    Position daily returns re-use each fill's real price path (entry ask ->
    mids -> exit bid). Entries that can't be funded are skipped."""
    fills = sorted(fills, key=lambda f: f.entry_day)
    d0 = min(f.entry_day for f in fills); d1 = max(f.exit_day for f in fills)
    days = np.arange(d0, d1 + 1)
    n = len(days)
    rf = e2.load_rf(days) / 365.0
    # per-fill daily return path (same construction as mtm_nav)
    paths = []
    for f in fills:
        b = bonds[f.six]
        day = b["day"]; mid = b["mid"]
        i0 = np.searchsorted(day, f.entry_day, side="right")
        i1 = np.searchsorted(day, f.exit_day, side="left")
        mds = [f.entry_day]; mks = [f.entry_px]
        for i in range(i0, i1):
            if not np.isnan(mid[i]):
                mds.append(int(day[i])); mks.append(float(mid[i]))
        mds.append(f.exit_day); mks.append(f.exit_px)
        acc = f.coupon / 100.0 / 365.0 * 100.0
        dr = np.zeros(f.exit_day - f.entry_day, dtype=np.float64)
        for k in range(1, len(mds)):
            gap = mds[k] - mds[k - 1]
            if gap <= 0:
                continue
            tot = (mks[k] + acc * gap) / mks[k - 1] - 1.0
            dr[mds[k - 1] - f.entry_day:mds[k] - f.entry_day] = \
                (1.0 + tot) ** (1.0 / gap) - 1.0
        paths.append(dr)
    # entry queue by day, with depth for the crisis key
    by_day = {}
    for idx, f in enumerate(fills):
        by_day.setdefault(f.entry_day, []).append(idx)
    nav = 1.0
    cash = 1.0
    open_pos = {}          # idx -> dollar value
    week_spent = 0.0
    week_anchor = d0
    navs = np.zeros(n)
    spent_crisis = spent_total = 0.0
    for t_i, t in enumerate(days):
        # accrue positions
        dead = []
        for idx, val in open_pos.items():
            f = fills[idx]
            k = t - f.entry_day
            if k >= len(paths[idx]):
                cash += val
                dead.append(idx)
            else:
                open_pos[idx] = val * (1.0 + paths[idx][k])
        for idx in dead:
            del open_pos[idx]
        cash *= (1.0 + rf[t_i])
        nav = cash + sum(open_pos.values())
        if t - week_anchor >= 7:
            week_anchor = t; week_spent = 0.0
        todays = by_day.get(int(t), [])
        if todays:
            depths = [depth_of(bonds, fills[idx]) for idx in todays]
            crisis = np.median(depths) >= crisis_depth
            cap = cap_frac * nav * (crisis_mult if crisis else 1.0)
            budget = max(cap - week_spent, 0.0)
            per = budget / len(todays) if todays else 0.0
            for idx in todays:
                take = min(per, cash)
                if take <= nav * 1e-5:
                    continue
                open_pos[idx] = take
                cash -= take
                week_spent += take
                spent_total += take
                if crisis:
                    spent_crisis += take
        navs[t_i] = cash + sum(open_pos.values())
    daily = np.diff(navs, prepend=navs[0]) / np.maximum(navs, 1e-9)
    daily[0] = 0.0
    ps = e2.perf_stats(days, navs / navs[0], daily)
    ps["crisis_spend_share"] = float(spent_crisis / max(spent_total, 1e-9))
    return ps


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}
    entries = cl_entries(bonds, IS_LO, IS_HI)
    xl = dynamic_exit(bonds, entries)
    print("\n[BASE]:", flush=True)
    base = book(bonds, xl, "XL base")
    out["base"] = base

    print("\n[F3] drawdown-conditioned exit loosening (-1.5pt while book <-15% HWM):",
          flush=True)
    days, under = dd_windows(bonds, xl)
    print(f"  loosened-exit regime active {under.mean()*100:.0f}% of days", flush=True)
    f3 = loosened_exit(bonds, entries, days, under, m=1.5)
    out["f3"] = book(bonds, f3, "F3 loosened", base)

    print("\n[F1] deployment pacing 4%/wk, 2x release at median queue depth>=4pt:",
          flush=True)
    ps = paced_capital_sim(bonds, xl)
    out["f1"] = ps
    keep = ps["cagr"] / base["cagr"]; ddcut = 1 - ps["maxdd"] / base["maxdd"]
    ok = keep >= 0.90 and ddcut >= 0.20
    ps["admit"] = bool(ok)
    print(f"  F1 paced                   cagr={ps['cagr']*100:+6.2f}% "
          f"sharpe={ps['sharpe_m']:.2f} maxdd={ps['maxdd']*100:6.1f}% "
          f"crisis_spend={ps['crisis_spend_share']*100:.0f}% "
          f"| keepCAGR={keep*100:.0f}% ddCut={ddcut*100:+.0f}% -> {'ADMIT' if ok else 'reject'}",
          flush=True)

    (ROOT / "research" / "downside3_is.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/downside3_is.json", flush=True)


if __name__ == "__main__":
    main()
