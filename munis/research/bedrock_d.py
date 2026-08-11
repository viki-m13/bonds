"""BEDROCK Sleeve D — de-minimis cliff + dislocation (muni), IS screen.

Pre-registered spec (BEDROCK_RESEARCH.md): KEYSTONE price_discount(3.0)
entries with limit cap, REAL issuer cap, lagged-mid recovery exits — the
audited honest stack — restricted to bonds whose signal-day mid sits in the
de-minimis cliff zone [threshold-3, threshold+1], threshold = 100 - 0.25 *
ceil(years-to-maturity). Baseline = same stack on the complement (outside the
zone). The sleeve's thesis: cliff-zone dislocations carry an EXTRA structural
concession (tax overpunishment) on top of the fire-sale discount.

Windows: IS 2012-2022, and (reported, not selected on) the 2013-2021 vs
2022+ regime split — the cliff universe is 92% post-2022, so IS breadth is
thin by construction; the event study (bedrock_d_event.py) carries the
identification weight and the OOS window carries the capital-relevant test.

  python munis/research/bedrock_d.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
from strategies import FACTORIES  # noqa: E402
from limit_transfer import load_bonds, limit_filter, limit_control  # noqa: E402
from keystone_xl import issuer_cap, issuer_of, med60_at, IS_LO, IS_HI, OOS_LO, OOS_HI  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAXH = 455
RNG = np.random.default_rng(131)


def cliff_map():
    uni = (pd.read_csv(ROOT / "data" / "universe" / "universe.csv.gz")
           .drop_duplicates("six").set_index("six"))
    mat = pd.to_datetime(uni["maturity"], errors="coerce")
    return mat


def exit_lagged(bonds, fills):
    out = []
    for f in fills:
        g = bonds[f.six]
        a = bt._arr(f.six, g)
        med = med60_at(g)
        try:
            tgt = float(med.asof(f.entry_date))
        except Exception:
            continue
        if not np.isfinite(tgt):
            continue
        day = g["date"].values.astype("datetime64[D]").astype(np.int64)
        mid = g["mid"].to_numpy(float)
        ed = np.datetime64(f.entry_date, "D").astype(np.int64)
        k0 = np.searchsorted(a.p_day, ed + 21, side="left")
        xd = xp = None
        for k in range(k0, len(a.p_day)):
            if a.p_day[k] > ed + MAXH:
                break
            di = np.searchsorted(day, a.p_day[k], side="left") - 1
            if di >= 0 and np.isfinite(mid[di]) and mid[di] >= tgt:
                xd, xp, st = int(a.p_day[k]), float(a.p_px_at[k]), False
                break
        if xd is None:
            k2 = np.searchsorted(a.p_day, ed + 455, side="right") - 1
            if k2 < 0 or a.p_day[k2] <= ed:
                continue
            xd, xp, st = ed + 455, float(a.p_px_at[k2]), True
        out.append(bt.Fill(f.six, f.entry_date, f.entry_px,
                           pd.Timestamp(xd, unit="D"), xp, f.coupon, st))
    return out


def in_zone(bonds, mats, f, lo_off=-3.0, hi_off=1.0):
    m = mats.get(f.six)
    if pd.isna(m):
        return None
    g = bonds[f.six]
    day = g["date"].values.astype("datetime64[D]").astype(np.int64)
    ed = np.datetime64(f.entry_date, "D").astype(np.int64)
    i = np.searchsorted(day, ed, side="left") - 1
    if i < 0 or not np.isfinite(g["mid"].iloc[i]):
        return None
    yrs = (m - g["date"].iloc[i]).days / 365.25
    if yrs <= 0.25:
        return None
    thr = 100 - 0.25 * np.ceil(yrs)
    d = float(g["mid"].iloc[i]) - thr
    return bool(lo_off <= d < hi_off)   # plain bool: np.True_ is not True


def summ(bonds, fills, tag, lo, hi, ctl=True):
    if not fills:
        print(f"  {tag:34} n=0", flush=True)
        return {"n": 0}
    r = np.array([f.ret for f in fills])
    out = {"n": len(fills), "mean": float(r.mean()), "win": float((r > 0).mean()),
           "hold": float(np.mean([f.hold_days for f in fills]))}
    if ctl:
        c = limit_control(bonds, fills, lo, hi)
        if len(c):
            out["excess"] = float(r.mean() - c.mean())
            by = {}
            for f in fills:
                by.setdefault(f.six, []).append(f.ret)
            keys = list(by); boots = []
            for _ in range(2000):
                ks = RNG.choice(len(keys), size=len(keys), replace=True)
                sm = np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean()
                boots.append(sm - RNG.choice(c, size=len(c), replace=True).mean())
            out["excess_p"] = float((np.array(boots) <= 0).mean())
    print(f"  {tag:34} n={out['n']:5} mean={out['mean']*100:+6.2f}% "
          f"win={out['win']*100:3.0f}% hold={out['hold']:4.0f}d"
          + (f" excess={out.get('excess', float('nan'))*100:+5.2f}% "
             f"p={out.get('excess_p', float('nan')):.3f}" if ctl and 'excess' in out else ""),
          flush=True)
    return out


def main():
    bonds = load_bonds()
    mats = cliff_map()
    fn = FACTORIES["price_discount"](discount=3.0)
    out = {}
    lo, hi = IS_LO, IS_HI          # IS ONLY — OOS runs once, later, after lock
    base = bt.run_signal(bonds, fn, min_hold=365, max_hold=MAXH,
                         date_lo=lo, date_hi=hi)
    stack = issuer_cap(limit_filter(bonds, base))
    zone_flags = {id(f): in_zone(bonds, mats, f) for f in stack}
    cliff = [f for f in stack if zone_flags[id(f)] is True]
    rest = [f for f in stack if zone_flags[id(f)] is False]
    print(f"\n[IS 2012-2022] cliff-zone vs complement (lagged exits, real cap):", flush=True)
    out["is_cliff"] = summ(bonds, exit_lagged(bonds, cliff), "CLIFF zone [thr-3,thr+1)", lo, hi)
    out["is_rest"] = summ(bonds, exit_lagged(bonds, rest), "complement", lo, hi)
    p = Path(__file__).resolve().parent / "results" / "bedrock_d_is.json"
    p.write_text(json.dumps(out, default=float))
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
