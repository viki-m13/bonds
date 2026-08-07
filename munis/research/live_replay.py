"""KEYSTONE-XL live-protocol replay (XL_AUDIT.md 6c):
chronological admission, REAL issuer cap vs actual open book, 30d cooldown
from actual exit, lagged-mid recovery exit, 455d stop."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/bonds/munis/research")
import backtest as bt
from strategies import FACTORIES
from limit_transfer import load_bonds
from keystone_xl import issuer_of, med60_at, IS_LO, IS_HI, OOS_LO, OOS_HI

MAXH = 455
bonds = load_bonds()
fn = FACTORIES["price_discount"](discount=3.0)
print(f"{len(bonds)} muni bonds", flush=True)

# candidate entries: signal -> next S print <=7d -> limit cap
cands = []
lo_day = np.datetime64(IS_LO, "D").astype(np.int64)
hi_day = np.datetime64(OOS_HI, "D").astype(np.int64)
for six, g in bonds.items():
    sig = fn(g)
    if sig is None:
        continue
    a = bt._arr(six, g)
    idx = np.flatnonzero(sig.to_numpy() & a.elig)
    if not len(idx):
        continue
    day = g["date"].values.astype("datetime64[D]").astype(np.int64)
    mid = g["mid"].to_numpy(float)
    seen = set()
    for i in idx:
        sd = a.day[i]
        j = np.searchsorted(a.s_day, sd, side="right")
        if j >= len(a.s_day) or a.s_day[j] - sd > 7:
            continue
        ed = int(a.s_day[j]); ep = float(a.s_px_at[j])
        if ed < lo_day or ed > hi_day or ed in seen:
            continue
        seen.add(ed)
        ii = np.searchsorted(day, ed, side="left") - 1
        if ii < 0 or not np.isfinite(mid[ii]) or ep > mid[ii] + 0.25:
            continue
        cands.append((ed, six, ep))
cands.sort()
print(f"candidate entries: {len(cands)}", flush=True)

def exit_lagged(six, g, ed_ts, ed):
    a = bt._arr(six, g)
    med = med60_at(g)
    try:
        tgt = float(med.asof(ed_ts))
    except Exception:
        return None
    if not np.isfinite(tgt):
        return None
    day = g["date"].values.astype("datetime64[D]").astype(np.int64)
    mid = g["mid"].to_numpy(float)
    k0 = np.searchsorted(a.p_day, ed + 21, side="left")
    for k in range(k0, len(a.p_day)):
        if a.p_day[k] > ed + MAXH:
            break
        di = np.searchsorted(day, a.p_day[k], side="left") - 1
        if di >= 0 and np.isfinite(mid[di]) and mid[di] >= tgt:
            return int(a.p_day[k]), float(a.p_px_at[k]), False
    k2 = np.searchsorted(a.p_day, ed + MAXH, side="right") - 1
    if k2 < 0 or a.p_day[k2] <= ed:
        return None
    return ed + MAXH, float(a.p_px_at[k2]), True

open_by_iss, last_exit_bond, fills = {}, {}, []
for ed, six, ep in cands:
    if ed - last_exit_bond.get(six, -10**9) < 30:
        continue
    iss = issuer_of(six)
    cur = [x for x in open_by_iss.get(iss, []) if x > ed]
    if len(cur) >= 1:
        open_by_iss[iss] = cur
        continue
    g = bonds[six]
    ed_ts = pd.Timestamp(ed, unit="D")
    ex = exit_lagged(six, g, ed_ts, ed)
    if ex is None:
        open_by_iss[iss] = cur
        continue
    xd, xp, st = ex
    a = bt._arr(six, g)
    fills.append(bt.Fill(six, ed_ts, ep, pd.Timestamp(xd, unit="D"), xp, a.coupon, st))
    cur.append(xd); open_by_iss[iss] = cur; last_exit_bond[six] = xd
print(f"admitted fills: {len(fills)}", flush=True)

from xl_equity import nav_series, stats
res = {}
for tag, lo, hi in [("IS", IS_LO, IS_HI), ("OOS", OOS_LO, OOS_HI), ("FULL", IS_LO, OOS_HI)]:
    fl = [f for f in fills if lo <= f.entry_date <= hi]
    r = np.array([f.ret for f in fl])
    st_ = stats(nav_series(fl)) if fl else {}
    res[tag] = {"n": len(fl), "mean": float(r.mean()), "win": float((r > 0).mean()), **st_}
    print(f"[{tag}] n={len(fl)} mean={r.mean()*100:+.2f}% win={(r>0).mean()*100:.0f}% "
          f"cagr={st_.get('cagr',float('nan'))*100:+.2f}% maxdd={st_.get('maxdd',float('nan'))*100:.1f}%", flush=True)
print("\npublished (fixed-cap) reference: IS +5.27%/n325/cagr7.84, OOS +6.85%/n336/cagr9.29", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(res, default=float))
print("done", flush=True)
