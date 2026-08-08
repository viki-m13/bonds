"""GRANITE-XL live-protocol replay, variant OPEN (XL_AUDIT.md 6c) — issuer
capacity freed at the ACTUAL recovery exit:

  * signals/entries/limit filter identical to the frozen spec
  * admission swept chronologically with the issuer cap checked against the
    ACTUAL open book (recovery exits), not the 1-year schedule
  * per-bond 30d cooldown from the actual exit
  * recovery exit triggered on the PRIOR day's mid (executable), 455d stop
  * carry at recovered real coupons (coupon_inv)
  * depth-proportional weights, honest MTM NAV

Compare with the published GRANITE-XL headline.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/user/bonds/corps")
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2
from combos import depth_of, show

MAXH = 455
FULL = (e2.D("2003-01-01"), e2.D("2025-03-31") - MAXH)
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)

bonds = e2.load_cache()
for six, b in bonds.items():
    b["_six"] = six
print(f"loaded {len(bonds)} bonds", flush=True)
has_cinv = sum(1 for b in bonds.values() if "coupon_inv" in b)
print(f"bonds with coupon_inv: {has_cinv}", flush=True)

# 1-2) candidate entries (bond, entry_day, entry_px), limit-capped, deduped
cands = []
for six, b in bonds.items():
    med = b.get("med60")
    if med is None:
        continue
    gate = b["elig"] & (b["mat"] <= 5)
    sig = (b["s_px"] - med) <= -3.0
    idx = np.flatnonzero(sig & gate)
    if not len(idx):
        continue
    day = b["day"]
    sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
    seen = set()
    for i in idx:
        sd = day[i]
        j = np.searchsorted(s_day, sd, side="right")
        if j >= len(s_day) or s_day[j] - sd > 7:
            continue
        ed = int(s_day[j]); ep = float(s_px[j])
        if ed < FULL[0] or ed > FULL[1] or ed in seen:
            continue
        seen.add(ed)
        ii = np.searchsorted(day, ed, side="left") - 1
        if ii < 0 or not np.isfinite(b["mid"][ii]) or ep > b["mid"][ii] + 0.25:
            continue
        cands.append((ed, six, ep))
cands.sort()
print(f"candidate entries: {len(cands)}", flush=True)

def exit_lagged(b, ed):
    day = b["day"]; mid = b["mid"]; med60 = b["med60"]
    i0 = np.searchsorted(day, ed, side="left") - 1
    tgt = float(med60[i0]) if i0 >= 0 and np.isfinite(med60[i0]) else None
    if tgt is None:
        return None
    pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
    k0 = np.searchsorted(p_day, ed + 21, side="left")
    for k in range(k0, len(p_day)):
        if p_day[k] > ed + MAXH:
            break
        di = np.searchsorted(day, p_day[k], side="left") - 1
        if di >= 0 and np.isfinite(mid[di]) and mid[di] >= tgt:
            return int(p_day[k]), float(p_px[k]), False
    k2 = np.searchsorted(p_day, ed + MAXH, side="right") - 1
    if k2 < 0 or p_day[k2] <= ed:
        return None
    return ed + MAXH, float(p_px[k2]), True

# 3-4) chronological admission against the actual open book
open_by_iss = {}         # issuer -> list of exit days
last_exit_bond = {}      # six -> exit day
fills = []
for ed, six, ep in cands:
    if ed - last_exit_bond.get(six, -10**9) < 30:
        continue
    iss = six[:6]
    cur = [x for x in open_by_iss.get(iss, []) if x > ed]
    if len(cur) >= 1:
        open_by_iss[iss] = cur
        continue
    b = bonds[six]
    ex = exit_lagged(b, ed)
    if ex is None:
        open_by_iss[iss] = cur
        continue
    xd, xp, st = ex
    coup = float(b.get("coupon_inv", b["coupon"]))
    fills.append(e2.Fill(six, ed, ep, xd, xp, coup, st))
    cur.append(xd)
    open_by_iss[iss] = cur
    last_exit_bond[six] = xd
print(f"admitted fills: {len(fills)}", flush=True)

res = {}
w_all = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
res["full"] = show(bonds, fills, "LIVE-SIM full 2003-24", weights=w_all)
for tag, (lo, hi) in [("IS", IS), ("OOS", OOS)]:
    sub = [(f, w) for f, w in zip(fills, w_all) if lo <= f.entry_day <= hi]
    fl = [f for f, _ in sub]; ww = [w for _, w in sub]
    res[tag] = show(bonds, fl, f"LIVE-SIM {tag}", weights=ww)
print("\npublished GRANITE-XL for reference: full +16.62%/1.03, IS +19.24%/1.15, OOS +17.13%/1.03", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(res, default=float))
print("done", flush=True)
