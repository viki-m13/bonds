"""GRANITE-XL live-protocol replay, variant TIGHT-LOCK (XL_AUDIT.md 6c):
after each accepted entry, lock the issuer (and bond) until the 1-year-book
exit schedule + 30d (a deliberate time-based rule a desk can follow), limit
filter applied BEFORE capacity, lagged-mid recovery exits, true coupons.
This is the closest implementable analog of the published admission."""
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
print(f"limit-passed candidates: {len(cands)}", flush=True)

def base_exit_day(b, ed):
    """The 1-year-book exit (for the lockout schedule only)."""
    day = b["day"]
    pm = ~np.isnan(b["p_px"]); p_day = day[pm]
    k = np.searchsorted(p_day, ed + 365, side="left")
    if k < len(p_day) and p_day[k] <= ed + MAXH:
        return int(p_day[k])
    k2 = np.searchsorted(p_day, ed + MAXH, side="right") - 1
    if k2 < 0:
        return None
    return ed + MAXH

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

lock_by_iss, lock_by_bond, fills = {}, {}, []
for ed, six, ep in cands:
    if ed < lock_by_bond.get(six, -10**9):
        continue
    iss = six[:6]
    if ed < lock_by_iss.get(iss, -10**9):
        continue
    b = bonds[six]
    bx = base_exit_day(b, ed)
    ex = exit_lagged(b, ed)
    if bx is None or ex is None:
        continue
    xd, xp, st = ex
    coup = float(b.get("coupon_inv", b["coupon"]))
    fills.append(e2.Fill(six, ed, ep, xd, xp, coup, st))
    lock_by_iss[iss] = bx + 30
    lock_by_bond[six] = bx + 30
print(f"admitted fills: {len(fills)}", flush=True)

res = {}
w_all = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
res["full"] = show(bonds, fills, "TIGHT-LOCK full", weights=w_all)
for tag, (lo, hi) in [("IS", IS), ("OOS", OOS)]:
    sub = [(f, w) for f, w in zip(fills, w_all) if lo <= f.entry_day <= hi]
    res[tag] = show(bonds, [f for f, _ in sub], f"TIGHT-LOCK {tag}", weights=[w for _, w in sub])
Path(__file__).with_suffix(".json").write_text(json.dumps(res, default=float))
print("done", flush=True)
