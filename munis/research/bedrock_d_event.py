"""BEDROCK Sleeve D — de-minimis event study on the EMMA tape.

Does the Ang-Bhansali-Xing market-discount concession exist in OUR data?
Three pre-registered tests (no strategy yet, measurement only):

  [T1] Yield discontinuity at the threshold: same-bond comparison of
       customer-buy YTW when the print is just below vs just above the bond's
       de-minimis threshold (within-bond, within-quarter pairs) — removes
       bond fixed effects the crude cross-section can't.
  [T2] Cross-sectional discontinuity: bucket bond-days by distance of price
       to threshold (pts), report mean YTW spread to each bond's own
       long-run median YTW, and customer-buy vs interdealer gap by bucket
       (the ABX "retail overpunishment" signature).
  [T3] Forward returns: 1-year forward return (customer-buy entry at the
       print -> customer-sell exit ~365d, engine conventions) for entries
       just below threshold vs matched entries 2-5 pts above, 2013+ where
       coverage exists; regime-split pre/post-2022.

Threshold = 100 - 0.25 * ceil(years-to-maturity)  (par-issue approximation;
OID bonds use revised issue price which we lack — noise, disclosed).

  python munis/research/bedrock_d_event.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results" / "bedrock_d_event.json"
res = {}

print("loading tape ...", flush=True)
pnl = pd.read_parquet(ROOT / "data" / "panel_daily.parquet")
uni = (pd.read_csv(ROOT / "data" / "universe" / "universe.csv.gz")
       .drop_duplicates("six").set_index("six"))
uni["maturity"] = pd.to_datetime(uni["maturity"], errors="coerce")
pnl = pnl.merge(uni[["maturity", "coupon"]], left_on="six",
                right_index=True, how="left")
pnl = pnl[pnl["maturity"].notna()].copy()
yrs = (pnl["maturity"] - pnl["date"]).dt.days / 365.25
pnl["yrs"] = yrs
pnl = pnl[pnl["yrs"] > 0.25]
pnl["thresh"] = 100 - 0.25 * np.ceil(pnl["yrs"])
pnl["dist"] = pnl["mid"] - pnl["thresh"]          # + above cliff, - below

# ---- T1: within-bond yield jump across the threshold ------------------------
print("[T1] within-bond crossings ...", flush=True)
rows = []
for six, g in pnl[np.isfinite(pnl["ytw"]) & np.isfinite(pnl["dist"])].groupby("six"):
    g = g.sort_values("date")
    below = g[(g["dist"] < 0) & (g["dist"] > -2)]
    above = g[(g["dist"] > 0) & (g["dist"] < 2)]
    if len(below) < 3 or len(above) < 3:
        continue
    # pair within 120 days to hold rates roughly constant
    for _, b in below.iterrows():
        cand = above[(above["date"] - b["date"]).abs().dt.days <= 120]
        if not len(cand):
            continue
        a = cand.iloc[(cand["date"] - b["date"]).abs().values.argmin()]
        rows.append({"six": six, "ytw_below": b["ytw"], "ytw_above": a["ytw"],
                     "gap_days": abs((a["date"] - b["date"]).days),
                     "year": b["date"].year})
t1 = pd.DataFrame(rows)
if len(t1):
    d = (t1["ytw_below"] - t1["ytw_above"]).astype(float)
    # cluster by bond
    bm = t1.assign(d=d).groupby("six")["d"].mean()
    res["t1"] = {"n_pairs": int(len(t1)), "n_bonds": int(t1["six"].nunique()),
                 "mean_bp": float(d.mean() * 100),
                 "median_bp": float(d.median() * 100),
                 "bond_mean_bp": float(bm.mean() * 100),
                 "t_bond": float(bm.mean() / (bm.std() / np.sqrt(len(bm))))}
    print(f"  pairs={len(t1)} bonds={t1['six'].nunique()} "
          f"ytw(below)-ytw(above): mean {d.mean()*100:+.1f}bp "
          f"bond-clustered mean {bm.mean()*100:+.1f}bp t={res['t1']['t_bond']:.2f}", flush=True)

# ---- T2: cross-sectional buckets --------------------------------------------
print("[T2] distance buckets ...", flush=True)
q = pnl[np.isfinite(pnl["ytw"])].copy()
med_ytw = q.groupby("six")["ytw"].transform("median")
q["ytw_ex"] = q["ytw"] - med_ytw
buckets = [(-8, -4), (-4, -2), (-2, -0.5), (-0.5, 0.5), (0.5, 2), (2, 4), (4, 8)]
t2 = []
for lo, hi in buckets:
    m = q[(q["dist"] >= lo) & (q["dist"] < hi)]
    two = m[np.isfinite(m["s_px"]) & np.isfinite(m["d_px"])]
    t2.append({"bucket": f"[{lo},{hi})", "n": int(len(m)),
               "ytw_excess_bp": float(m["ytw_ex"].mean() * 100),
               "buy_over_dealer_pts": float((two["s_px"] - two["d_px"]).mean())
               if len(two) else None})
    print(f"  dist {f'[{lo},{hi})':>10} n={len(m):7,} ytw_excess={t2[-1]['ytw_excess_bp']:+7.1f}bp "
          f"s-d={t2[-1]['buy_over_dealer_pts'] if t2[-1]['buy_over_dealer_pts'] is not None else float('nan'):+.3f}pts", flush=True)
res["t2"] = t2

# ---- T3: forward 1y returns below vs above ----------------------------------
print("[T3] forward returns (engine conventions) ...", flush=True)
coupons = uni["coupon"]
bonds = bt.prepare(pnl.drop(columns=["maturity", "coupon", "yrs", "thresh", "dist"]),
                   coupons)
dist_lu = {(r.six, r.date): r.dist for r in
           pnl[["six", "date", "dist"]].itertuples()}

def fwd_returns(sel):
    fills = []
    for six, g in bonds.items():
        a = bt._arr(six, g)
        for i in np.flatnonzero(a.elig):
            key = (six, g["date"].iloc[i])
            dv = dist_lu.get(key)
            if dv is None or not sel(dv):
                continue
            sd = a.day[i]
            j = np.searchsorted(a.s_day, sd, side="right")
            if j >= len(a.s_day) or a.s_day[j] - sd > 7:
                continue
            ed = int(a.s_day[j]); ep = float(a.s_px_at[j])
            ex = bt._exit_for(a, ed, 365, 455)
            if ex is None:
                continue
            xd, xp, st = ex
            acc = a.coupon / 100 / 365 * (xd - ed) * 100
            fills.append(((xp - ep + acc) / ep, pd.Timestamp(ed, unit="D").year, six))
    return fills

for tag, sel in [("below (-2,0)", lambda d: -2 <= d < 0),
                 ("above (+2,+5)", lambda d: 2 <= d < 5)]:
    fl = fwd_returns(sel)
    r = np.array([x[0] for x in fl])
    yr = np.array([x[1] for x in fl])
    for era, m in [("2013-2021", (yr >= 2013) & (yr <= 2021)),
                   ("2022-2026", yr >= 2022), ("all", yr >= 2005)]:
        if m.sum() < 10:
            continue
        print(f"  {tag:14} {era}: n={m.sum():5} mean={r[m].mean()*100:+6.2f}% "
              f"win={(r[m]>0).mean()*100:3.0f}%", flush=True)
        res[f"t3_{tag.split()[0]}_{era}"] = {"n": int(m.sum()),
                                             "mean": float(r[m].mean()),
                                             "win": float((r[m] > 0).mean())}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(res, default=float))
print(f"wrote {OUT}", flush=True)
