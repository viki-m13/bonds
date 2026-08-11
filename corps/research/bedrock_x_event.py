"""BEDROCK Sleeve X — index-exclusion concession event study (measurement
only; Dick-Nielsen & Rossi RFS 2019 replication on our tape).

OSBAP truncates coverage at bond_maturity = 1.0y — exactly the Bloomberg index
exclusion boundary. The truncated cohort (min observed maturity ~1.0) are the
bonds that hit the exclusion; their final observable weeks are the exclusion
selling window. Pre-registered measurement:

  For each truncated-cohort bond, event time = calendar days until its 1.0y
  crossing (last_date + (last_mat - 1.0) * 365.25). Difference-in-difference:
  bond's credit spread in event-time buckets minus its own [-180,-120) mean,
  minus the same-period market-wide median cs change (same calendar months).
  Also: the final-10-day ask-implied YTM spread over the bond's own baseline -
  the harvestable concession for a hold-to-maturity buyer.

  python corps/research/bedrock_x_event.py [SRC_PARQUET]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = (sys.argv[1] if len(sys.argv) > 1 else
       "/tmp/claude-0/-home-user-bonds/05b2c7c5-ea74-5833-b5c3-802491eebbc1/"
       "scratchpad/stage1_osbap_0k_volume_2025.parquet")
COLS = ["cusip_id", "trd_exctn_dt", "bond_maturity", "credit_spread",
        "ytm", "pr", "prc_ask"]

pf = pq.ParquetFile(SRC)
parts = []
print(f"scanning {pf.num_row_groups} row groups ...", flush=True)
for i in range(pf.num_row_groups):
    df = pf.read_row_group(i, columns=COLS).to_pandas()
    df = df[np.isfinite(df["bond_maturity"])]
    parts.append(df)
    if i % 10 == 0:
        print(f"  rg {i}", flush=True)
raw = pd.concat(parts, ignore_index=True)
del parts
raw["cusip_id"] = raw["cusip_id"].astype(str)
print(f"{len(raw):,} rows, {raw['cusip_id'].nunique():,} bonds; "
      f"global min maturity = {raw['bond_maturity'].min():.3f}", flush=True)

g = raw.groupby("cusip_id")["bond_maturity"].min()
cohort = set(g[(g >= 0.99) & (g <= 1.06)].index)
print(f"truncated-at-1y cohort: {len(cohort):,} bonds", flush=True)

# market-wide median cs by month (for the diff-in-diff market leg)
raw["ym"] = raw["trd_exctn_dt"].values.astype("datetime64[M]")
mkt = raw.groupby("ym")["credit_spread"].median()

sub = raw[raw["cusip_id"].isin(cohort)].copy()
del raw
# crossing date per bond
last = sub.sort_values("trd_exctn_dt").groupby("cusip_id").tail(1)
cross = {r.cusip_id: r.trd_exctn_dt + pd.Timedelta(days=float((r.bond_maturity - 1.0) * 365.25))
         for r in last.itertuples()}
sub["cross"] = sub["cusip_id"].map(cross)
sub["ev"] = (sub["trd_exctn_dt"] - sub["cross"]).dt.days   # negative = before crossing

BUCKETS = [(-180, -120), (-120, -60), (-60, -30), (-30, -10), (-10, 1)]
rows = []
base_lo, base_hi = -180, -120
res = {"n_cohort": len(cohort)}
print("\n[diff-in-diff] cs vs own baseline vs market (bp):", flush=True)
per_bond = []
for six, gg in sub[np.isfinite(sub["credit_spread"])].groupby("cusip_id"):
    b0 = gg[(gg["ev"] >= base_lo) & (gg["ev"] < base_hi)]
    if len(b0) < 3:
        continue
    own0 = b0["credit_spread"].median()
    m0 = mkt.reindex(b0["ym"]).median()
    row = {"six": six}
    for lo, hi in BUCKETS[1:]:
        w = gg[(gg["ev"] >= lo) & (gg["ev"] < hi)]
        if len(w) < 2:
            continue
        own = w["credit_spread"].median()
        mw = mkt.reindex(w["ym"]).median()
        row[f"{lo}_{hi}"] = (own - own0) - (mw - m0)
    per_bond.append(row)
pb = pd.DataFrame(per_bond).set_index("six")
for lo, hi in BUCKETS[1:]:
    c = f"{lo}_{hi}"
    if c in pb:
        v = pb[c].dropna() * 100.0   # cs is decimal? OSBAP credit_spread in decimal or %?
        print(f"  ev[{lo:>4},{hi:>4})  n={len(v):5}  mean={v.mean():+7.1f}  "
              f"median={v.median():+7.1f}  t={v.mean()/(v.std()/np.sqrt(len(v))):+5.2f}", flush=True)
        res[f"dd_{c}"] = {"n": int(len(v)), "mean": float(v.mean()),
                          "median": float(v.median()),
                          "t": float(v.mean() / (v.std() / np.sqrt(len(v))))}

# harvestable ask-side concession: final-10-day ask YTM minus own baseline ytm
print("\n[ask-side] final-10d ask-implied ytm pickup vs own baseline:", flush=True)
picks = []
for six, gg in sub.groupby("cusip_id"):
    fin = gg[(gg["ev"] >= -10)]
    fin = fin[np.isfinite(fin["ytm"])]
    b0 = gg[(gg["ev"] >= base_lo) & (gg["ev"] < base_hi)]
    b0 = b0[np.isfinite(b0["ytm"])]
    if len(fin) and len(b0) >= 3:
        picks.append(fin["ytm"].median() - b0["ytm"].median())
pk = pd.Series(picks) * 100.0
print(f"  n={len(pk)} mean={pk.mean():+.1f} median={pk.median():+.1f} "
      f"p75={pk.quantile(.75):+.1f} (units: ytm*100)", flush=True)
res["ask_pickup"] = {"n": int(len(pk)), "mean": float(pk.mean()),
                     "median": float(pk.median()), "p75": float(pk.quantile(.75))}

p = ROOT / "research" / "bedrock_x_event.json"
p.write_text(json.dumps(res, default=float))
print(f"wrote {p}", flush=True)
