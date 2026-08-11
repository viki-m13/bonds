"""BEDROCK seasonality event study (munis) — measurement only.

Pre-registered contrasts (literature: LPL/BlackRock summer technicals,
April tax-selling trough, Dec/Jan reinvestment):

  [M1] MUB total-return monthly seasonality 2008-2026: mean return by
       calendar month; contrast May-Jul vs Sep-Oct and vs Mar-Apr.
  [M2] KEYSTONE entry-vintage seasonality: base price_discount(3.0) fills
       (full window), forward ~1y return by ENTRY month, demeaned by entry
       YEAR (removes era effects). Contrast Mar-Apr entries vs Jun-Aug
       entries: if the reinvestment wave richens the tape into summer, the
       cheap entries should cluster in the spring trough.
  [M3] Tape-wide customer-sell imbalance by month (p_par share of two-sided
       par) - do sellers cluster in Dec (tax-loss) and Mar-Apr (tax bill)?

  python munis/research/bedrock_s_event.py
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
from limit_transfer import load_bonds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
res = {}

# ---- M1: MUB monthly seasonality --------------------------------------------
mub = (pd.read_csv(ROOT / "data" / "mub_daily.csv.gz", parse_dates=["date"])
       .set_index("date").iloc[:, 0])
mret = mub.resample("ME").last().pct_change().dropna()
bym = mret.groupby(mret.index.month).agg(["mean", "count"])
print("[M1] MUB mean monthly return by calendar month:", flush=True)
for m in range(1, 13):
    if m in bym.index:
        print(f"  {m:2d}: {bym.loc[m,'mean']*100:+.3f}%  (n={int(bym.loc[m,'count'])})", flush=True)
may_jul = mret[mret.index.month.isin([5, 6, 7])]
sep_oct = mret[mret.index.month.isin([9, 10])]
mar_apr = mret[mret.index.month.isin([3, 4])]
def tt(a, b):
    return float((a.mean() - b.mean()) /
                 np.sqrt(a.var() / len(a) + b.var() / len(b)))
res["m1"] = {"by_month": {int(k): float(v) for k, v in bym["mean"].items()},
             "mayjul_minus_sepoct_bp": float((may_jul.mean() - sep_oct.mean()) * 1e4),
             "t_mayjul_sepoct": tt(may_jul, sep_oct),
             "mayjul_minus_marapr_bp": float((may_jul.mean() - mar_apr.mean()) * 1e4),
             "t_mayjul_marapr": tt(may_jul, mar_apr)}
print(f"  May-Jul minus Sep-Oct: {res['m1']['mayjul_minus_sepoct_bp']:+.0f}bp/mo "
      f"(t={res['m1']['t_mayjul_sepoct']:+.2f}); "
      f"minus Mar-Apr: {res['m1']['mayjul_minus_marapr_bp']:+.0f}bp/mo "
      f"(t={res['m1']['t_mayjul_marapr']:+.2f})", flush=True)

# ---- M2: entry-vintage seasonality ------------------------------------------
print("\n[M2] KEYSTONE base entries: forward return by entry month (year-demeaned):", flush=True)
bonds = load_bonds()
fn = FACTORIES["price_discount"](discount=3.0)
fills = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                      date_lo=pd.Timestamp("2012-01-01"),
                      date_hi=pd.Timestamp("2025-04-08"))     # censor-safe
df = pd.DataFrame({"ret": [f.ret for f in fills],
                   "m": [f.entry_date.month for f in fills],
                   "y": [f.entry_date.year for f in fills]})
df["dm"] = df["ret"] - df.groupby("y")["ret"].transform("mean")
bym2 = df.groupby("m")["dm"].agg(["mean", "count"])
for m in range(1, 13):
    if m in bym2.index:
        print(f"  {m:2d}: {bym2.loc[m,'mean']*100:+.2f}%  (n={int(bym2.loc[m,'count'])})", flush=True)
spring = df[df["m"].isin([3, 4])]["dm"]; summer = df[df["m"].isin([6, 7, 8])]["dm"]
res["m2"] = {"by_month": {int(k): float(v) for k, v in bym2["mean"].items()},
             "spring_minus_summer_pp": float((spring.mean() - summer.mean()) * 100),
             "t": tt(spring, summer), "n_spring": int(len(spring)), "n_summer": int(len(summer))}
print(f"  Mar-Apr minus Jun-Aug entry vintage: "
      f"{res['m2']['spring_minus_summer_pp']:+.2f}pp (t={res['m2']['t']:+.2f}, "
      f"n={len(spring)}/{len(summer)})", flush=True)

# ---- M3: seller-imbalance seasonality ---------------------------------------
print("\n[M3] customer-sell share of two-sided par by month:", flush=True)
pnl = pd.read_parquet(ROOT / "data" / "panel_daily.parquet")
two = pnl[np.isfinite(pnl.get("s_par", np.nan)) & np.isfinite(pnl.get("p_par", np.nan))].copy()
two["sell_share"] = two["p_par"] / (two["p_par"] + two["s_par"])
bym3 = two.groupby(two["date"].dt.month)["sell_share"].mean()
for m in range(1, 13):
    if m in bym3.index:
        print(f"  {m:2d}: {bym3.loc[m]*100:.1f}%", flush=True)
res["m3"] = {int(k): float(v) for k, v in bym3.items()}

p = Path(__file__).resolve().parent / "results" / "bedrock_s_event.json"
p.write_text(json.dumps(res, default=float))
print(f"\nwrote {p}", flush=True)
