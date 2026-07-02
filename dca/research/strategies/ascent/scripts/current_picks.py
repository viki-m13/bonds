"""Emit the current pick lists as of the latest data date.
  MODE 2 (stock-form leaders basket): top-5 NDX PIT members by dollar volume,
    trend gate (px > 10mo MA), with exit status of a hypothetical held book.
  Reference: honest ML top-12 (runner-gated) — published for transparency, NOT
    as an expected QQQ-beater (see FINDINGS).
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M).reindex(columns=cols)
ma10 = me.rolling(10, min_periods=10).mean()
mom3 = me / me.shift(3) - 1

dt = M[-1]
print(f"As of month bar: {dt.date()} (data through 2026-06-18)")

mem = pd.read_parquet(f"{REPO}/data/pit/n100_panel_member.parquet")
mem.index = pd.to_datetime(mem.index)
memM = mem.resample("MS").last().reindex(M).ffill(limit=2).fillna(False)
memM = memM.reindex(columns=cols, fill_value=False).astype(bool)

# MODE 2
eln = liq.loc[dt] & memM.loc[dt] & (me.loc[dt] > ma10.loc[dt])
dvsc = dv.loc[dt].where(eln)
top = dvsc.sort_values(ascending=False).head(8)
print("\nMODE 2 — leaders basket (top-5 NDX by $ volume, trend-gated); buy top-5:")
for i, (tk, v) in enumerate(top.items(), 1):
    star = " <== BUY" if i <= 5 else " (next in line)"
    print(f"  {i}. {tk:6} ADV ${v/1e6:,.0f}M  px {me.loc[dt, tk]:,.2f}  "
          f"10moMA {ma10.loc[dt, tk]:,.2f}  mom3 {mom3.loc[dt, tk]:+.1%}{star}")

# Reference ML list
accel = PROB - PROB.shift(2)
elig = liq.loc[dt] & (me.loc[dt] >= 3) & (dv.loc[dt] >= 2e6) & (me.loc[dt] > ma10.loc[dt]) & (mom3.loc[dt] > 0)
sc = PROB.loc[dt].where(elig & (accel.loc[dt] > 0)).dropna().sort_values(ascending=False)
print("\nReference — ML top-12 (runner+accel gated), NOT an expected QQQ-beater:")
for i, (tk, v) in enumerate(sc.head(12).items(), 1):
    print(f"  {i:2}. {tk:6} prob {v:.3f}  ADV ${dv.loc[dt, tk]/1e6:,.0f}M  mom3 {mom3.loc[dt, tk]:+.1%}")

out = {"asof": str(dt.date()),
       "mode2_buy": list(top.head(5).index),
       "ml_reference": list(sc.head(12).index)}
with open(f"{HERE}/current_picks.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nsaved current_picks.json")
