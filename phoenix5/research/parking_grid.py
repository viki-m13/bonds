"""Grid: idle-capital parking choice x overlay aggressiveness.

Goal: strict OOS dominance vs production PHOENIX (CAGR>35.7, MDD>-17.7, vol<=14.7).
Parking candidates (all margin-free; MOSAIC shorts expressible via inverse ETFs):
  BIL, DIV (CREDLO+MFUT), MFUT, MOSAIC, MOSMF (50/50 MOSAIC+MFUT)
Overlay variants:
  prod (dd-10, gate 0.5), gate25 (deeper vol-gate de-risk), dd8 (earlier throttle),
  dd12 (later throttle) — all with 3d smoothing.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phoenix5"))
import phoenix5x as X  # noqa: E402

df = X.load_sleeves()
raw = df @ pd.Series(X.W_PROD)
bil = X.px("BIL").pct_change().reindex(raw.index).fillna(0)

credlo = X.build_credlo().reindex(raw.index)
mfut = pd.concat([X.px(t).pct_change() for t in ["DBMF", "KMLM", "CTA"]],
                 axis=1, sort=True).mean(axis=1).reindex(raw.index)
mosaic = pd.read_csv(ROOT / "phoenix5/results/mosaic_adaptive.csv",
                     parse_dates=[0], index_col=0)["ret"].reindex(raw.index)

def with_floor(s):
    return s.fillna(bil)

div = with_floor(pd.concat({"c": credlo, "m": mfut}, axis=1).mean(axis=1))
parks = {
    "BIL": bil,
    "DIV": div,
    "MFUT": with_floor(mfut),
    "MOSAIC": with_floor(mosaic),
    "MOSMF": with_floor(pd.concat({"a": mosaic, "b": mfut}, axis=1).mean(axis=1)),
}

def overlay(raw, park, dd_floor=-0.10, gate_lvl=0.5, smooth=3):
    rv = raw.rolling(60).std() * np.sqrt(252)
    vol_mult = (0.15 / rv).clip(0.25, 1.0).shift(1).fillna(1.0)
    scaled = raw * vol_mult
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(252, min_periods=30).max()
    dd_mult = (1.0 + (cum / hwm - 1) / dd_floor).clip(0, 1).shift(1).fillna(1.0)
    sv = scaled.rolling(60).std()
    thr = sv.rolling(252, min_periods=60).quantile(0.99)
    ok = (sv <= thr).shift(1).fillna(True).astype(float)
    gate_mult = ok + (1 - ok) * gate_lvl
    total = (vol_mult * dd_mult * gate_mult).ewm(span=smooth).mean().clip(0, 1.0)
    idle = (1 - total).clip(lower=0)
    tc = total.diff().abs().fillna(0) * (10 / 1e4)
    return raw * total + idle * park - tc

bench = X.metrics(pd.read_csv(ROOT / "data/results/phoenix_production_returns.csv",
                              parse_dates=["Date"]).set_index("Date")["net_ret"].loc[X.OOS:])
print(f"bench OOS: CAGR={bench['cagr']*100:.1f}% MDD={bench['mdd']*100:.1f}% vol={bench['vol']*100:.1f}% SR={bench['sr']:.2f}\n")

overlays = {
    "prod": dict(dd_floor=-0.10, gate_lvl=0.5),
    "gate25": dict(dd_floor=-0.10, gate_lvl=0.25),
    "gate0": dict(dd_floor=-0.10, gate_lvl=0.0),
    "dd8": dict(dd_floor=-0.08, gate_lvl=0.5),
    "dd8gate25": dict(dd_floor=-0.08, gate_lvl=0.25),
    "dd12": dict(dd_floor=-0.12, gate_lvl=0.5),
}
rows = []
for pname, park in parks.items():
    for oname, kw in overlays.items():
        r = overlay(raw, park, **kw)
        o = X.metrics(r.loc[X.OOS:])
        i = X.metrics(r.loc[:X.IS_END])
        dom = (o["cagr"] > bench["cagr"]) and (o["mdd"] > bench["mdd"]) and (o["vol"] <= bench["vol"])
        rows.append({"park": pname, "ovl": oname, "oos_sr": round(o["sr"], 2),
                     "oos_cagr": round(o["cagr"] * 100, 1), "oos_vol": round(o["vol"] * 100, 1),
                     "oos_mdd": round(o["mdd"] * 100, 1), "is_sr": round(i["sr"], 2),
                     "dom": "DOM" if dom else ""})
t = pd.DataFrame(rows).sort_values(["dom", "oos_cagr"], ascending=[False, False])
print(t.to_string(index=False))
t.to_csv(ROOT / "phoenix5/results/parking_grid.csv", index=False)
