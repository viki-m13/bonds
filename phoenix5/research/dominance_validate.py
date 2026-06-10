"""Validate the dominant configs: are the gains structural or grid luck?

1. Gate-depth x gate-percentile matrix (does deeper de-risking help broadly?)
2. Blended parking (BIL/DIV mixes) for a bigger CAGR margin within the vol cap.
3. Year-by-year diff of final candidate vs production.
4. Episode analysis: what days does the deep gate change?
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
div = pd.concat({"c": credlo, "m": mfut}, axis=1).mean(axis=1).fillna(bil)


def overlay(raw, park, gate_lvl=0.5, gate_pct=0.99, smooth=3):
    rv = raw.rolling(60).std() * np.sqrt(252)
    vol_mult = (0.15 / rv).clip(0.25, 1.0).shift(1).fillna(1.0)
    scaled = raw * vol_mult
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(252, min_periods=30).max()
    dd_mult = (1.0 + (cum / hwm - 1) / -0.10).clip(0, 1).shift(1).fillna(1.0)
    sv = scaled.rolling(60).std()
    thr = sv.rolling(252, min_periods=60).quantile(gate_pct)
    ok = (sv <= thr).shift(1).fillna(True).astype(float)
    gate_mult = ok + (1 - ok) * gate_lvl
    total = (vol_mult * dd_mult * gate_mult).ewm(span=smooth).mean().clip(0, 1.0)
    idle = (1 - total).clip(lower=0)
    tc = total.diff().abs().fillna(0) * (10 / 1e4)
    return raw * total + idle * park - tc, (1 - ok)


print("1) gate depth x percentile (parking=BIL, smoothing=3d) — OOS SR / OOS CAGR / OOS MDD / IS SR")
for gp in [0.95, 0.97, 0.99]:
    line = f"  pct={gp:.2f}: "
    for gl in [0.5, 0.25, 0.0]:
        r, _ = overlay(raw, bil, gate_lvl=gl, gate_pct=gp)
        o, i = X.metrics(r.loc[X.OOS:]), X.metrics(r.loc[:X.IS_END])
        line += f"[lvl={gl}: {o['sr']:.2f}/{o['cagr']*100:.1f}%/{o['mdd']*100:.1f}%/is{i['sr']:.2f}]  "
    print(line)

print("\n2) parking mixes with gate25 (pct .99):")
bench = X.metrics(pd.read_csv(ROOT / "data/results/phoenix_production_returns.csv",
                              parse_dates=["Date"]).set_index("Date")["net_ret"].loc[X.OOS:])
best = None
for bfrac in [1.0, 0.75, 0.5, 0.25, 0.0]:
    park = bfrac * bil + (1 - bfrac) * div
    r, _ = overlay(raw, park, gate_lvl=0.25)
    o = X.metrics(r.loc[X.OOS:])
    dom = o["cagr"] > bench["cagr"] and o["mdd"] > bench["mdd"] and o["vol"] <= bench["vol"]
    print(f"  BIL {bfrac:.2f}/DIV {1-bfrac:.2f}: SR={o['sr']:.2f} CAGR={o['cagr']*100:.1f}% "
          f"vol={o['vol']*100:.2f}% MDD={o['mdd']*100:.1f}% {'DOM' if dom else ''}")
    if dom and (best is None or o["cagr"] > best[1]["cagr"]):
        best = (bfrac, o, r)

bfrac, o, rbest = best
print(f"\nfinal candidate: BIL {bfrac:.2f}/DIV {1-bfrac:.2f}, gate25, 3d smooth, idle parked")

print("\n3) yearly: candidate minus production (pct pts):")
phx = pd.read_csv(ROOT / "data/results/phoenix_production_returns.csv",
                  parse_dates=["Date"]).set_index("Date")["net_ret"]
idx = rbest.dropna().index.intersection(phx.index)
ya = rbest.loc[idx].groupby(idx.year).apply(lambda x: (1 + x).prod() - 1)
yb = phx.loc[idx].groupby(idx.year).apply(lambda x: (1 + x).prod() - 1)
print(((ya - yb) * 100).round(1).to_string())
print(f"  years better: {(ya > yb).sum()}/{len(ya)}")

print("\n4) gate episodes (days where 60d vol > 99th pct):")
_, gate_fire = overlay(raw, bil, gate_lvl=0.25)
fire = gate_fire[gate_fire > 0]
print(f"  total fire days: {len(fire)};  by year:")
print(fire.groupby(fire.index.year).size().to_string())
