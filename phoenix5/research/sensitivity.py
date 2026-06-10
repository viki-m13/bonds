"""Sensitivity of PHOENIX-5 OOS Sharpe to its meta-parameters.

If the headline OOS number only appears at one corner of the grid, it's
fragile/overfit. We want the OOS Sharpe distribution across reasonable
parameter neighborhoods.
"""
import sys
import warnings
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phoenix5"))

import phoenix5 as P  # noqa: E402


def run(tilt_strength, sleeve_vt, refresh, tilt_cap):
    df = P.pd.concat({s.name: s for s in
                      (P.build_phxcore(), P.build_mosaic(), P.build_credlo(), P.build_mfut())},
                     axis=1, sort=True).loc["2010-03-11":]
    vt = df.apply(lambda r: P.vol_target(r, sleeve_vt))
    W = P.pd.DataFrame(np.nan, index=vt.index, columns=vt.columns)
    for i in range(P.MIN_HIST, len(vt), refresh):
        hist = vt.iloc[max(0, i - 252):i]
        avail = [c for c in vt.columns if hist[c].notna().sum() >= P.MIN_HIST]
        if not avail:
            continue
        h = hist[avail]
        iv = 1.0 / h.std().clip(lower=1e-8)
        base = iv / iv.sum()
        srs = (h.mean() / h.std().clip(lower=1e-8)) * np.sqrt(252)
        tilt = np.exp((srs - srs.mean()).clip(-2, 2) * tilt_strength).clip(1 / tilt_cap, tilt_cap)
        w = base * tilt
        W.iloc[i, [vt.columns.get_loc(c) for c in avail]] = (w / w.sum()).values
    W = W.ffill()
    raw = (vt.fillna(0) * W).sum(axis=1)
    raw = raw[W.notna().any(axis=1)].dropna()
    bil = P.px("BIL").pct_change()
    net, _ = P.overlay(raw, bil)
    m_is = P.metrics(net.loc[:P.IS_END])["sharpe"]
    m_oos = P.metrics(net.loc[P.OOS_START:])["sharpe"]
    return m_is, m_oos


rows = []
for tilt_s, svt, refresh, tcap in itertools.product(
        [0.0, 0.25, 0.5, 1.0], [0.08, 0.10, 0.12], [21, 42], [2.0, 3.0]):
    m_is, m_oos = run(tilt_s, svt, refresh, tcap)
    rows.append({"tilt": tilt_s, "sleeve_vt": svt, "refresh": refresh,
                 "tilt_cap": tcap, "is_sr": m_is, "oos_sr": m_oos})
    print(rows[-1])

t = pd.DataFrame(rows)
print("\nOOS Sharpe distribution across grid:")
print(t["oos_sr"].describe().round(3).to_string())
print(f"\nmin={t.oos_sr.min():.2f}  median={t.oos_sr.median():.2f}  max={t.oos_sr.max():.2f}")
t.to_csv(ROOT / "phoenix5/results/sensitivity_grid.csv", index=False)
