"""Second cache augmentation for the sleeve screens. All columns are lagged
(shift-1 on print days) => usable point-in-time on the signal day.

Adds per bond:
  cs60    trailing-60d median credit spread (shift-1, >=8 obs)
  med15   trailing-15d median mid (shift-1, >=4 obs)
  spr60   trailing-60d median observed (s_px - p_px) on paired days (shift-1, >=5)
  qvmed90 trailing-90d median daily qvolume (shift-1, >=5)
  act90   trailing-90d active-day count (excl. today)
Global (saved separately, corps/data/cache_global.pkl):
  csmat_med: per-day cross-sectional median of cs/mat over gated bonds

  python corps/research/augment2.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402

GLOBAL = ROOT / "data" / "cache_global.pkl"


def main():
    bonds = e2.load_cache()
    print(f"{len(bonds)} bonds; augment2 ...", flush=True)
    day_acc = {}   # day -> list of cs/mat values (gated)
    for k, (six, b) in enumerate(bonds.items()):
        day = b["day"]
        idx = pd.to_datetime(day, unit="D")
        cs = pd.Series(b["cs"].astype(np.float64), index=idx)
        b["cs60"] = (cs.rolling("60D", min_periods=8).median().shift(1)
                     .to_numpy().astype(np.float32))
        mid = pd.Series(b["mid"].astype(np.float64), index=idx)
        b["med15"] = (mid.rolling("15D", min_periods=4).median().shift(1)
                      .to_numpy().astype(np.float32))
        spr = pd.Series((b["s_px"] - b["p_px"]).astype(np.float64), index=idx)
        b["spr60"] = (spr.rolling("60D", min_periods=5).median().shift(1)
                      .to_numpy().astype(np.float32))
        qv = pd.Series(np.nan_to_num(b["qv"].astype(np.float64)), index=idx)
        b["qvmed90"] = (qv.rolling("90D", min_periods=5).median().shift(1)
                        .to_numpy().astype(np.float32))
        b["act90"] = (np.searchsorted(day, day) -
                      np.searchsorted(day, day - 90)).astype(np.int16)
        g = b["elig"]
        mat = np.maximum(b["mat"].astype(np.float64), 0.5)
        v = b["cs"].astype(np.float64) / mat
        for d, val, ok in zip(day[g], v[g], np.isfinite(v[g])):
            if ok:
                day_acc.setdefault(int(d), []).append(val)
        if k % 10000 == 0:
            print(f"  {k} ...", flush=True)
    print("cross-sectional medians ...", flush=True)
    days_sorted = np.array(sorted(day_acc))
    med = np.array([np.median(day_acc[d]) for d in days_sorted], dtype=np.float32)
    with open(GLOBAL, "wb") as f:
        pickle.dump({"xs_days": days_sorted.astype(np.int32),
                     "csmat_med": med}, f, protocol=5)
    with open(e2.CACHE, "wb") as f:
        pickle.dump(bonds, f, protocol=5)
    print(f"rewrote cache ({e2.CACHE.stat().st_size/1e9:.2f} GB) + {GLOBAL}", flush=True)


if __name__ == "__main__":
    main()
