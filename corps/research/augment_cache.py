"""One-time cache augmentation: per-bond lagged trailing-60d median of mid
(med60) — the GRANITE signal input — plus trailing-90d mean daily volume
(qv90, point-in-time, excludes today) for liquidity tiering.

  python corps/research/augment_cache.py
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


def main():
    bonds = e2.load_cache()
    print(f"{len(bonds)} bonds; augmenting ...", flush=True)
    for k, (six, b) in enumerate(bonds.items()):
        day = b["day"]
        idx = pd.to_datetime(day, unit="D")
        s = pd.Series(b["mid"].astype(np.float64), index=idx)
        b["med60"] = (s.rolling("60D", min_periods=5).median().shift(1)
                      .to_numpy().astype(np.float32))
        # trailing-90d mean traded volume, excluding today
        qv = b["qv"].astype(np.float64)
        qv = np.where(np.isnan(qv), 0.0, qv)
        cq = np.concatenate([[0.0], np.cumsum(qv)])
        j0 = np.searchsorted(day, day - 90)
        j1 = np.searchsorted(day, day)          # rows strictly before today
        cnt = np.maximum(j1 - j0, 1)
        b["qv90"] = ((cq[j1] - cq[j0]) / cnt).astype(np.float32)
        if k % 10000 == 0:
            print(f"  {k} ...", flush=True)
    with open(e2.CACHE, "wb") as f:
        pickle.dump(bonds, f, protocol=5)
    print(f"rewrote {e2.CACHE} ({e2.CACHE.stat().st_size/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
