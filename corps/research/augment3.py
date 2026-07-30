"""Third cache augmentation: per-bond ytw array (dropped by the original cache
build, needed by ENDGAME/BALLAST/DEBUT).

  python corps/research/augment3.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from panel_io import load_full  # noqa: E402


def main():
    bonds = e2.load_cache()
    p = load_full(columns=["six", "date", "ytw"])
    p["day"] = p["date"].values.astype("datetime64[D]").astype(np.int32)
    print(f"panel {len(p)} rows; mapping ytw ...", flush=True)
    for k, (six, g) in enumerate(p.groupby("six")):
        b = bonds.get(six)
        if b is None:
            continue
        day = g["day"].to_numpy()
        order = np.argsort(day, kind="stable")
        y = g["ytw"].to_numpy(np.float32)[order]
        if len(y) != len(b["day"]):
            # align by day (defensive)
            idx = np.searchsorted(day[order], b["day"])
            idx = np.clip(idx, 0, len(y) - 1)
            y = y[idx]
        b["ytw"] = y
        if k % 10000 == 0:
            print(f"  {k} ...", flush=True)
    with open(e2.CACHE, "wb") as f:
        pickle.dump(bonds, f, protocol=5)
    print(f"rewrote cache ({e2.CACHE.stat().st_size/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
