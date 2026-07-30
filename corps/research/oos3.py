"""THIRD one-shot OOS batch (disclosed) — combo variants admitted by the
pre-registered IS gate in combos.py (must beat plain GRANITE-CL on both IS
Sharpe and IS CAGR). Paired design: identical entries, different exit policy
and/or position weighting; MTM decides.

  python corps/research/oos3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from combos import cl_fills, dynamic_exit, depth_of, show  # noqa: E402

MAXH = 455
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)


def main():
    admits = json.loads((ROOT / "research" / "combos_is.json").read_text())
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    fills = cl_fills(bonds, *OOS)
    print(f"\nGRANITE-CL OOS fills: {len(fills)}", flush=True)
    print("=" * 64, flush=True)
    print("ONE-SHOT OOS (batch 3) — 2016-2025 — paired on identical entries",
          flush=True)
    print("=" * 64, flush=True)
    out["base"] = show(bonds, fills, "CL base (1y hold)")

    if admits.get("sized", {}).get("admit_oos"):
        w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
        out["sized"] = show(bonds, fills, "depth-weighted", weights=w)
    if admits.get("dynamic", {}).get("admit_oos"):
        dfills = dynamic_exit(bonds, fills)
        out["dynamic"] = show(bonds, dfills, "recovery exit (21d+)")
        if admits.get("sized_dynamic", {}).get("admit_oos"):
            wd = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in dfills]
            out["sized_dynamic"] = show(bonds, dfills, "depth-wt + recovery", weights=wd)

    (ROOT / "research" / "oos3_results.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/oos3_results.json", flush=True)


if __name__ == "__main__":
    main()
