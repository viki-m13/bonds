"""Honest mark-to-market for the ML-selected book (GRANITE-MLX = XL entries
ranked by the walk-forward GBM, top-half by training-median threshold).

Rebuilds the actual XL fills (with real ask/bid prices), reproduces the
deterministic walk-forward picks (random_state=7), and computes honest daily
MTM (marks at real mid prints) for all / top / bottom books, IS and OOS.

  python corps/research/mlx_mtm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from flowml import xl_fills, build_features, walk_forward, FEATURES  # noqa: E402


def mtm(bonds, fills, weights=None):
    r = e2.mtm_nav(bonds, fills, weights=weights)
    return e2.perf_stats(*r) if r else None


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    fills = xl_fills(bonds)
    rows = build_features(bonds, fills)
    # align fills to rows by (six, entry_day)
    key2fill = {(f.six, f.entry_day): f for f in fills}
    afills = [key2fill[(r["six"], r["entry_day"])] for r in rows]
    picks_gbm, picks_r, preds, yr = walk_forward(rows)
    out = {}
    for tag, y0, y1 in [("IS", 2008, 2015), ("OOS", 2016, 2024)]:
        span = (yr >= y0) & (yr <= y1)
        print(f"\n{tag} {y0}-{y1} (honest MTM, marks at real mid prints):", flush=True)
        for name, mask in [("all XL", span), ("MLX top", span & picks_gbm),
                           ("bottom", span & ~picks_gbm)]:
            fl = [f for f, m in zip(afills, mask) if m]
            ps = mtm(bonds, fl)
            if ps:
                rr = np.array([f.ret for f in fl])
                out[f"{tag}_{name.replace(' ','_')}"] = {
                    **ps, "n": len(fl), "mean_ret": float(rr.mean()),
                    "win": float((rr > 0).mean())}
                print(f"  {name:10} n={len(fl):5} mean={rr.mean()*100:+6.2f}% "
                      f"win={(rr>0).mean()*100:3.0f}% cagr={ps['cagr']*100:+6.2f}% "
                      f"sharpe_m={ps['sharpe_m']:5.2f} maxdd={ps['maxdd']*100:6.1f}%",
                      flush=True)
    (ROOT / "research" / "mlx_mtm.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/mlx_mtm.json", flush=True)


if __name__ == "__main__":
    main()
