"""Add the BEDROCK-V block to docs/granite_xl_data.json (merge, not
overwrite): monthly NAV series of the BEDROCK-V pipeline book (real coupons,
lagged recovery exits, depth weights — the audited conventions, same as the
page's GRANITE-XL series) aligned to the existing series dates, plus the
validation tables assembled from the committed result JSONs.

  python corps/research/bedrock_page_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bedrock_v import (cl_fills, real_coupons, exit_lagged, build_cs_median,  # noqa: E402
                       gate_value, gate_issuer_curve)
from combos import depth_of  # noqa: E402

DOCS = ROOT.parent / "docs"
MAXH = 455
FULL = (e2.D("2003-01-01"), e2.D("2025-03-31") - MAXH)
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()

    fills = real_coupons(bonds, exit_lagged(
        bonds, gate_issuer_curve(bonds, issuers,
                                 gate_value(bonds, cl_fills(bonds, *FULL), med))))
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    days, nav, daily = e2.mtm_nav(bonds, fills, weights=w)
    ps_full = e2.perf_stats(days, nav, daily)
    r = np.array([f.ret for f in fills])
    ps_full.update({"n": len(fills), "mean_ret": float(r.mean()),
                    "win": float((r > 0).mean()),
                    "mean_hold": float(np.mean([f.hold for f in fills]))})
    oo = [(f, wi) for f, wi in zip(fills, w) if OOS[0] <= f.entry_day <= OOS[1]]
    d2, n2, dl2 = e2.mtm_nav(bonds, [f for f, _ in oo], weights=[wi for _, wi in oo])
    ps_oos = e2.perf_stats(d2, n2, dl2)
    r2 = np.array([f.ret for f, _ in oo])
    ps_oos.update({"n": len(oo), "mean_ret": float(r2.mean()),
                   "win": float((r2 > 0).mean())})
    print(f"BEDROCK-V pipeline: full {ps_full['cagr']*100:+.2f}%/{ps_full['sharpe_m']:.2f} "
          f"| OOS {ps_oos['cagr']*100:+.2f}%/{ps_oos['sharpe_m']:.2f}", flush=True)

    # monthly NAV aligned to the existing page series
    data = json.loads((DOCS / "granite_xl_data.json").read_text())
    ts = pd.Series(nav, index=pd.to_datetime(days, unit="D"))
    mon = ts.resample("MS").first().dropna()
    mon_map = {d.strftime("%Y-%m-%d"): round(float(v), 4) for d, v in mon.items()}
    for p in data["series"]:
        data["series"][data["series"].index(p)] = {**p, "bv": mon_map.get(p["date"])}

    fin = json.loads((ROOT / "research" / "bedrock_v_final.json").read_text())
    live = json.loads((ROOT / "research" / "bedrock_v_live_row.json").read_text())
    oosj = json.loads((ROOT / "research" / "bedrock_oos_results.json").read_text())

    data["bedrock"] = {
        "full": ps_full, "oos": ps_oos,
        "oos_paired": {"baseline": oosj["v2_baseline"], "bv": oosj["v2_g1g4"],
                       "excess_baseline": oosj["v2_baseline_excess"],
                       "excess_bv": oosj["v2_g1g4_excess"]},
        "replay": {k: fin[k] for k in ("replay_position_full", "replay_position_oos",
                                       "replay_tight_full", "replay_tight_oos")},
        "live_row": live,
        "era": fin["era"], "perturb": {k: {q: v[q] for q in ("n", "mean", "cagr", "sharpe_m")}
                                       for k, v in fin["perturb"].items()},
        "slip": {k: fin[k] for k in ("slip_0.0", "slip_0.125", "slip_0.25", "slip_0.5")},
    }
    (DOCS / "granite_xl_data.json").write_text(json.dumps(data, default=float))
    print("wrote docs/granite_xl_data.json (+bedrock, series.bv)", flush=True)


if __name__ == "__main__":
    main()
