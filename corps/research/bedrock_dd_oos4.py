"""BEDROCK-V drawdown round 4c — the ONE-SHOT OOS batch (2016+), §8g.

Admitted by the disclosed 4c rule (committed before this run): T2 and T2uT4
at K=100 only. Params frozen from IS: q75 = 5.77% stress threshold, 20d
slope, NAV<90% of 1y high AND NAV<NAV[-20d] for the T4 leg, all lagged 1d.
Run once, report as printed.

  python corps/research/bedrock_dd_oos4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bedrock_v import (cl_fills, real_coupons, exit_lagged, build_cs_median,  # noqa: E402
                       gate_value, gate_issuer_curve)
from bedrock_dd_screen import stress_series  # noqa: E402
from bedrock_dd_screen4 import cap_sim, stats, concurrency, IS, OOS  # noqa: E402

K = 100


def main():
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    s_lo, s_arr, _ = stress_series(bonds)
    is_days = np.arange(IS[0], IS[1] + 1)
    iv = s_arr[np.clip(is_days - s_lo, 0, len(s_arr) - 1)]
    Q75 = float(np.nanquantile(iv, 0.75))       # frozen from IS
    print(f"FROZEN (IS) q75={Q75*100:.2f}%", flush=True)

    base_full = cl_fills(bonds, *(e2.D("2003-01-01"), OOS[1]))
    pipe_all = real_coupons(bonds, exit_lagged(
        bonds, gate_issuer_curve(bonds, issuers, gate_value(bonds, base_full, med))))
    fills = [f for f in pipe_all if OOS[0] <= f.entry_day <= OOS[1]]
    mx, p90 = concurrency(fills)
    print(f"\nOOS 2016+: {len(fills)} fills; concurrency max={mx} p90={p90}",
          flush=True)
    d0 = min(f.entry_day for f in fills); d1 = max(f.exit_day for f in fills)
    days = np.arange(d0, d1 + 1)
    rf_d = e2.load_rf(days) / 365.0

    i = np.clip(days - 1 - s_lo, 0, len(s_arr) - 1)
    j = np.clip(i - 20, 0, len(s_arr) - 1)
    a, b = s_arr[i], s_arr[j]
    t2 = np.isfinite(a) & np.isfinite(b) & (a > Q75) & (a > b)

    out = {"q75": Q75, "n": len(fills)}
    r = cap_sim(bonds, fills, K, rf_d, d0, d1)
    base = stats(r[0], r[1], r[2], f"K={K} base (no trigger)")
    out["base"] = {**base, **r[3]}

    # IS reference gains (from bedrock_dd_screen4/4b JSONs, for the 60% rule)
    IS_GAIN = {"T2": 0.045, "T2uT4": 0.068}
    for vname, kw in (("T2", {"halt": t2}),
                      ("T2uT4", {"halt": t2, "own_eq": True, "eq_slope": True})):
        r = cap_sim(bonds, fills, K, rf_d, d0, d1, **kw)
        st = stats(r[0], r[1], r[2], f"K={K} {vname}", base)
        info = r[3]
        gain = st["maxdd"] - base["maxdd"]
        adopt = (gain >= 0.60 * IS_GAIN[vname]
                 and base["cagr"] - st["cagr"] <= 0.02
                 and st["sharpe_m"] >= base["sharpe_m"])
        st["oos_dd_gain"] = gain
        st["retention_vs_is"] = gain / IS_GAIN[vname]
        st["adopt"] = bool(adopt)
        out[vname] = {**st, **info}
        print(f"      taken={info['taken']} halt-skip={info['skipped_halt']} "
              f"cash-skip={info['skipped_cash']} inv_mean={info['inv_mean']*100:.0f}%",
              flush=True)
        print(f"      OOS ddGain={gain*100:+.1f}pp = {gain/IS_GAIN[vname]*100:.0f}% "
              f"of IS gain -> {'ADOPT' if adopt else 'KILL'}", flush=True)

    p = ROOT / "research" / "bedrock_dd_oos4.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
