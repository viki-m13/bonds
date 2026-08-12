"""BEDROCK-V drawdown round 4b — T4 equity-slope halt and T2 u T4 union
(§8g addendum 4b, pre-registered).

T4: halt new entries while NAV < 90% of trailing 1y high AND NAV < NAV 20d
ago (lagged 1d). T2 u T4: also halt while tape stress > q75 & rising.
Same capital sim, K in {50,100}; gates vs same-K base; survivors get the
ONE OOS look.

  python corps/research/bedrock_dd_screen4b.py
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
    Q75 = float(np.nanquantile(iv, 0.75))
    print(f"FROZEN q75={Q75*100:.2f}%", flush=True)

    base_full = cl_fills(bonds, *(e2.D("2003-01-01"), OOS[1]))
    pipe_all = real_coupons(bonds, exit_lagged(
        bonds, gate_issuer_curve(bonds, issuers, gate_value(bonds, base_full, med))))
    out = {"q75": Q75}

    def t2_arr(days):
        i = np.clip(days - 1 - s_lo, 0, len(s_arr) - 1)
        j = np.clip(i - 20, 0, len(s_arr) - 1)
        a, b = s_arr[i], s_arr[j]
        return np.isfinite(a) & np.isfinite(b) & (a > Q75) & (a > b)

    def run_window(tag, lo, hi, admits=None):
        fills = [f for f in pipe_all if lo <= f.entry_day <= hi]
        mx, p90 = concurrency(fills)
        print(f"\n=== {tag}: {len(fills)} fills; concurrency max={mx} p90={p90} ===",
              flush=True)
        d0 = min(f.entry_day for f in fills); d1 = max(f.exit_day for f in fills)
        days = np.arange(d0, d1 + 1)
        rf_d = e2.load_rf(days) / 365.0
        res = {"n": len(fills)}
        variants = {
            "T4 eq-slope": {"own_eq": True, "eq_slope": True},
            "T2uT4": {"own_eq": True, "eq_slope": True, "halt": t2_arr(days)},
        }
        for K in (50, 100):
            b = cap_sim(bonds, fills, K, rf_d, d0, d1)
            base = stats(b[0], b[1], b[2], f"K={K} base (no trigger)")
            res[f"K{K}_base"] = {**base, **b[3]}
            for vname, kw in variants.items():
                if admits is not None and vname not in admits:
                    continue
                r = cap_sim(bonds, fills, K, rf_d, d0, d1, **kw)
                st = stats(r[0], r[1], r[2], f"K={K} {vname}", base)
                res[f"K{K}_{vname}"] = {**st, **r[3]}
                info = r[3]
                print(f"      taken={info['taken']} halt-skip={info['skipped_halt']} "
                      f"cash-skip={info['skipped_cash']} "
                      f"inv_mean={info['inv_mean']*100:.0f}%", flush=True)
        return res

    out["IS"] = run_window("IS 2003-2015", *IS)
    admits = set()
    for K in (50, 100):
        for v in ("T4 eq-slope", "T2uT4"):
            if out["IS"].get(f"K{K}_{v}", {}).get("admit"):
                admits.add(v)
    print(f"\nIS SURVIVORS: {sorted(admits) if admits else 'NONE'}", flush=True)
    if admits:
        out["OOS"] = run_window("OOS 2016+", *OOS, admits=admits)

    p = ROOT / "research" / "bedrock_dd_screen4b.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
