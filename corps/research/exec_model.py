"""Execution & slippage model for GRANITE-XL — can this actually be run live?

[S] Slippage sensitivity: haircut h on every fill (pay entry ask + h, receive
    exit bid - h) for h in {0, 1/8, 1/4, 1/2} pt. The backtest already pays
    the full printed bid-ask; h models COMPETITION for the print (you must
    outbid the actual customer to take their fill). Reports mean/trade, CAGR,
    Sharpe per h, and the breakeven h where CAGR drops to LQD's.
[C] Participation capacity: entry-day printed volume per traded name, position
    size at a 25% participation cap, deployable dollars per year, implied AUM
    at observed concurrency.
[O] Ops stats: fills/year, signals/day distribution, fill latency (from CL
    diagnostics), share of entries during crisis clusters.

Writes docs/exec_corp.json.

  python corps/research/exec_model.py
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
from flowml import xl_fills  # noqa: E402
from combos import depth_of  # noqa: E402

DOCS = ROOT.parent / "docs"


def haircut_fills(fills, h):
    out = []
    for f in fills:
        out.append(e2.Fill(f.six, f.entry_day, f.entry_px + h,
                           f.exit_day, max(f.exit_px - h, 1.0), f.coupon, f.stale))
    return out


def book_stats(bonds, fills, w):
    r = e2.mtm_nav(bonds, fills, weights=w)
    ps = e2.perf_stats(*r)
    rr = np.array([f.ret for f in fills])
    return {"mean_ret": float(rr.mean()), "win": float((rr > 0).mean()),
            "cagr": ps["cagr"], "sharpe_m": ps["sharpe_m"], "maxdd": ps["maxdd"]}


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    fills = xl_fills(bonds)
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    print(f"XL fills {len(fills)}", flush=True)

    out = {"slippage": []}
    for h in (0.0, 0.125, 0.25, 0.5):
        fh = haircut_fills(fills, h)
        s = book_stats(bonds, fh, w)
        s["h"] = h
        out["slippage"].append(s)
        print(f"  h={h:5.3f}  mean={s['mean_ret']*100:+6.2f}% cagr={s['cagr']*100:+6.2f}% "
              f"sharpe={s['sharpe_m']:.2f} maxdd={s['maxdd']*100:.1f}%", flush=True)
    lqd = (pd.read_csv(ROOT / "data" / "etf" / "LQD.csv.gz", parse_dates=["date"])
           .set_index("date")["adjclose"])
    yrs = (lqd.index[-1] - lqd.index[0]).days / 365.25
    lqd_cagr = float((lqd.iloc[-1] / lqd.iloc[0]) ** (1 / yrs) - 1)
    hs = [s["h"] for s in out["slippage"]]
    cs = [s["cagr"] for s in out["slippage"]]
    be = None
    for i in range(1, len(hs)):
        if cs[i] <= lqd_cagr <= cs[i - 1]:
            be = hs[i - 1] + (cs[i - 1] - lqd_cagr) / (cs[i - 1] - cs[i]) * (hs[i] - hs[i - 1])
            break
    out["breakeven_h_vs_lqd"] = be
    out["lqd_cagr"] = lqd_cagr
    print(f"  breakeven haircut vs LQD ({lqd_cagr*100:.2f}%): "
          f"{be:.2f} pt" if be else "  no breakeven within 0.5pt", flush=True)

    # capacity at 25% participation of entry-day printed volume
    entry_vol = []
    for f in fills:
        b = bonds[f.six]
        i = np.searchsorted(b["day"], f.entry_day)
        i = min(max(i, 0), len(b["day"]) - 1)
        v = float(np.nan_to_num(b["qv"][i]))
        entry_vol.append(v)          # $MM that day
    ev = np.array(entry_vol)
    part = 0.25
    pos = part * ev                  # $MM position at 25% participation
    span_days = (max(f.exit_day for f in fills) - min(f.entry_day for f in fills))
    concurrency = sum(f.hold for f in fills) / span_days
    annual_deploy = float(pos.sum() / (span_days / 365.25))
    out["capacity"] = {
        "participation": part,
        "median_pos_mm": float(np.median(pos)),
        "p25_pos_mm": float(np.percentile(pos, 25)),
        "p75_pos_mm": float(np.percentile(pos, 75)),
        "avg_concurrency": float(concurrency),
        "annual_deployable_mm": annual_deploy,
        "implied_aum_mm": float(np.median(pos) * concurrency),
    }
    print(f"  capacity: median pos ${np.median(pos):.2f}M @25% part., "
          f"concurrency {concurrency:.0f}, implied AUM ${np.median(pos)*concurrency:.0f}M",
          flush=True)

    ent_years = pd.Series([1970 + f.entry_day // 365 for f in fills])
    out["ops"] = {"fills_per_year": float(len(fills) / (span_days / 365.25)),
                  "max_fills_year": int(ent_years.value_counts().max()),
                  "min_fills_year": int(ent_years.value_counts().min())}
    (DOCS / "exec_corp.json").write_text(json.dumps(out, default=float))
    print("wrote docs/exec_corp.json", flush=True)


if __name__ == "__main__":
    main()
