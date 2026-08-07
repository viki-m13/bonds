"""Build docs/granite_xl_data.json — the data backing for the GRANITE-XL
strategy page. Honest MTM NAV (marks at real mid prints) monthly series vs
LQD, stats for full/IS/OOS, era decomposition of the XL book, atlas excerpt,
and the validation trail.

  python corps/research/xl_page_data.py
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
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - 455)
ERAS = [("2004-2007", "2004-01-01", "2007-12-31"),
        ("2008-2009 GFC", "2008-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2021-2023", "2021-01-01", "2023-12-31"),
        ("2024-2025", "2024-01-01", "2025-03-31")]


def stats_for(bonds, fills, weights=None):
    r = e2.mtm_nav(bonds, fills, weights=weights)
    ps = e2.perf_stats(*r)
    rr = np.array([f.ret for f in fills])
    ps.update({"n": len(fills), "mean_ret": float(rr.mean()),
               "win": float((rr > 0).mean()),
               "mean_hold": float(np.mean([f.hold for f in fills]))})
    return ps, r


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    fills = xl_fills(bonds)
    w_all = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    print(f"XL fills: {len(fills)}", flush=True)

    full, (days, nav, daily) = stats_for(bonds, fills, w_all)
    fis = [f for f in fills if IS[0] <= f.entry_day <= IS[1]]
    wis = [w for f, w in zip(fills, w_all) if IS[0] <= f.entry_day <= IS[1]]
    foos = [f for f in fills if OOS[0] <= f.entry_day <= OOS[1]]
    woos = [w for f, w in zip(fills, w_all) if OOS[0] <= f.entry_day <= OOS[1]]
    is_s, _ = stats_for(bonds, fis, wis)
    oos_s, _ = stats_for(bonds, foos, woos)
    print(f"full: cagr={full['cagr']*100:+.2f}% sharpe={full['sharpe_m']:.2f}", flush=True)

    # 2026-08 audit: the same fills re-priced with recovered REAL coupons
    # (coupon_inv, augment4) — the panel's median-YTW carry proxy overstates
    # accrual on the discount/distressed bonds this book buys. See XL_AUDIT.md.
    def refill(f):
        c = float(bonds[f.six].get("coupon_inv", f.coupon))
        return e2.Fill(f.six, f.entry_day, f.entry_px, f.exit_day, f.exit_px, c, f.stale)
    audit = None
    if any("coupon_inv" in b for b in bonds.values()):
        ftrue = [refill(f) for f in fills]
        a_full, (days, nav, daily) = stats_for(bonds, ftrue, w_all)
        a_is, _ = stats_for(bonds, [refill(f) for f in fis], wis)
        a_oos, _ = stats_for(bonds, [refill(f) for f in foos], woos)
        audit = {"full": a_full, "is": a_is, "oos": a_oos}
        # the plotted growth-of-$1 series switches to the real-coupon NAV
        # (days/nav shadowed above) — the honest carry path
        print(f"audit (real coupons): full cagr={a_full['cagr']*100:+.2f}% "
              f"sharpe_m={a_full['sharpe_m']:.2f} | oos cagr={a_oos['cagr']*100:+.2f}% "
              f"sharpe_m={a_oos['sharpe_m']:.2f}", flush=True)

    # monthly NAV series vs LQD
    ts = pd.Series(nav, index=pd.to_datetime(days, unit="D"))
    mon = ts.resample("MS").first().dropna()
    lqd = (pd.read_csv(ROOT / "data" / "etf" / "LQD.csv.gz", parse_dates=["date"])
           .set_index("date")["adjclose"])
    lqd = lqd.reindex(mon.index, method="ffill").bfill()
    lqd_eq = lqd / lqd.iloc[0]
    series = [{"date": d.strftime("%Y-%m-%d"), "strat": round(float(v), 4),
               "lqd": round(float(l), 4)}
              for d, v, l in zip(mon.index, mon.values, lqd_eq.values)]
    lqd_full = (lqd_eq.iloc[-1]) ** (1 / ((mon.index[-1] - mon.index[0]).days / 365.25)) - 1
    lqd_dd = float((lqd_eq / lqd_eq.cummax() - 1).min())

    # era rows (per-trade, XL book; real coupons when available — audit)
    era_fills = [refill(f) for f in fills] if audit else fills
    era_rows = []
    for lab, lo, hi in ERAS:
        lo_d, hi_d = e2.D(lo), e2.D(hi)
        fl = [f for f in era_fills if lo_d <= f.entry_day <= hi_d]
        if len(fl) < 20:
            continue
        rr = np.array([f.ret for f in fl])
        era_rows.append({"era": lab, "n": len(fl), "mean": float(rr.mean()),
                         "win": float((rr > 0).mean())})

    atlas = json.loads((ROOT / "research" / "explore4_results.json").read_text())["atlas"]
    cl_diag = json.loads((ROOT / "research" / "cl_diagnostics.json").read_text())
    transfer = json.loads((ROOT.parent / "munis" / "research" / "results" /
                           "limit_transfer.json").read_text())
    kxl_p = ROOT.parent / "munis" / "research" / "results" / "keystone_xl.json"
    kxl = json.loads(kxl_p.read_text()) if kxl_p.exists() else None

    data = {"full": full, "is": is_s, "oos": oos_s, "series": series,
            "lqd": {"cagr": float(lqd_full), "maxdd": lqd_dd,
                    "total": float(lqd_eq.iloc[-1] - 1)},
            "era": era_rows, "atlas": atlas,
            "cl_era_excess": cl_diag.get("era"),
            "transfer": transfer, "keystone_xl": kxl, "audit": audit}
    (DOCS / "granite_xl_data.json").write_text(json.dumps(data, default=float))
    print("wrote docs/granite_xl_data.json", flush=True)


if __name__ == "__main__":
    main()
