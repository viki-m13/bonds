"""ANCHOR — short-dated money-good paper bought at the ask and HELD TO
REDEMPTION.

The thesis is structural, not predictive. Every strategy screened so far pays
the bid-ask TWICE (buy at ask, sell at bid), which is what killed the
short-horizon sleeves. A bond held to maturity pays it ONCE: the exit is
redemption at par, not a sale into a dealer's bid. Simultaneously, a bond
approaching maturity has a terminal value pinned at 100, so its mark
volatility collapses as it ages — high Sharpe by construction, if (and only
if) the credit is money-good.

This is the cash-plus / short-duration credit book that real desks run. The
honest question is not whether it makes money (carry is mechanical) but
whether default losses eat the carry, and what Sharpe/CAGR it actually
delivers net.

REDEMPTION ASSUMPTION (the one material assumption; disclosed and measured).
The panel has no redemption record — a bond simply stops printing, whether it
matured, was called, or defaulted. We treat a bond as redeemed at par only if
its final print is >= 90 with credit spread <= 5%, and it never printed below
80 in its final year. Anything else exits at its LAST OBSERVED BID, taking the
realized loss. Sensitivity to this assumption is reported: results are shown
with par redemption and with a punitive last-bid-only exit (no par credit at
all), which brackets the truth.

  python corps/research/anchor.py
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

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")


def classify_ending(b):
    """(redeemed_at_par, last_bid_day, last_bid_px) using only the bond's own
    tape. Conservative: par credit requires a healthy final print."""
    day = b["day"]; mid = b["mid"]
    pm = ~np.isnan(b["p_px"])
    if not pm.any():
        return False, None, None
    p_day = day[pm]; p_px = b["p_px"][pm]
    last_bid_day, last_bid_px = int(p_day[-1]), float(p_px[-1])
    last_mid = mid[-1] if np.isfinite(mid[-1]) else last_bid_px
    last_cs = b["cs"][-1]
    j = np.searchsorted(day, day[-1] - 365)
    min_final_year = np.nanmin(mid[j:]) if len(mid[j:]) else np.nan
    healthy = (np.isfinite(last_mid) and last_mid >= 90
               and np.isfinite(last_cs) and last_cs <= 0.05
               and (not np.isfinite(min_final_year) or min_final_year >= 80))
    return bool(healthy), last_bid_day, last_bid_px


def run_anchor(bonds, lo, hi, max_mat=2, cs_lo=0.01, cs_hi=0.05,
               par_redemption=True, min_px=90.0):
    """Buy at the first ask after the bond enters the <=max_mat maturity bucket
    while money-good; hold to the end of its tape."""
    fills = []
    diag = {"n_par": 0, "n_lastbid": 0}
    for six, b in bonds.items():
        day = b["day"]; mat = b["mat"]; mid = b["mid"]; cs = b["cs"]
        if len(day) < 20:
            continue
        elig = (b["elig"] & (mat <= max_mat) & (mat >= 1)
                & (cs >= cs_lo) & (cs <= cs_hi) & (mid >= min_px) & (mid <= 105))
        idx = np.flatnonzero(elig)
        if not len(idx):
            continue
        i = int(idx[0])
        # no-distress history gate (point-in-time: looks only backwards)
        j0 = np.searchsorted(day, day[i] - 365)
        hist = mid[j0:i + 1]
        if len(hist) and np.nanmin(hist) < 80:
            continue
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        if not len(s_day):
            continue
        j = np.searchsorted(s_day, day[i], side="right")
        if j >= len(s_day) or s_day[j] - day[i] > 7:
            continue
        ed = int(s_day[j]); ep = float(s_px[j])
        if ed < lo or ed > hi or ep <= 0:
            continue
        redeemed, lb_day, lb_px = classify_ending(b)
        if lb_day is None or lb_day <= ed:
            continue
        if par_redemption and redeemed:
            xd, xp = int(day[-1]), 100.0
            diag["n_par"] += 1
        else:
            xd, xp = lb_day, lb_px
            diag["n_lastbid"] += 1
        if xd <= ed:
            continue
        fills.append(e2.Fill(six, ed, ep, xd, xp, b["coupon"], not redeemed))
    return fills, diag


def report(name, bonds, fills, diag=None):
    if not fills:
        print(f"  {name}: no fills", flush=True)
        return None
    r = np.array([f.ret for f in fills])
    hold = np.mean([f.hold for f in fills])
    days, nav, daily = e2.mtm_nav(bonds, fills)
    ps = e2.perf_stats(days, nav, daily)
    ps["n"] = len(fills)
    ps["mean_ret"] = float(r.mean())
    ps["win_rate"] = float((r > 0).mean())
    ps["mean_hold"] = float(hold)
    if diag:
        ps.update(diag)
    print(f"  {name:26} n={len(fills):6} win={(r>0).mean()*100:3.0f}% "
          f"mean={r.mean()*100:+6.2f}% hold={hold:4.0f}d", flush=True)
    print(f"    MTM cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:5.2f} "
          f"maxdd={ps['maxdd']*100:6.1f}% vol={ps['vol_m_ann']*100:5.2f}%", flush=True)
    return ps


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds\n", flush=True)
    out = {}

    print("[ANCHOR] IS 2003-2015 — maturity bucket sweep (par redemption):", flush=True)
    for mm in (1, 2, 3):
        f, d = run_anchor(bonds, IS_LO, IS_HI, max_mat=mm)
        out[f"anchor_mat{mm}"] = report(f"<={mm}y", bonds, f, d)

    print("\n[ANCHOR] credit band sweep (<=2y, par redemption):", flush=True)
    for lo_c, hi_c, tag in [(0.005, 0.02, "cs 0.5-2% (IG)"),
                            (0.01, 0.05, "cs 1-5% (base)"),
                            (0.03, 0.08, "cs 3-8% (crossover/HY)")]:
        f, d = run_anchor(bonds, IS_LO, IS_HI, max_mat=2, cs_lo=lo_c, cs_hi=hi_c)
        out[f"anchor_{tag}"] = report(tag, bonds, f, d)

    print("\n[ANCHOR] REDEMPTION-ASSUMPTION SENSITIVITY (<=2y, cs 1-5%):", flush=True)
    f, d = run_anchor(bonds, IS_LO, IS_HI, max_mat=2, par_redemption=True)
    out["anchor_par"] = report("par redemption", bonds, f, d)
    f2, d2 = run_anchor(bonds, IS_LO, IS_HI, max_mat=2, par_redemption=False)
    out["anchor_nopar"] = report("last-bid only (punitive)", bonds, f2, d2)
    if out.get("anchor_par") and out.get("anchor_nopar"):
        print(f"    par-credited fills: {d['n_par']} / {d['n_par']+d['n_lastbid']} "
              f"({100*d['n_par']/max(d['n_par']+d['n_lastbid'],1):.0f}%)", flush=True)

    from combine import save_fills
    save_fills(f, ROOT / "research" / "fills_anchor_is.json")
    (ROOT / "research" / "anchor_is.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/anchor_is.json", flush=True)


if __name__ == "__main__":
    main()
