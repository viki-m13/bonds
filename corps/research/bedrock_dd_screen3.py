"""BEDROCK-V drawdown round 3b — SH state-contingent index hedge, IS SCREEN
(§8f addendum, pre-registered).

State: CRISIS = 20d tape stress > threshold (q90 or q75, IS-frozen);
RISING = stress > stress 20 calendar days ago. State lagged 1 day.
Hedge: subtract h * (etf_ret - rf) from the book's daily while state ON.

Diagnostic first: share of each IS episode's peak-to-trough fall that occurs
state-ON — if the fall precedes the state, SH is dead by construction.

  python corps/research/bedrock_dd_screen3.py
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
from bedrock_dd_screen import stress_series, book, report  # noqa: E402
from bedrock_dd_diag import etf_daily, episodes, ds  # noqa: E402

IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))


def main():
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    s_lo, s_arr, _ = stress_series(bonds)

    base_f = cl_fills(bonds, *IS)
    pipe = real_coupons(bonds, exit_lagged(
        bonds, gate_issuer_curve(bonds, issuers, gate_value(bonds, base_f, med))))
    days, nav, daily = book(bonds, pipe)
    out = {}
    print("\n[IS base]", flush=True)
    out["base"] = report(days, nav, daily, "base", n=len(pipe))
    base = out["base"]
    rf = e2.load_rf(days) / 365.0

    # daily stress + slope, aligned to book days, LAGGED one day
    idx = np.clip(days - s_lo, 0, len(s_arr) - 1)
    st_d = s_arr[idx]
    prev = np.clip(idx - 20, 0, len(s_arr) - 1)
    ris_d = np.isfinite(st_d) & np.isfinite(s_arr[prev]) & (st_d > s_arr[prev])
    is_days = np.arange(IS[0], IS[1] + 1)
    iv = s_arr[np.clip(is_days - s_lo, 0, len(s_arr) - 1)]
    Q90 = float(np.nanquantile(iv, 0.90)); Q75 = float(np.nanquantile(iv, 0.75))
    out["q90"], out["q75"] = Q90, Q75
    print(f"FROZEN: q90={Q90*100:.2f}% q75={Q75*100:.2f}%", flush=True)

    states = {
        "q90&rising": (st_d > Q90) & ris_d,
        "q90": st_d > Q90,
        "q75&rising": (st_d > Q75) & ris_d,
    }
    for k in states:                       # lag one day (no lookahead)
        s = np.roll(states[k], 1); s[0] = False
        states[k] = s

    # ---- diagnostic: episode fall coverage by state -------------------------
    print("\n[diag] share of each episode's fall occurring state-ON "
          "(log-return share):", flush=True)
    eps = episodes(days, nav)
    out["diag"] = {}
    lr = np.log(1 + daily)
    for e in eps:
        m_fall = (days > e["peak_day"]) & (days <= e["trough_day"])
        tot = lr[m_fall].sum()
        row = {"depth": e["depth"], "fall_logret": float(tot)}
        for k, s in states.items():
            row[k] = float(lr[m_fall & s].sum() / tot) if tot != 0 else None
        out["diag"][ds(e["peak_day"])] = row
        cov = "  ".join(f"{k}:{row[k]*100:5.1f}%" for k in states)
        print(f"  {ds(e['peak_day'])} ({e['depth']*100:5.1f}%)  {cov}", flush=True)

    # ---- SH variants --------------------------------------------------------
    print("\n[SH] state-contingent hedge:", flush=True)
    for etf in ("HYG", "LQD"):
        er = etf_daily(etf, days)
        has = np.zeros(len(days), bool)
        df = pd.read_csv(ROOT / "data" / "etf" / f"{etf}.csv.gz",
                         parse_dates=["date"])
        first = np.datetime64(df["date"].iloc[0], "D").astype(np.int64)
        has = days >= first
        for sk, s in states.items():
            on = s & has
            for h in (0.5, 1.0):
                dl = daily - np.where(on, h * (er - rf), 0.0)
                switches = int(np.abs(np.diff(on.astype(int))).sum())
                tag = f"{etf} h={h} {sk}"
                st = report(days, np.cumprod(1 + dl), dl, tag, base)
                st["on_share"] = float(on.mean()); st["switches"] = switches
                out[f"SH_{etf}_{h}_{sk}"] = st

    p = ROOT / "research" / "bedrock_dd_screen3.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
