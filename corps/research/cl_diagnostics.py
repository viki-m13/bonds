"""GRANITE-CL diagnostics (IS-only except the era decomposition of the
already-spent OOS look, which is pure reporting — the spec stays frozen).

  [A] cap-parameter monotonicity (IS): cap in {0.0, 0.25, 0.5, 1.0, none}.
      Pre-registered expectation: per-trade mean and excess should tighten
      monotonically as the cap tightens (the cap is an implicit depth filter).
      Non-monotone => the +0.25 choice is suspect and gets flagged.
  [B] era robustness: excess vs limit-capped control per era (full sample —
      reporting; includes the spent OOS years for transparency).
  [C] liquidity quartiles (IS): does the CL edge survive in the most liquid
      names (capacity)?
  [D] fill latency: days from signal to accepted fill; entry-price vs
      signal-mid distribution (how deep are accepted fills really?).

  python corps/research/cl_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from granite_experiments import sig_disc, gate_mat5, issuer_cap_filter  # noqa: E402
from oos2 import limit_filter, limit_control  # noqa: E402

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
FULL = (e2.D("2003-01-01"), e2.D("2025-03-31") - MAXH)
ERAS = [("2004-2007", "2004-01-01", "2007-12-31"),
        ("2008-2009 GFC", "2008-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020 COVID", "2020-01-01", "2020-12-31"),
        ("2021-2023", "2021-01-01", "2023-12-31")]


def base_fills(bonds, lo, hi):
    f = e2.run_events(bonds, sig_disc(3.0), min_hold=365, max_hold=MAXH,
                      date_lo=lo, date_hi=hi, extra_gate=gate_mat5)
    return issuer_cap_filter(f, cap=1)


def line(tag, s):
    print(f"  {tag:16} n={s.get('n',0):5} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+6.2f}% "
          f"excess={s.get('excess_vs_control',0)*100:+5.2f}% "
          f"p={s.get('excess_p_boot',1):.3f}", flush=True)


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    print("\n[A] cap monotonicity (IS):", flush=True)
    fis = base_fills(bonds, *IS)
    for cap in (0.0, 0.25, 0.5, 1.0, None):
        fl = limit_filter(bonds, fis, cap=cap) if cap is not None else fis
        ctl = limit_control(bonds, fl, *IS, cap=cap if cap is not None else 1e9)
        s = e2.summarize(fl, control=ctl)
        out[f"cap_{cap}"] = s
        line(f"cap={cap}", s)
        if fl:
            r = e2.mtm_nav(bonds, fl)
            if r:
                ps = e2.perf_stats(*r)
                s["mtm"] = ps
                print(f"    MTM cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:.2f} "
                      f"maxdd={ps['maxdd']*100:.1f}%", flush=True)

    print("\n[B] era robustness (cap=0.25, full sample — reporting):", flush=True)
    ffull = limit_filter(bonds, base_fills(bonds, *FULL), cap=0.25)
    out["era"] = {}
    for lab, lo, hi in ERAS:
        lo_d, hi_d = e2.D(lo), e2.D(hi)
        fl = [f for f in ffull if lo_d <= f.entry_day <= hi_d]
        if not fl:
            continue
        ctl = limit_control(bonds, fl, lo_d, hi_d)
        s = e2.summarize(fl, control=ctl)
        out["era"][lab] = s
        line(lab, s)

    print("\n[C] liquidity quartiles (IS, cap=0.25):", flush=True)
    fcl = limit_filter(bonds, fis, cap=0.25)
    qv = []
    for f in fcl:
        b = bonds[f.six]
        i = min(max(np.searchsorted(b["day"], f.entry_day), 0), len(b["day"]) - 1)
        qv.append(float(b["qv90"][i]))
    qv = np.array(qv)
    qs = np.nanquantile(qv, [0.25, 0.5, 0.75])
    out["liquidity"] = {}
    for k, tag in enumerate(["Q1 least", "Q2", "Q3", "Q4 most"]):
        m = (np.searchsorted(qs, qv, side="right") == k)
        fl = [f for f, mm in zip(fcl, m) if mm]
        if not fl:
            continue
        ctl = limit_control(bonds, fl, *IS)
        s = e2.summarize(fl, control=ctl)
        out["liquidity"][tag] = s
        line(tag, s)

    print("\n[D] fill latency & effective depth (IS, cap=0.25):", flush=True)
    lat, eff = [], []
    for f in fcl:
        b = bonds[f.six]
        i = np.searchsorted(b["day"], f.entry_day, side="left") - 1
        if i >= 0 and np.isfinite(b["mid"][i]) and np.isfinite(b["med60"][i]):
            eff.append(float(b["med60"][i] - f.entry_px))
        # latency: entry_day minus most recent signal day <= entry
        sig = (b["s_px"] - b["med60"]) <= -3.0
        gated = sig & b["elig"] & (b["mat"] <= 5)
        j = np.searchsorted(b["day"], f.entry_day, side="left") - 1
        while j >= 0 and not gated[j]:
            j -= 1
        if j >= 0:
            lat.append(int(f.entry_day - b["day"][j]))
    out["latency"] = {"median_days": float(np.median(lat)),
                      "p90_days": float(np.percentile(lat, 90))}
    out["eff_depth"] = {"median_pts": float(np.median(eff)),
                        "p25": float(np.percentile(eff, 25)),
                        "p75": float(np.percentile(eff, 75))}
    print(f"  latency: median {out['latency']['median_days']:.0f}d, "
          f"p90 {out['latency']['p90_days']:.0f}d", flush=True)
    print(f"  effective depth (med60 - fill): median {out['eff_depth']['median_pts']:.2f} "
          f"pts [{out['eff_depth']['p25']:.2f}, {out['eff_depth']['p75']:.2f}]", flush=True)

    (ROOT / "research" / "cl_diagnostics.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/cl_diagnostics.json", flush=True)


if __name__ == "__main__":
    main()
