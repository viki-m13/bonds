"""ONE-SHOT out-of-sample validation (2016-2025) of the sleeves that survived
in-sample screening. This script is run ONCE. Whatever it prints is the result;
no re-tuning is permitted afterwards.

Survivors entering OOS (all specs frozen exactly as screened on 2003-2015):
  FLOWBACK-S  volume-confirmed fire-sale reversal   (IS excess +0.40%, p<0.001,
                                                     monotone in both knobs)
  DEBUT       new-issue concession                  (IS Sharpe 0.58)
  GRANITE-C   dislocation-reversion <=5y, 1/issuer  (IS Sharpe 0.86) [reference]
  combined    frozen rules: inv-vol 24m, 5% vol target, cash at T-bill

  python corps/research/oos_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2                                     # noqa: E402
from combine import save_fills, sleeve_monthly, combine  # noqa: E402
from sleeves_events import run_flowback, build_issuer_map, report  # noqa: E402
from sleeves_wave3 import run_debut                      # noqa: E402
from granite_experiments import sig_disc, gate_mat5, issuer_cap_filter  # noqa: E402

OOS_LO = e2.D("2016-01-01")
OOS_HI = e2.D("2025-03-31") - 455


def stats_for(bonds, fills, name):
    if not fills:
        print(f"  {name}: no fills", flush=True)
        return None
    r = e2.mtm_nav(bonds, fills)
    if r is None:
        return None
    ps = e2.perf_stats(*r)
    ps["n"] = len(fills)
    print(f"  {name:12} n={len(fills):6} cagr={ps['cagr']*100:+6.2f}% "
          f"sharpe_m={ps['sharpe_m']:5.2f} maxdd={ps['maxdd']*100:6.1f}% "
          f"vol={ps['vol_m_ann']*100:5.2f}%", flush=True)
    return ps


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    imap = build_issuer_map(bonds)
    print(f"loaded {len(bonds)} bonds\n", flush=True)
    print("=" * 68, flush=True)
    print("ONE-SHOT OUT-OF-SAMPLE TEST — 2016-2025 — specs frozen from IS",
          flush=True)
    print("=" * 68, flush=True)
    out = {}

    print("\n[FLOWBACK-S] frozen at 4x volume / 1.25pt drop:", flush=True)
    f_flow, c_flow = run_flowback(bonds, imap, OOS_LO, OOS_HI)
    out["flowback"] = report("OOS", f_flow, c_flow, bonds)
    save_fills(f_flow, ROOT / "research" / "fills_flowback_oos.json")

    print("\n[DEBUT] frozen spec:", flush=True)
    f_deb, c_deb = run_debut(bonds, imap, OOS_LO, OOS_HI)
    out["debut"] = report("OOS", f_deb, c_deb, bonds)
    save_fills(f_deb, ROOT / "research" / "fills_debut_oos.json")

    print("\n[GRANITE-C] reference (already published OOS, re-marked honestly):",
          flush=True)
    f_gran = e2.run_events(bonds, sig_disc(3.0), min_hold=365, max_hold=455,
                           date_lo=OOS_LO, date_hi=OOS_HI, extra_gate=gate_mat5)
    f_gran = issuer_cap_filter(f_gran, cap=1)
    ctl_g = e2.matched_control(bonds, f_gran, min_hold=365, max_hold=455,
                               extra_gate=gate_mat5)
    out["granite_c"] = report("OOS", f_gran, ctl_g, bonds)
    save_fills(f_gran, ROOT / "research" / "fills_granite_c_oos.json")

    print("\n[COMBINED] frozen rules (inv-vol 24m, 5% vol target, cash@T-bill):",
          flush=True)
    monthlies = {}
    for k, fl in [("granite_c", f_gran), ("flowback", f_flow), ("debut", f_deb)]:
        m = sleeve_monthly(bonds, fl)
        if m is not None and len(m) >= 24:
            monthlies[k] = m
    if len(monthlies) >= 2:
        M = pd.DataFrame(monthlies)
        print("  correlation matrix:", flush=True)
        print("   " + M.corr().round(3).to_string().replace("\n", "\n   "), flush=True)
        priors = {"granite_c": 0.40, "flowback": 0.35, "debut": 0.25}
        comb, series = combine(monthlies, priors)
        out["combined"] = comb
        print(f"  combined: cagr={comb['cagr']*100:+.2f}% "
              f"sharpe_m={comb['sharpe_m']:.2f} maxdd={comb['maxdd']*100:.1f}% "
              f"vol={comb['vol_ann']*100:.2f}%", flush=True)

    print("\n" + "=" * 68, flush=True)
    print("TARGET CHECK — Sharpe 3.0 / CAGR 10%:", flush=True)
    for k, v in out.items():
        if v and v.get("sharpe_m") is not None:
            hit = "MET" if (v["sharpe_m"] >= 3 and v["cagr"] >= 0.10) else "not met"
            print(f"  {k:12} sharpe={v['sharpe_m']:5.2f} cagr={v['cagr']*100:+6.2f}%"
                  f"  -> {hit}", flush=True)
    print("=" * 68, flush=True)

    (ROOT / "research" / "oos_results.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/oos_results.json", flush=True)


if __name__ == "__main__":
    main()
