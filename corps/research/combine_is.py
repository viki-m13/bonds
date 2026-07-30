"""Combination test on the IS window: do the surviving sleeves diversify?

Surviving sleeves after IS screening:
  FLOWBACK-S  fire-sale reversal        (excess +0.40% p<0.001, monotone knobs)
  DEBUT       new-issue concession      (excess +0.15% p=0.004, best sleeve Sharpe)
  GRANITE-C   dislocation-reversion, <=5y, 1 position/issuer (the existing book)

Generates GRANITE-C IS fills, then runs the frozen combination rules
(inverse-vol on trailing 24m, 5% vol target, cash at T-bill) and reports the
correlation matrix — the number that decides whether combination can lift Sharpe.

  python corps/research/combine_is.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2                       # noqa: E402
from combine import save_fills, load_fills, sleeve_monthly, combine  # noqa: E402
from granite_experiments import sig_disc, gate_mat5, issuer_cap_filter  # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")


def granite_is_fills(bonds):
    p = ROOT / "research" / "fills_granite_c_is.json"
    if p.exists():
        return load_fills(p)
    fills = e2.run_events(bonds, sig_disc(3.0), min_hold=365, max_hold=455,
                          date_lo=IS_LO, date_hi=IS_HI, extra_gate=gate_mat5)
    fills = issuer_cap_filter(fills, cap=1)
    save_fills(fills, p)
    return fills


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)

    print("building GRANITE-C IS fills ...", flush=True)
    gfills = granite_is_fills(bonds)
    print(f"  GRANITE-C: {len(gfills)} fills", flush=True)

    sleeves = {"granite_c": gfills}
    for name, path in [("flowback", "fills_flowback_is.json"),
                       ("debut", "fills_debut_is.json")]:
        p = ROOT / "research" / path
        if p.exists():
            sleeves[name] = load_fills(p)
            print(f"  {name}: {len(sleeves[name])} fills", flush=True)

    monthlies = {}
    for k, f in sleeves.items():
        m = sleeve_monthly(bonds, f)
        if m is None or len(m) < 24:
            print(f"  {k}: insufficient history", flush=True)
            continue
        monthlies[k] = m
        rf = e2.load_rf(m.index.values.astype("datetime64[D]").astype(np.int64))
        rfm = pd.Series((1 + rf / 365) ** 30 - 1, index=m.index)
        ex = (m - rfm).dropna()
        sh = ex.mean() / ex.std() * np.sqrt(12) if ex.std() > 0 else None
        print(f"  {k:10} months={len(m):3} ann_ret={((1+m).prod()**(12/len(m))-1)*100:+6.2f}% "
              f"vol={m.std()*np.sqrt(12)*100:5.2f}% sharpe={sh:.2f}", flush=True)

    print("\ncorrelation matrix (monthly):", flush=True)
    M = pd.DataFrame(monthlies)
    print(M.corr().round(3).to_string(), flush=True)

    print("\ncombined book (frozen rules: inv-vol 24m, 5% vol target, cash@T-bill):",
          flush=True)
    priors = {"granite_c": 0.40, "flowback": 0.35, "debut": 0.25}
    out, series = combine(monthlies, priors)
    for k in ("cagr", "sharpe_m", "maxdd", "vol_ann"):
        v = out[k]
        print(f"  {k:9} = {v*100:+.2f}%" if k != "sharpe_m" else f"  {k:9} = {v:.2f}",
              flush=True)
    print(f"  avg weights: { {k: round(v,3) for k,v in out['avg_weights'].items()} }",
          flush=True)

    res = {"sleeves": {k: {"months": len(v),
                           "ann_ret": float((1 + v).prod() ** (12 / len(v)) - 1),
                           "vol_ann": float(v.std() * np.sqrt(12))}
                       for k, v in monthlies.items()},
           "corr": M.corr().round(3).to_dict(), "combined": out}
    (ROOT / "research" / "combine_is.json").write_text(json.dumps(res, default=float))
    print("\nwrote corps/research/combine_is.json", flush=True)


if __name__ == "__main__":
    main()
