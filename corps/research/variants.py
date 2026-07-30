"""GRANITE-C refinement experiments (IS-only), all frozen-rule, no free knobs:

  [V] vol-managed sizing (Moreira-Muir): monthly exposure w_t = min(1,
      var_target / realized_var(prev 6m)), var_target = the book's own
      full-IS variance. Raises Sharpe for many risk factors; costs CAGR.
      No leverage. Cash earns T-bill.
  [L] limit-entry discipline: take the entry only if the post-signal ask <=
      signal-day mid + 0.25 (never chase a bounced price).
  [C] pre-registered calendar tests (2 hypotheses, no fishing):
      H1 December entries (tax-loss selling) have higher mean return than
         non-December; H2 quarter-end-month entries higher than mid-quarter.

  python corps/research/variants.py
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
from combine import load_fills, sleeve_monthly  # noqa: E402
from granite_experiments import sig_disc, gate_mat5, issuer_cap_filter  # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")


def monthly_stats(m, label):
    days = m.index.values.astype("datetime64[D]").astype(np.int64)
    rf = pd.Series((1 + e2.load_rf(days) / 365) ** 30 - 1, index=m.index)
    ex = (m - rf).dropna()
    sh = float(ex.mean() / ex.std() * np.sqrt(12)) if ex.std() > 0 else None
    nav = (1 + m).cumprod()
    cagr = float(nav.iloc[-1] ** (12 / len(m)) - 1)
    mdd = float((nav / nav.cummax() - 1).min())
    print(f"  {label:28} cagr={cagr*100:+6.2f}% sharpe_m={sh:5.2f} "
          f"maxdd={mdd*100:6.1f}% vol={m.std()*np.sqrt(12)*100:5.2f}%", flush=True)
    return {"cagr": cagr, "sharpe_m": sh, "maxdd": mdd,
            "vol_ann": float(m.std() * np.sqrt(12))}


def vol_managed(m):
    """w_t = min(1, target_var / realized_var(prev 6 months)); target = full-IS
    variance of the unmanaged book (no free parameter)."""
    days = m.index.values.astype("datetime64[D]").astype(np.int64)
    rf = pd.Series((1 + e2.load_rf(days) / 365) ** 30 - 1, index=m.index)
    tgt = m.var()
    rv = m.rolling(6).var().shift(1)
    w = (tgt / rv).clip(upper=1.0).fillna(1.0)
    return w * m + (1 - w) * rf


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    gfills = load_fills(ROOT / "research" / "fills_granite_c_is.json")
    m = sleeve_monthly(bonds, gfills)

    print("\n[V] vol-managed GRANITE-C (frozen Moreira-Muir rule):", flush=True)
    out["base"] = monthly_stats(m, "GRANITE-C unmanaged")
    out["vol_managed"] = monthly_stats(vol_managed(m), "GRANITE-C vol-managed")

    print("\n[L] limit-entry discipline (ask <= signal mid + 0.25):", flush=True)
    med_cache = {}

    def sig3(b):
        med = b.get("med60")
        if med is None:
            return None
        med_cache[b["_six"]] = (b["mid"], med)
        return (b["s_px"] - med) <= -3.0

    fills_all = e2.run_events(bonds, sig3, min_hold=365, max_hold=455,
                              date_lo=IS_LO, date_hi=IS_HI, extra_gate=gate_mat5)
    fills_all = issuer_cap_filter(fills_all, cap=1)
    # apply limit filter: entry price <= signal-day mid + 0.25. We re-derive the
    # signal-day mid as the last mid at/before entry_day - 1.
    kept = []
    for f in fills_all:
        b = bonds[f.six]
        i = np.searchsorted(b["day"], f.entry_day, side="left") - 1
        if i >= 0 and np.isfinite(b["mid"][i]) and f.entry_px <= b["mid"][i] + 0.25:
            kept.append(f)
    print(f"  fills: {len(fills_all)} -> {len(kept)} after limit filter", flush=True)
    for tag, fl in [("all fills", fills_all), ("limit-entry", kept)]:
        r = np.array([f.ret for f in fl])
        print(f"  {tag:14} n={len(fl):5} mean={r.mean()*100:+.2f}% "
              f"win={(r>0).mean()*100:.0f}%", flush=True)
        mm = sleeve_monthly(bonds, fl)
        out[f"limit_{tag.split()[0]}"] = monthly_stats(mm, f"  {tag} MTM")

    print("\n[C] pre-registered calendar tests on GRANITE-C fills:", flush=True)
    ent = pd.to_datetime([f.entry_day for f in gfills], unit="D")
    ret = np.array([f.ret for f in gfills])
    dec = ent.month == 12
    qe = ent.month.isin([3, 6, 9, 12])
    for name, mask in [("H1 December", dec), ("H2 quarter-end", qe)]:
        a, bmask = ret[mask], ret[~mask]
        t = ((a.mean() - bmask.mean())
             / np.sqrt(a.var() / len(a) + bmask.var() / len(bmask)))
        print(f"  {name:16} n={mask.sum():5} mean={a.mean()*100:+.2f}% "
              f"vs other {bmask.mean()*100:+.2f}%  t={t:+.2f}", flush=True)
        out[name] = {"n": int(mask.sum()), "mean_in": float(a.mean()),
                     "mean_out": float(bmask.mean()), "t": float(t)}

    (ROOT / "research" / "variants_is.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/variants_is.json", flush=True)


if __name__ == "__main__":
    main()
