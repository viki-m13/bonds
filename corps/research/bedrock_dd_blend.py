"""BL lever — KEYSTONE-XL (muni) x BEDROCK-V (corp) monthly-rebalanced blend
(§8e second addendum; pre-registered before running).

Uses the PUBLISHED monthly NAV series (docs/keystone_xl_curve.json "xl",
docs/granite_xl_data.json series[]["bv"]) on their common window. Reports
correlation, per-book stats, blends {50/50, 60/40, 70/30 corp}, plus the
T-bill-dilution reference at matched CAGR give-up. maxDD is monthly-resolution
(disclosed: daily DD is deeper; the corp book's daily-vs-monthly full-window
gap is about 4-6pp).

  python corps/research/bedrock_dd_blend.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def load_series():
    g = json.loads((DOCS / "granite_xl_data.json").read_text())
    k = json.loads((DOCS / "keystone_xl_curve.json").read_text())
    bv = pd.Series({pd.Timestamp(p["date"]): p["bv"] for p in g["series"]
                    if p.get("bv") is not None}).sort_index()
    xl = pd.Series({pd.Timestamp(p["date"]): p["xl"] for p in k["series"]
                    if p.get("xl") is not None}).sort_index()
    xl = xl.resample("MS").last()          # muni curve has some mid-month pts
    bv = bv.resample("MS").last()
    return bv, xl


def rf_monthly(idx):
    df = pd.read_csv(ROOT / "corps" / "data" / "etf" / "DGS3MO.csv",
                     parse_dates=["observation_date"])
    s = df.set_index("observation_date").iloc[:, 0] / 100.0
    s = s.resample("MS").mean().reindex(idx).ffill().fillna(0.0)
    return (1 + s) ** (1 / 12) - 1


def stats(r, rf, tag, base=None):
    nav = (1 + r).cumprod()
    yrs = len(r) / 12
    cagr = float(nav.iloc[-1] ** (1 / yrs) - 1)
    dd = float((nav / nav.cummax() - 1).min())
    ex = r - rf
    sh = float(ex.mean() / ex.std() * np.sqrt(12))
    row = {"cagr": cagr, "maxdd": dd, "sharpe_m": sh, "months": len(r)}
    verdict = ""
    if base is not None:
        gain = dd - base["maxdd"]
        ok = gain >= 0.08 and base["cagr"] - cagr <= 0.02 and sh >= base["sharpe_m"]
        row["admit"] = bool(ok)
        verdict = f"  ddGain={gain*100:+.1f}pp -> {'ADMIT' if ok else 'reject'}"
    print(f"  {tag:26} cagr={cagr*100:+6.2f}% sharpe_m={sh:5.2f} "
          f"dd={dd*100:6.1f}%{verdict}", flush=True)
    return row


def main():
    bv, xl = load_series()
    lo = max(bv.index[0], xl.index[0]); hi = min(bv.index[-1], xl.index[-1])
    bvr = bv.pct_change().loc[lo:hi].dropna()
    xlr = xl.pct_change().loc[lo:hi].dropna()
    idx = bvr.index.intersection(xlr.index)
    bvr, xlr = bvr[idx], xlr[idx]
    rf = rf_monthly(idx)
    corr = float(np.corrcoef(bvr, xlr)[0, 1])
    dn = bvr < 0
    corr_dn = float(np.corrcoef(bvr[dn], xlr[dn])[0, 1]) if dn.sum() > 5 else None
    print(f"common window {idx[0].date()} .. {idx[-1].date()}  ({len(idx)} mo)")
    print(f"monthly corr = {corr:+.2f}   corr in corp-down months = "
          f"{corr_dn:+.2f}" if corr_dn is not None else "", flush=True)
    out = {"window": [str(idx[0].date()), str(idx[-1].date())],
           "corr": corr, "corr_dn": corr_dn}

    print("\n[books on the common window]", flush=True)
    out["bv"] = stats(bvr, rf, "BEDROCK-V (corp) 100%")
    out["xl"] = stats(xlr, rf, "KEYSTONE-XL (muni) 100%")
    base = out["bv"]

    print("\n[BL blends, monthly rebalanced]", flush=True)
    for wc in (0.5, 0.6, 0.7):
        r = wc * bvr + (1 - wc) * xlr
        out[f"bl_{int(wc*100)}"] = stats(r, rf, f"{int(wc*100)}/{int((1-wc)*100)} corp/muni",
                                         base)

    print("\n[reference: T-bill dilution of BEDROCK-V]", flush=True)
    for wc in (0.5, 0.6, 0.7):
        r = wc * bvr + (1 - wc) * rf
        out[f"tb_{int(wc*100)}"] = stats(r, rf, f"{int(wc*100)}% BV + T-bills", base)

    print("\n[corr robustness — is the near-zero correlation window-stable?]",
          flush=True)
    out["subwin"] = {}
    for tag, lo2 in (("2016+ (corp OOS)", "2016-01-01"),
                     ("2020+", "2020-01-01"),
                     ("2023+ (both OOS)", "2023-01-01")):
        m = idx >= pd.Timestamp(lo2)
        if m.sum() < 18:
            continue
        c = float(np.corrcoef(bvr[m], xlr[m])[0, 1])
        r = 0.5 * bvr[m] + 0.5 * xlr[m]
        nav = (1 + r).cumprod()
        dd = float((nav / nav.cummax() - 1).min())
        bnav = (1 + bvr[m]).cumprod()
        bdd = float((bnav / bnav.cummax() - 1).min())
        out["subwin"][tag] = {"months": int(m.sum()), "corr": c,
                              "dd_5050": dd, "dd_bv": bdd}
        print(f"  {tag:18} n={m.sum():3} corr={c:+.2f}  dd 50/50 {dd*100:6.1f}% "
              f"vs BV {bdd*100:6.1f}%", flush=True)

    p = ROOT / "corps" / "research" / "bedrock_dd_blend.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
