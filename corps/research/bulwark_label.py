"""BULWARK step 1 — a DEFAULT/CREDIT-LOSS labeller for the corp tape, built on
the audited engine2 cache (same bonds, same conventions as GRANITE/BEDROCK).

The tape has no default flag, so the label must be inferred and DISCLOSED.
Signatures used (all point-in-time observable):

  DISTRESS DAY : mid < 70 AND cs > 0.20 (2000bp) AND mat >= 1  -- a price/yield
                 combination that only occurs when the market prices material
                 default risk. Both legs required (guards data glitches and
                 deep-discount/zero-coupon bonds, whose price is low but whose
                 spread is not).
  BUST (bond)  : the bond's LAST 10 prints median mid < 55 while >1.25y of
                 maturity remains -- it left the tape in distress rather than
                 maturing, being called, or simply going quiet near par.
  GOOD EXIT    : last prints >= 85, or terminal maturity <= 1.25y (OSBAP
                 truncates coverage at ~1.0y, so that is coverage, not credit).

Reported: per-year new-distress counts (must spike 2008-09, 2015-16, 2020 if
the labeller is measuring credit and not noise), bust counts, and the
issuer-level (CUSIP6) rollup.

  python corps/research/bulwark_label.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402

PX_DISTRESS, CS_DISTRESS = 70.0, 0.20
PX_BUST, TAIL_N, MAT_LEFT = 55.0, 10, 1.25


def label(bonds):
    rows = []
    for six, b in bonds.items():
        mid, cs, mat, day = b["mid"], b["cs"], b["mat"], b["day"]
        ok = ~np.isnan(mid)
        if ok.sum() < 10:
            continue
        dist = ok & (mid < PX_DISTRESS) & np.isfinite(cs) & (cs > CS_DISTRESS) & (mat >= 1)
        first_d = int(day[np.argmax(dist)]) if dist.any() else None
        idx = np.flatnonzero(ok)
        tail = idx[-TAIL_N:]
        term_px = float(np.median(mid[tail]))
        term_mat = float(mat[idx[-1]])
        bust = bool(term_px < PX_BUST and term_mat > MAT_LEFT)
        rows.append({"six": six, "cusip6": six[:6], "n_obs": int(ok.sum()),
                     "first_day": int(day[idx[0]]), "last_day": int(day[idx[-1]]),
                     "ever_distress": bool(dist.any()), "first_distress": first_d,
                     "term_px": term_px, "term_mat": term_mat, "bust": bust,
                     "min_px": float(np.nanmin(mid)), "max_cs": float(np.nanmax(cs))})
    return pd.DataFrame(rows)


def main():
    bonds = e2.load_cache()
    print(f"loaded {len(bonds)} bonds", flush=True)
    df = label(bonds)
    df["first_distress_dt"] = pd.to_datetime(df["first_distress"], unit="D")
    df["last_dt"] = pd.to_datetime(df["last_day"], unit="D")
    print(f"\n{len(df):,} bonds labelled")
    print(f"  ever distressed : {df.ever_distress.sum():,} "
          f"({df.ever_distress.mean()*100:.2f}%)")
    print(f"  bust (terminal) : {df.bust.sum():,} ({df.bust.mean()*100:.2f}%)")
    print(f"  issuers (CUSIP6): {df.cusip6.nunique():,}; with a bust: "
          f"{df[df.bust].cusip6.nunique():,}")

    print("\n[validation] NEW distress events by year "
          "(expect spikes 2008-09, 2015-16, 2020):", flush=True)
    yr = df[df.ever_distress].first_distress_dt.dt.year.value_counts().sort_index()
    ybust = df[df.bust].last_dt.dt.year.value_counts().sort_index()
    for y in range(2003, 2026):
        n, nb = int(yr.get(y, 0)), int(ybust.get(y, 0))
        bar = "#" * int(min(n, 600) / 12)
        print(f"  {y}  distress {n:5}  bust-exit {nb:4}  {bar}", flush=True)

    out = ROOT / "research" / "bulwark_labels.parquet"
    df.to_parquet(out)
    print(f"\nwrote {out}", flush=True)
    summ = {"n_bonds": len(df), "ever_distress": int(df.ever_distress.sum()),
            "bust": int(df.bust.sum()), "issuers": int(df.cusip6.nunique()),
            "bust_issuers": int(df[df.bust].cusip6.nunique()),
            "by_year_distress": {int(k): int(v) for k, v in yr.items()},
            "by_year_bust": {int(k): int(v) for k, v in ybust.items()}}
    (ROOT / "research" / "bulwark_labels.json").write_text(json.dumps(summ))


if __name__ == "__main__":
    main()
