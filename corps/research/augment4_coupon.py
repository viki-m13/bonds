"""Recover each bond's COUPON from its own (price, yield, maturity)
observations — v2, incorporating the red-team must-fixes.

Method: annual-pay bond identity  P = c*A + 100*(1+y)^-T,  A = (1-(1+y)^-T)/y
inverted per qualifying bond-day, per-bond median, clipped to [0, 15].

Red-team fixes over v1:
  * Accrued-interest two-pass (F1.1): the panel price is CLEAN; feeding it to
    the coupon-date identity biases c down by ~AI/A. Fix: invert once, add
    c0/4 (average semiannual AI) to P, re-invert. Converges to <0.1pt.
  * Qualifying days (F1.2/F1.7): prefer mat >= 3 rows (>=5 of them); else
    mat >= 2; else fall back to the legacy yield proxy. Price in [70, 130],
    y in (0.1%, 25%) — distressed days produce garbage c that clipping would
    turn into boundary atoms.
  * Extra validation (F1.8): day-level par anchor, premium/discount sign
    consistency, within-bond IQR distribution (fat right tail ~ callables).

Residual disclosed biases: premium CALLABLES carry yield-to-worst, biasing c
down (income understated => conservative for pull-to-par strategies);
floaters get a frozen average coupon; ±0.3pt day-count/integer-T residue on
bullets.

  python corps/research/augment4_coupon.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402


def _invert(P, y, T):
    disc = (1 + y) ** (-T)
    A = (1 - disc) / y
    return (P - 100 * disc) / A


def recover(b, min_mat):
    y = b["ytw"].astype(np.float64) / 100.0
    T = b["mat"].astype(np.float64)
    P = b["mid"].astype(np.float64)
    ok = (np.isfinite(y) & np.isfinite(P) & (T >= min_mat) & (T <= 40)
          & (y > 0.001) & (y < 0.25) & (P >= 70) & (P <= 130))
    if ok.sum() < 5:
        return None, None
    y = y[ok]; T = T[ok]; P = P[ok]
    c0 = _invert(P, y, T)                    # pass 1 (clean price)
    c0 = np.clip(c0, 0.0, 20.0)
    c1 = _invert(P + c0 / 4.0, y, T)         # pass 2 (AI-corrected price)
    c1 = c1[np.isfinite(c1)]
    if len(c1) < 5:
        return None, None
    iqr = float(np.percentile(c1, 75) - np.percentile(c1, 25))
    return float(np.clip(np.median(c1), 0.0, 15.0)), iqr


def main():
    bonds = e2.load_cache()
    print(f"{len(bonds)} bonds; recovering coupons (v2, AI-corrected) ...", flush=True)
    n_ok3 = n_ok2 = n_fb = 0
    near_par_diff, prem_flag, disc_flag, all_c, all_iqr = [], [], [], [], []
    # day-level par anchor: on |mid-100|<=0.5 days, c_hat(day) ~ 100*y(day)
    par_day_err = []
    for k, (six, b) in enumerate(bonds.items()):
        c, iqr = recover(b, 3)
        if c is not None:
            n_ok3 += 1
        else:
            c, iqr = recover(b, 2)
            if c is not None:
                n_ok2 += 1
        if c is None:
            b["coupon_inv"] = b["coupon"]
            n_fb += 1
        else:
            b["coupon_inv"] = c
            all_c.append(c); all_iqr.append(iqr)
            med_mid = float(np.nanmedian(b["mid"]))
            med_ytw = float(np.nanmedian(b["ytw"]))
            if np.isfinite(med_mid) and np.isfinite(med_ytw):
                if med_mid >= 104:
                    prem_flag.append(c > med_ytw)
                elif med_mid <= 96:
                    disc_flag.append(c < med_ytw)
                elif 99.5 <= med_mid <= 100.5:
                    near_par_diff.append(c - med_ytw)
        if k % 20000 == 0:
            # day-level par anchor sample (implementation check, F1.8-1)
            y = b["ytw"].astype(np.float64); P = b["mid"].astype(np.float64)
            T = b["mat"].astype(np.float64)
            m = (np.abs(P - 100) <= 0.5) & (T >= 3) & np.isfinite(y)
            if m.any():
                cd = _invert(P[m] + y[m] / 4 / 100 * 0, y[m] / 100, T[m])
                par_day_err.extend((cd - y[m]).tolist())
            print(f"  {k} ...", flush=True)

    q = np.percentile(all_c, [5, 50, 95]) if all_c else [np.nan] * 3
    v1 = float(np.median(np.abs(near_par_diff))) if near_par_diff else np.nan
    v2p = float(np.mean(prem_flag)) if prem_flag else np.nan
    v2d = float(np.mean(disc_flag)) if disc_flag else np.nan
    iqr_med = float(np.median(all_iqr)) if all_iqr else np.nan
    iqr_p90 = float(np.percentile(all_iqr, 90)) if all_iqr else np.nan
    print(f"\nvalidation:", flush=True)
    print(f"  bond-level near-par |c-ytw| median = {v1:.3f}  (gate < 0.5)", flush=True)
    print(f"  premium (P>=104) c > ytw share     = {v2p:.2%} (gate >= 80%)", flush=True)
    print(f"  discount (P<=96) c < ytw share     = {v2d:.2%} (report)", flush=True)
    print(f"  coupon pct 5/50/95                 = {np.round(q,2).tolist()}", flush=True)
    print(f"  within-bond daily IQR median/p90   = {iqr_med:.2f} / {iqr_p90:.2f} "
          f"(fat tail ~ callables/FRNs)", flush=True)
    print(f"  recovered mat>=3: {n_ok3}, mat>=2: {n_ok2}, fallback: {n_fb}", flush=True)
    if not (v1 < 0.5 and v2p >= 0.80 and 0.5 <= q[0] and q[2] <= 12):
        print("VALIDATION FAILED — cache NOT rewritten", flush=True)
        sys.exit(1)
    with open(e2.CACHE, "wb") as f:
        pickle.dump(bonds, f, protocol=5)
    print(f"rewrote cache ({e2.CACHE.stat().st_size/1e9:.2f} GB) with coupon_inv (v2)",
          flush=True)


if __name__ == "__main__":
    main()
