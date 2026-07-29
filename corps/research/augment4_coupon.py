"""Fourth cache augmentation: recover each bond's COUPON from its own
(price, yield, maturity) observations.

The OSBAP daily panel omits the coupon; we had proxied it with the bond's
median yield, which understates income on premium bonds and overstates it on
discount bonds — a bias that any pull-to-par strategy inherits (see
NEW_STRATEGY_SEARCH.md §4).

But every (clean price P, yield y, maturity T) triple embeds the coupon via
the annual-pay bond identity:

    P = c*A(y,T) + 100*(1+y)^-T,   A = (1 - (1+y)^-T) / y
    =>  c = (P - 100*(1+y)^-T) / A

We invert per bond-day (T = integer `mat` >= 1 only; the T<1 rows are dropped
because A -> 0 amplifies noise) and take the per-bond MEDIAN across days,
clipped to [0, 15]. Known approximations (annual-pay vs semiannual, clean-vs-
dirty conventions, integer-T truncation, yield-to-worst on callables) are
noisy at the day level; the median across hundreds of days is far closer to
the true coupon than the yield proxy. Validation gates below must pass or the
augment aborts:

  V1  near-par bonds (median mid in [99.5, 100.5]): recovered coupon must
      track median ytw closely (|median diff| < 0.5).
  V2  premium bonds (mid >= 104): recovered coupon must EXCEED median ytw for
      >= 80% of bonds (the whole point of the fix).
  V3  distribution: 5th-95th pct of recovered coupons within [0.5, 12].

Writes b["coupon_inv"] (recovered) alongside b["coupon"] (legacy proxy, kept
so published GRANITE numbers remain reproducible).

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


def recover(b):
    y = b["ytw"].astype(np.float64) / 100.0
    T = b["mat"].astype(np.float64)
    P = b["mid"].astype(np.float64)
    ok = (np.isfinite(y) & np.isfinite(P) & (T >= 1) & (T <= 40)
          & (y > 0.001) & (y < 0.5) & (P > 40) & (P < 160))
    if ok.sum() < 5:
        return None, 0
    y = y[ok]; T = T[ok]; P = P[ok]
    disc = (1 + y) ** (-T)
    A = (1 - disc) / y
    c = (P - 100 * disc) / A
    c = c[np.isfinite(c)]
    if len(c) < 5:
        return None, 0
    return float(np.clip(np.median(c), 0.0, 15.0)), int(len(c))


def main():
    bonds = e2.load_cache()
    print(f"{len(bonds)} bonds; recovering coupons ...", flush=True)
    stats = {"n_ok": 0, "n_fallback": 0}
    near_par_diff, premium_flag, all_c = [], [], []
    for k, (six, b) in enumerate(bonds.items()):
        c, nobs = recover(b)
        med_ytw = float(np.nanmedian(b["ytw"])) if np.isfinite(b["ytw"]).any() else np.nan
        if c is None:
            b["coupon_inv"] = b["coupon"]      # fallback to legacy proxy
            stats["n_fallback"] += 1
        else:
            b["coupon_inv"] = c
            stats["n_ok"] += 1
            all_c.append(c)
            med_mid = float(np.nanmedian(b["mid"]))
            if np.isfinite(med_mid) and np.isfinite(med_ytw):
                if 99.5 <= med_mid <= 100.5:
                    near_par_diff.append(c - med_ytw)
                elif med_mid >= 104:
                    premium_flag.append(c > med_ytw)
        if k % 10000 == 0:
            print(f"  {k} ...", flush=True)

    # validation gates
    v1 = float(np.median(np.abs(near_par_diff))) if near_par_diff else np.nan
    v2 = float(np.mean(premium_flag)) if premium_flag else np.nan
    q = np.percentile(all_c, [5, 50, 95]) if all_c else [np.nan] * 3
    print(f"\nvalidation: V1 near-par |c - ytw| median = {v1:.3f} "
          f"(n={len(near_par_diff)}, gate < 0.5)", flush=True)
    print(f"            V2 premium-bond c > ytw share = {v2:.2%} "
          f"(n={len(premium_flag)}, gate >= 80%)", flush=True)
    print(f"            V3 coupon pct 5/50/95 = {np.round(q,2).tolist()} "
          f"(gate within [0.5, 12])", flush=True)
    print(f"            recovered {stats['n_ok']}, fallback {stats['n_fallback']}",
          flush=True)
    if not (v1 < 0.5 and v2 >= 0.80 and 0.5 <= q[0] and q[2] <= 12):
        print("VALIDATION FAILED — cache NOT rewritten", flush=True)
        sys.exit(1)
    with open(e2.CACHE, "wb") as f:
        pickle.dump(bonds, f, protocol=5)
    print(f"rewrote cache ({e2.CACHE.stat().st_size/1e9:.2f} GB) with coupon_inv",
          flush=True)


if __name__ == "__main__":
    main()
