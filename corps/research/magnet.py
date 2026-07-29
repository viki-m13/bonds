"""MAGNET — dislocation ENTRY + hold-to-redemption EXIT (IS-only screen).

The hybrid of the two structural findings of this program:
  - the only OOS-proven timing alpha is the dislocation entry (GRANITE), and
  - the only exit that does not pay the dealer's bid is redemption (ANCHOR),
so: buy >=k-pt dislocations in short-dated money-good paper at the ask, then
never sell — collect coupons (now the RECOVERED coupon, coupon_inv) and
redeem at par. The bid-ask is paid once; there is no exit-timing beta.

Redemption accounting (identical to ANCHOR, bracketed): par credit only if the
bond's final tape print is >=90 with cs<=5% and no sub-80 print in its final
year; else the position exits at the LAST OBSERVED BID. The punitive
last-bid-only variant is reported alongside.

Pre-registered gates (fixed before running):
  G1 excess vs matched control positive and MONOTONE in entry depth k=1,2,3,4
  G2 par-bracket AND punitive-bracket monthly Sharpe both above the GRANITE-C
     IS baseline (0.86) for the sleeve to claim a Sharpe improvement
  G3 maturity gradient <=1y / <=2y / <=3y reported; a pattern where the edge
     lives only in the least-marked bucket is treated as a stale-mark artifact

Control: random entry days on the SAME bonds passing the SAME gates (money-
good, maturity band, liquidity), same hold-to-redemption exit — isolates the
dislocation-entry timing from carry, universe drift, and the redemption
assumption itself.

  python corps/research/magnet.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from anchor import classify_ending  # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")
RNG = np.random.default_rng(31)


def money_good_gate(b, max_mat):
    day = b["day"]; mid = b["mid"]
    # trailing-365d min mid >= 80 (point-in-time, backward-looking)
    n = len(day)
    ok_hist = np.ones(n, bool)
    j0s = np.searchsorted(day, day - 365)
    run_min = np.full(n, np.nan)
    for i in range(n):
        seg = mid[j0s[i]:i + 1]
        if len(seg):
            run_min[i] = np.nanmin(seg)
    ok_hist = ~(run_min < 80)
    return (b["elig"] & (b["mat"] >= 1) & (b["mat"] <= max_mat)
            & (b["cs"] <= 0.05) & (mid >= 85) & ok_hist)


def run_magnet(bonds, lo, hi, depth=3.0, max_mat=3, par_redemption=True,
               n_ctl=8, use_inv_coupon=True):
    fills, ctl = [], []
    for six, b in bonds.items():
        day = b["day"]
        med60 = b.get("med60")
        if med60 is None or len(day) < 30:
            continue
        gate = money_good_gate(b, max_mat)
        if not gate.any():
            continue
        coupon = b["coupon_inv"] if use_inv_coupon else b["coupon"]
        sig = gate & ((b["s_px"] - med60) <= -depth)
        idx = np.flatnonzero(sig)
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        redeemed, lb_day, lb_px = classify_ending(b)
        if lb_day is None:
            continue
        if par_redemption and redeemed:
            xd, xp = int(day[-1]), 100.0
        else:
            xd, xp = lb_day, lb_px

        def fill_at(i):
            t = day[i]
            j = np.searchsorted(s_day, t, side="right")
            if j >= len(s_day) or s_day[j] - t > 7:
                return None
            ed = int(s_day[j]); ep = float(s_px[j])
            if ed < lo or ed > hi or xd <= ed or ep <= 0:
                return None
            return e2.Fill(six, ed, ep, xd, xp, coupon, not redeemed)

        took = False
        for i in idx:
            f = fill_at(i)
            if f is not None:
                fills.append(f); took = True
                break                      # one entry per bond (no pyramiding)
        if took:
            pool = np.flatnonzero(gate)
            for _ in range(n_ctl):
                cf = fill_at(int(RNG.choice(pool)))
                if cf is not None:
                    ctl.append(cf.ret)
    return fills, np.array(ctl)


def report(name, bonds, fills, ctl):
    s = e2.summarize(fills, control=ctl if len(ctl) else None)
    print(f"  {name:24} n={s.get('n',0):5} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+6.2f}% hold={s.get('mean_hold',0):4.0f}d "
          f"excess={s.get('excess_vs_control',0)*100:+5.2f}% "
          f"p={s.get('excess_p_boot',1):.3f} stale={s.get('stale_share',0)*100:.0f}%",
          flush=True)
    if fills:
        r = e2.mtm_nav(bonds, fills)
        if r:
            ps = e2.perf_stats(*r)
            s["mtm"] = ps
            print(f"    MTM cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:5.2f} "
                  f"maxdd={ps['maxdd']*100:6.1f}% vol={ps['vol_m_ann']*100:.1f}%",
                  flush=True)
    return s


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
        if "coupon_inv" not in b:
            b["coupon_inv"] = b["coupon"]
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    print("\n[G1] depth monotonicity (<=3y, par bracket):", flush=True)
    for k in (1.0, 2.0, 3.0, 4.0):
        f, c = run_magnet(bonds, IS_LO, IS_HI, depth=k, max_mat=3)
        out[f"depth{k:.0f}"] = report(f"k={k:.0f}pt", bonds, f, c)

    print("\n[G3] maturity gradient (k=3, par bracket):", flush=True)
    for mm in (1, 2, 3, 5):
        f, c = run_magnet(bonds, IS_LO, IS_HI, depth=3.0, max_mat=mm)
        out[f"mat{mm}"] = report(f"<={mm}y", bonds, f, c)

    print("\n[G2] redemption-assumption bracket (k=3, <=3y):", flush=True)
    f, c = run_magnet(bonds, IS_LO, IS_HI, depth=3.0, max_mat=3, par_redemption=True)
    out["par"] = report("par bracket", bonds, f, c)
    f2, c2 = run_magnet(bonds, IS_LO, IS_HI, depth=3.0, max_mat=3, par_redemption=False)
    out["punitive"] = report("punitive bracket", bonds, f2, c2)

    print("\n[legacy-coupon sensitivity] (k=3, <=3y, par, OLD yield proxy):", flush=True)
    f3, c3 = run_magnet(bonds, IS_LO, IS_HI, depth=3.0, max_mat=3, use_inv_coupon=False)
    out["legacy_coupon"] = report("median-ytw proxy", bonds, f3, c3)

    from combine import save_fills
    save_fills(f, ROOT / "research" / "fills_magnet_is.json")
    (ROOT / "research" / "magnet_is.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/magnet_is.json", flush=True)


if __name__ == "__main__":
    main()
