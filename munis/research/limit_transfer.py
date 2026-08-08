"""REVERSE TRANSFER TEST — the corporate limit-entry rule applied to munis.

The GRANITE-CL refinement (accept the post-signal fill only if the customer-buy
price is <= the latest prior mid + 0.25) was designed and validated ENTIRELY on
corporate bonds. If the rule is structural (an implicit depth filter on
dislocation entries) rather than fitted, it should also improve the muni
dislocation strategy it was never designed on — the same logic, in reverse, as
the muni->corps parameter transfer that anchors the GRANITE audit.

Frozen: cap = +0.25 (verbatim from corps), muni locked config price_discount
discount=3.0, muni IS/OOS windows (IS 2012-2022, OOS 2023-2026). The matched
control faces the same fill cap.

  python munis/research/limit_transfer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt          # noqa: E402
import panel as panel_mod      # noqa: E402
from strategies import FACTORIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IS_LO, IS_HI = pd.Timestamp("2012-01-01"), bt.IS_END
OOS_LO, OOS_HI = pd.Timestamp("2023-01-01"), bt.OOS_END
CAP = 0.25
RNG = np.random.default_rng(53)


def load_bonds():
    print("loading muni panel ...", flush=True)
    pnl = panel_mod.load()
    uni = pd.read_csv(ROOT / "data" / "universe" / "universe.csv.gz")
    coupons = uni.set_index("six")["coupon"]
    coupons = coupons[~coupons.index.duplicated()]
    return bt.prepare(pnl, coupons)


def limit_filter(bonds, fills, cap=CAP):
    kept = []
    for f in fills:
        g = bonds[f.six]
        day = g["date"].values.astype("datetime64[D]").astype(np.int64)
        ed = np.datetime64(f.entry_date, "D").astype(np.int64)
        i = np.searchsorted(day, ed, side="left") - 1
        if i >= 0:
            mid = float(g["mid"].iloc[i])
            if np.isfinite(mid) and f.entry_px <= mid + cap:
                kept.append(f)
    return kept


def limit_control(bonds, fills, lo, hi, cap=CAP, k=10,
                  min_hold=365, max_hold=455):
    lo_d = np.datetime64(lo, "D").astype(np.int64)
    hi_d = np.datetime64(hi, "D").astype(np.int64)
    rets = []
    for f in fills:
        g = bonds[f.six]
        a = bt._arr(f.six, g)
        mid = g["mid"].to_numpy(float)
        pool = np.flatnonzero(a.elig)
        if not len(pool):
            continue
        for _ in range(k):
            i = int(RNG.choice(pool))
            sd = a.day[i]
            j = np.searchsorted(a.s_day, sd, side="right")
            if j >= len(a.s_day) or a.s_day[j] - sd > 7:
                continue
            ed = int(a.s_day[j]); ep = float(a.s_px_at[j])
            if ed < lo_d or ed > hi_d:
                continue
            ii = np.searchsorted(a.day, ed, side="left") - 1
            if ii < 0 or not np.isfinite(mid[ii]) or ep > mid[ii] + cap:
                continue
            ex = bt._exit_for(a, ed, min_hold, max_hold)
            if ex is None:
                continue
            xd, xp, _ = ex
            acc = a.coupon / 100 / 365 * (xd - ed) * 100
            rets.append((xp - ep + acc) / ep)
    return np.array(rets)


def evaluate(bonds, lo, hi, label, use_limit):
    fn = FACTORIES["price_discount"](discount=3.0)
    fills = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                          date_lo=lo, date_hi=hi)
    if use_limit:
        fills = limit_filter(bonds, fills)
        ctl = limit_control(bonds, fills, lo, hi)
        s = bt.summarize(fills, label)
        if len(ctl):
            s["excess_vs_control"] = s.get("mean_ret", 0) - float(ctl.mean())
            # bond-cluster bootstrap
            by = {}
            for f in fills:
                by.setdefault(f.six, []).append(f.ret)
            keys = list(by); cm = float(ctl.mean())
            stats = []
            for _ in range(2000):
                ks = RNG.choice(len(keys), size=len(keys), replace=True)
                stats.append(np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean() - cm)
            s["excess_p_boot"] = float((np.array(stats) <= 0).mean())
    else:
        ctl = bt.matched_random_control(bonds, fills, min_hold=365,
                                        max_hold=455, n_draws=10)
        s = bt.summarize(fills, label, control=ctl)
    print(f"  {label:22} n={s.get('n',0):5} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+6.2f}% "
          f"excess={s.get('excess_vs_control',0)*100:+5.2f}% "
          f"p={s.get('excess_p_boot',1):.3f}", flush=True)
    return s


def main():
    bonds = load_bonds()
    print(f"{len(bonds)} muni bonds prepared", flush=True)
    out = {}
    print("\n[TRANSFER] muni price_discount 3.0, hold ~1y:", flush=True)
    for lo, hi, w in [(IS_LO, IS_HI, "IS 2012-2022"), (OOS_LO, OOS_HI, "OOS 2023-2026")]:
        out[f"base_{w[:3].strip()}"] = evaluate(bonds, lo, hi, f"{w} base", False)
        out[f"limit_{w[:3].strip()}"] = evaluate(bonds, lo, hi, f"{w} +limit cap", True)
    (Path(__file__).resolve().parent / "results" / "limit_transfer.json").write_text(
        json.dumps(out, default=float))
    print("\nwrote munis/research/results/limit_transfer.json", flush=True)


if __name__ == "__main__":
    main()
