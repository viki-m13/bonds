"""KEYSTONE-XL — transfer the full corporate XL stack to munis, stepwise:

  base       price_discount 3.0, ~1y hold (the locked KEYSTONE config)
  +limit     entry only if customer-buy price <= latest prior mid + 0.25
  +issuer    max 1 concurrent position per issuer (6-char CUSIP prefix)
  +recovery  exit at the first customer-sell print (>=21d) once the day's mid
             has recovered to the ENTRY-day trailing-60d median; 455d stop

Each rule frozen verbatim from corps. IS 2012-2022 and OOS 2023-2026 per the
muni convention. Controls for the excess figures share the limit cap.

  python munis/research/keystone_xl.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt          # noqa: E402
from strategies import FACTORIES  # noqa: E402
from limit_transfer import load_bonds, limit_filter, limit_control  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IS_LO, IS_HI = pd.Timestamp("2012-01-01"), bt.IS_END
OOS_LO, OOS_HI = pd.Timestamp("2023-01-01"), bt.OOS_END
RNG = np.random.default_rng(71)

_ISSUER: dict[str, str] | None = None


def issuer_of(six: str) -> str:
    """Real issuer key for a muni security. EMMA securityIds are opaque
    33-char hashes, so (unlike corp CUSIPs) a 6-char prefix carries no issuer
    information — the issuer is recovered from the universe description
    (text before the '/' series separator). Falls back to the id itself."""
    global _ISSUER
    if _ISSUER is None:
        uni = pd.read_csv(ROOT / "data" / "universe" / "universe.csv.gz").drop_duplicates("six")
        key = (uni["desc"].astype(str).str.upper().str.split("/").str[0]
               .str.strip().str.replace(r"\s+", " ", regex=True))
        _ISSUER = dict(zip(uni["six"], key))
    return _ISSUER.get(six, six)


def issuer_cap(fills, cap=1):
    """AUDIT FIX 2026-08: previously grouped by six[:6] — near-unique for
    EMMA hash ids, so the cap never bound (removed 1 of 1058 fills). Now
    groups by the real issuer; the cap binds as documented."""
    fills = sorted(fills, key=lambda f: f.entry_date)
    open_by, kept = {}, []
    for f in fills:
        iss = issuer_of(f.six)
        cur = [x for x in open_by.get(iss, []) if x > f.entry_date]
        if len(cur) < cap:
            kept.append(f); cur.append(f.exit_date)
        open_by[iss] = cur
    return kept


def med60_at(g, cache={}):
    key = id(g)
    if key not in cache:
        s = g.set_index("date")["mid"]
        cache[key] = s.rolling("60D", min_periods=5).median().shift(1)
    return cache[key]


def recovery_exit(bonds, fills):
    out = []
    for f in fills:
        g = bonds[f.six]
        a = bt._arr(f.six, g)
        med = med60_at(g)
        try:
            tgt = float(med.asof(f.entry_date))
        except Exception:
            tgt = np.nan
        if not np.isfinite(tgt):
            continue
        day = g["date"].values.astype("datetime64[D]").astype(np.int64)
        mid = g["mid"].to_numpy(float)
        ed = np.datetime64(f.entry_date, "D").astype(np.int64)
        k0 = np.searchsorted(a.p_day, ed + 21, side="left")
        xd = xp = None
        for k in range(k0, len(a.p_day)):
            if a.p_day[k] > ed + 455:
                break
            di = np.searchsorted(day, a.p_day[k], side="right") - 1
            if di >= 0 and np.isfinite(mid[di]) and mid[di] >= tgt:
                xd, xp, st = int(a.p_day[k]), float(a.p_px_at[k]), False
                break
        if xd is None:
            k2 = np.searchsorted(a.p_day, ed + 455, side="right") - 1
            if k2 < 0 or a.p_day[k2] <= ed:
                continue
            xd, xp, st = ed + 455, float(a.p_px_at[k2]), True
        out.append(bt.Fill(f.six, f.entry_date, f.entry_px,
                           pd.Timestamp(xd, unit="D"), xp, f.coupon, st))
    return out


def nav_stats(fills, label):
    """Monthly-marked NAV from per-trade daily-compounded returns (the muni
    engine has no daily mid-mark MTM; linear attribution — comparative)."""
    if not fills:
        return None
    d0 = min(f.entry_date for f in fills); d1 = max(f.exit_date for f in fills)
    days = pd.date_range(d0, d1, freq="D")
    di = {d: i for i, d in enumerate(days)}
    sret = np.zeros(len(days)); cnt = np.zeros(len(days))
    for f in fills:
        if f.hold_days <= 0:
            continue
        dr = (1 + f.ret) ** (1 / f.hold_days) - 1
        sret[di[f.entry_date]:di[f.exit_date]] += dr
        cnt[di[f.entry_date]:di[f.exit_date]] += 1
    daily = np.where(cnt > 0, sret / np.where(cnt > 0, cnt, 1), 0)
    nav = np.cumprod(1 + daily)
    yrs = (days[-1] - days[0]).days / 365.25
    peak = np.maximum.accumulate(nav)
    return {"cagr": float(nav[-1] ** (1 / yrs) - 1),
            "maxdd": float((nav / peak - 1).min()),
            "total": float(nav[-1] - 1)}


def report(bonds, fills, label, lo, hi, with_ctl=True):
    s = bt.summarize(fills, label)
    if with_ctl and fills:
        ctl = limit_control(bonds, fills, lo, hi)
        if len(ctl):
            s["excess_vs_control"] = s.get("mean_ret", 0) - float(ctl.mean())
            by = {}
            for f in fills:
                by.setdefault(f.six, []).append(f.ret)
            keys = list(by); cm = float(ctl.mean())
            boots = []
            for _ in range(2000):
                ks = RNG.choice(len(keys), size=len(keys), replace=True)
                boots.append(np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean() - cm)
            s["excess_p_boot"] = float((np.array(boots) <= 0).mean())
    ns = nav_stats(fills, label)
    if ns:
        s.update({f"nav_{k}": v for k, v in ns.items()})
    print(f"  {label:26} n={s.get('n',0):5} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+6.2f}% hold={s.get('mean_hold',0):4.0f}d "
          f"excess={s.get('excess_vs_control',0)*100:+5.2f}% "
          f"p={s.get('excess_p_boot',1):.3f} "
          f"cagr={ns['cagr']*100:+6.2f}% dd={ns['maxdd']*100:.1f}%" if ns else "",
          flush=True)
    return s


def main():
    bonds = load_bonds()
    print(f"{len(bonds)} muni bonds prepared", flush=True)
    fn = FACTORIES["price_discount"](discount=3.0)
    out = {}
    for lo, hi, w in [(IS_LO, IS_HI, "IS"), (OOS_LO, OOS_HI, "OOS")]:
        print(f"\n[{w}] {lo.date()}..{hi.date()}:", flush=True)
        base = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                             date_lo=lo, date_hi=hi)
        out[f"{w}_base"] = report(bonds, base, "base", lo, hi)
        lim = limit_filter(bonds, base)
        out[f"{w}_limit"] = report(bonds, lim, "+limit", lo, hi)
        cap = issuer_cap(lim)
        out[f"{w}_cap"] = report(bonds, cap, "+limit+issuer", lo, hi)
        rec = recovery_exit(bonds, cap)
        out[f"{w}_xl"] = report(bonds, rec, "KEYSTONE-XL (+recovery)", lo, hi)
    p = Path(__file__).resolve().parent / "results" / "keystone_xl.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
