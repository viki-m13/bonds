"""BEDROCK-V drawdown levers — IS SCREEN (2003-2015), §8e pre-registered.

Levers (params frozen from IS data alone; see BEDROCK_RESEARCH.md §8e):
  H  : constant LQD beta-hedge, h in {0.5*beta_IS, 1.0*beta_IS}
  VT : vol-target trailing-63d, target in {10%, 12%} ann, cap 1x, cash at rf
  TR : LQD 200d-high trend de-risk, -5% => 0.5x exposure
  ST : cap monthly new entries at the IS median monthly count
  AD : adaptive depth — when tape dislocation share (20d-smoothed share of
       ask prints >=3pts under med60) > IS q90, require own dislocation
       <= -4 (variant -5) at the signal row

Kill gates (frozen): maxDD better by >= 8pp AND CAGR give-up <= 2pp AND
Sharpe(m) not lower than the unmodified IS book.

  python corps/research/bedrock_dd_screen.py
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
from bedrock_v import (cl_fills, real_coupons, exit_lagged, build_cs_median,  # noqa: E402
                       gate_value, gate_issuer_curve)
from bedrock_dd_diag import etf_daily  # noqa: E402
from combos import depth_of  # noqa: E402

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))


def stress_series(bonds):
    """Daily 20d-smoothed share of customer-ask prints >=3pts under med60."""
    print("building tape stress series ...", flush=True)
    days, flags = [], []
    for b in bonds.values():
        m60 = b.get("med60")
        if m60 is None:
            continue
        ok = ~np.isnan(b["s_px"]) & np.isfinite(m60)
        if not ok.any():
            continue
        days.append(b["day"][ok])
        flags.append((b["s_px"][ok] - m60[ok]) <= -3.0)
    d = np.concatenate(days); f = np.concatenate(flags)
    lo = d.min()
    nsig = np.bincount(d - lo, weights=f.astype(float))
    ntot = np.bincount(d - lo)
    idx = pd.to_datetime(np.arange(lo, lo + len(ntot)), unit="D")
    share = pd.Series(np.where(ntot > 0, nsig / np.where(ntot > 0, ntot, 1),
                               np.nan), index=idx)
    sm = share.rolling("20D", min_periods=5).mean()
    print(f"  {int((ntot > 0).sum()):,} tape days; stress share "
          f"median {sm.median()*100:.2f}%", flush=True)
    return lo, sm.to_numpy(), sm.index[0]


def stress_at(lo, arr, day):
    i = int(day - lo)
    if i < 0:
        return np.nan
    return arr[min(i, len(arr) - 1)]


def book(bonds, fills):
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    return e2.mtm_nav(bonds, fills, weights=w)


def report(days, nav, daily, tag, base=None, n=None):
    ps = e2.perf_stats(days, nav, daily)
    if n is not None:
        ps["n"] = n
    verdict = ""
    if base:
        dd_gain = ps["maxdd"] - base["maxdd"]          # positive = shallower DD
        ok = (dd_gain >= 0.08 and base["cagr"] - ps["cagr"] <= 0.02
              and ps["sharpe_m"] >= base["sharpe_m"])
        ps["admit"] = bool(ok)
        verdict = f"  ddGain={dd_gain*100:+.1f}pp -> {'ADMIT' if ok else 'reject'}"
    print(f"  {tag:26} cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:5.2f} "
          f"dd={ps['maxdd']*100:6.1f}%" + (f" n={n}" if n else "") + verdict,
          flush=True)
    return ps


def main():
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    s_lo, s_arr, _ = stress_series(bonds)
    is_days = np.arange(IS[0], IS[1] + 1)
    is_vals = np.array([stress_at(s_lo, s_arr, t) for t in is_days])
    Q90 = float(np.nanquantile(is_vals, 0.90))
    print(f"IS q90 stress threshold (FROZEN): {Q90*100:.2f}%", flush=True)

    base_f = cl_fills(bonds, *IS)
    pipe = real_coupons(bonds, exit_lagged(
        bonds, gate_issuer_curve(bonds, issuers, gate_value(bonds, base_f, med))))
    days, nav, daily = book(bonds, pipe)
    print(f"\n[IS base] BEDROCK-V pipeline:", flush=True)
    out = {"q90": Q90}
    out["base"] = report(days, nav, daily, "base", n=len(pipe))
    base = out["base"]
    rf = e2.load_rf(days) / 365.0
    lqd = etf_daily("LQD", days)

    # --- H: constant beta hedge (beta from IS monthly regression, frozen) ----
    ts = pd.Series(daily - rf, index=pd.to_datetime(days, unit="D"))
    em = pd.Series(lqd - rf, index=ts.index)
    bm = (1 + ts).resample("ME").prod() - 1
    lm = (1 + em).resample("ME").prod() - 1
    m = pd.concat([bm, lm], axis=1, keys=["b", "e"]).dropna()
    beta_is = float(np.cov(m["b"], m["e"])[0, 1] / np.var(m["e"]))
    out["beta_is"] = beta_is
    print(f"\n[H] beta_IS = {beta_is:.2f}", flush=True)
    for k in (0.5, 1.0):
        h = k * beta_is
        dl = daily - h * (lqd - rf)
        out[f"H_{k}"] = report(days, np.cumprod(1 + dl), dl,
                               f"hedge h={h:.2f}", base)

    # --- VT: vol target ------------------------------------------------------
    print("\n[VT]", flush=True)
    dser = pd.Series(daily, index=pd.to_datetime(days, unit="D"))
    vol = (dser.rolling(63).std() * np.sqrt(365)).shift(1)
    for tgt in (0.10, 0.12):
        expo = (tgt / vol).clip(upper=1.0).fillna(1.0).to_numpy()
        dl = expo * daily + (1 - expo) * rf
        out[f"VT_{tgt}"] = report(days, np.cumprod(1 + dl), dl,
                                  f"vol-target {int(tgt*100)}%", base)

    # --- TR: LQD trend de-risk ----------------------------------------------
    print("\n[TR]", flush=True)
    lqdpx = np.cumprod(1.0 + lqd)
    roll = pd.Series(lqdpx).rolling(200, min_periods=1).max().to_numpy()
    derisk = (lqdpx / roll - 1.0) < -0.05
    expo = np.where(np.roll(derisk, 1), 0.5, 1.0); expo[0] = 1.0
    dl = expo * daily + (1 - expo) * rf
    out["TR"] = report(days, np.cumprod(1 + dl), dl, "LQD -5% -> 0.5x", base)

    # --- ST: monthly entry cap ----------------------------------------------
    print("\n[ST]", flush=True)
    ym = pd.PeriodIndex(pd.to_datetime([f.entry_day for f in pipe], unit="D"),
                        freq="M")
    cap = int(pd.Series(1, index=ym).groupby(level=0).sum().median())
    print(f"  IS median monthly entries (FROZEN cap): {cap}", flush=True)
    cnt, kept = {}, []
    for f in sorted(pipe, key=lambda f: f.entry_day):
        k = pd.Timestamp(f.entry_day, unit="D").to_period("M")
        if cnt.get(k, 0) < cap:
            kept.append(f); cnt[k] = cnt.get(k, 0) + 1
    d2, n2, dl2 = book(bonds, kept)
    out["ST"] = report(d2, n2, dl2, f"entry cap {cap}/mo", base, n=len(kept))

    # --- AD: adaptive depth --------------------------------------------------
    print("\n[AD]", flush=True)
    for req in (-4.0, -5.0):
        kept = []
        for f in pipe:
            b = bonds[f.six]
            i = np.searchsorted(b["day"], f.entry_day, side="left") - 1
            sv = stress_at(s_lo, s_arr, int(b["day"][i])) if i >= 0 else np.nan
            if not (np.isfinite(sv) and sv > Q90):
                kept.append(f); continue
            m60 = b.get("med60")
            if i < 0 or m60 is None or not (np.isfinite(b["mid"][i])
                                            and np.isfinite(m60[i])):
                kept.append(f); continue
            if float(b["mid"][i] - m60[i]) <= req:
                kept.append(f)
        d2, n2, dl2 = book(bonds, kept)
        r = np.array([f.ret for f in kept])
        st = report(d2, n2, dl2, f"depth<={req:.0f} in stress", base, n=len(kept))
        st["mean"] = float(r.mean()); st["win"] = float((r > 0).mean())
        out[f"AD_{req}"] = st

    p = ROOT / "research" / "bedrock_dd_screen.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
