"""Round 4 — trade like stocks: response-curve atlas + four equity-style
strategies, IS-only (design wall 2015; survivors go to one batch-4 OOS).

  [ATLAS] conditional forward-return curves (marks-based study, not fills):
          for each event type, forward mid+accrual returns at 7 horizons vs
          a matched unconditional baseline on the same universe.
  [PYR]   pyramiding: XL entries (>=3pt) plus a SECOND unit if the bond
          prints >=5pt below median >=7d later while the first is open
          (max 2 units/bond). Adds vs initials compared per-trade.
  [BRK]   breakout-carry: monthly, bonds within 0.5% of their 250d high AND
          cs in [1.5%, 8%] (strength + fat carry), rank by cs, top 60.
  [OBV]   accumulation divergence: trailing-60d signed customer flow > 0
          (net buying) while price sits >=1pt under its median — buy what
          smart money quietly accumulates. Hold 60-240d.
  [MOM]   bond-level 6-1 momentum, HY band (cs 3-10%), top 60 monthly.

  python corps/research/explore4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
import portfolio as pf  # noqa: E402
from granite_experiments import sig_disc, gate_mat5, issuer_cap_filter  # noqa: E402
from oos2 import limit_filter  # noqa: E402
from combos import dynamic_exit  # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")
HORIZONS = [5, 10, 21, 63, 126, 252, 365]
RNG = np.random.default_rng(61)


# ------------------------------------------------------------------- ATLAS

def fwd_ret(b, i, h):
    day = b["day"]; mid = b["mid"]
    j = np.searchsorted(day, day[i] + h, side="right") - 1
    if j <= i or day[j] - (day[i] + h) < -h * 0.5:
        return None
    m0, m1 = mid[i], mid[j]
    if not (np.isfinite(m0) and np.isfinite(m1)) or m0 <= 0:
        return None
    acc = b["coupon_inv"] / 100 / 365 * (day[j] - day[i]) * 100
    return (m1 - m0 + acc) / m0


def atlas(bonds):
    conds = {
        "drop>=1pt": lambda b: (b["s_px"] - b["med60"]) <= -1,
        "drop>=2pt": lambda b: (b["s_px"] - b["med60"]) <= -2,
        "drop>=3pt": lambda b: (b["s_px"] - b["med60"]) <= -3,
        "drop>=5pt": lambda b: (b["s_px"] - b["med60"]) <= -5,
        "vol>=5x": lambda b: np.nan_to_num(b["qv"]) >= 5 * np.maximum(np.nan_to_num(b["qvmed90"]), 1e-3),
        "hi250": None,   # computed inline (250d high proximity)
        "baseline": lambda b: np.ones(len(b["day"]), bool),
    }
    out = {k: {h: [] for h in HORIZONS} for k in conds}
    for six, b in bonds.items():
        med60 = b.get("med60")
        if med60 is None or len(b["day"]) < 40:
            continue
        gate = b["elig"] & (b["mat"] <= 5) & np.isfinite(med60)
        if not gate.any():
            continue
        day = b["day"]; mid = b["mid"]
        # 250d rolling high (backward)
        j250 = np.searchsorted(day, day - 250)
        hi = np.array([np.nanmax(mid[j250[i]:i + 1]) if i > j250[i] else np.nan
                       for i in range(len(day))])
        for name, fn in conds.items():
            if name == "hi250":
                sig = gate & np.isfinite(hi) & (mid >= hi - 0.5)
            elif name == "baseline":
                g = np.flatnonzero(gate)
                sig = np.zeros(len(day), bool)
                if len(g):
                    sig[RNG.choice(g, size=min(len(g), 6), replace=False)] = True
            else:
                sig = gate & np.nan_to_num(fn(b), nan=False).astype(bool)
            idx = np.flatnonzero(sig)
            if name.startswith("drop") and len(idx):
                keep = [idx[0]]
                for i in idx[1:]:
                    if day[i] - day[keep[-1]] >= 30:
                        keep.append(i)
                idx = np.array(keep)
            if len(idx) > 8 and name not in ("baseline",):
                idx = RNG.choice(idx, size=8, replace=False)
            for i in idx:
                if not (IS_LO <= day[i] <= IS_HI):
                    continue
                for h in HORIZONS:
                    r = fwd_ret(b, int(i), h)
                    if r is not None:
                        out[name][h].append(r)
    print("\n[ATLAS] forward mid+accrual returns (IS, % — excess vs baseline in parens):",
          flush=True)
    base_m = {h: float(np.mean(out["baseline"][h])) if out["baseline"][h] else np.nan
              for h in HORIZONS}
    res = {}
    hdr = "  " + f"{'condition':12}" + "".join(f"{h:>9}d" for h in HORIZONS)
    print(hdr, flush=True)
    for name in conds:
        row = f"  {name:12}"
        res[name] = {}
        for h in HORIZONS:
            v = out[name][h]
            if not v:
                row += f"{'—':>10}"
                continue
            m = float(np.mean(v))
            res[name][h] = {"n": len(v), "mean": m, "excess": m - base_m[h]}
            row += f"{(m - base_m[h])*100:+9.2f}" if name != "baseline" else f"{m*100:9.2f}"
        print(row, flush=True)
    return res


# ------------------------------------------------------------------ PYRAMID

def pyramid_fills(bonds, lo, hi):
    initials, adds = [], []
    for six, b in bonds.items():
        med60 = b.get("med60")
        if med60 is None:
            continue
        gate = b["elig"] & (b["mat"] <= 5) & np.isfinite(med60)
        day = b["day"]
        sig3 = gate & ((b["s_px"] - med60) <= -3)
        sig5 = gate & ((b["s_px"] - med60) <= -5)
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]

        def try_fill(i):
            t = day[i]
            j = np.searchsorted(s_day, t, side="right")
            if j >= len(s_day) or s_day[j] - t > 7:
                return None
            ed = int(s_day[j]); ep = float(s_px[j])
            ii = np.searchsorted(day, ed, side="left") - 1
            if ii < 0 or not np.isfinite(b["mid"][ii]) or ep > b["mid"][ii] + 0.25:
                return None
            if ed < lo or ed > hi:
                return None
            k = np.searchsorted(p_day, ed + 365, side="left")
            if k < len(p_day) and p_day[k] <= ed + 455:
                xd, xp, st = int(p_day[k]), float(p_px[k]), False
            else:
                k2 = np.searchsorted(p_day, ed + 455, side="right") - 1
                if k2 < 0 or p_day[k2] <= ed:
                    return None
                xd, xp, st = ed + 455, float(p_px[k2]), True
            return e2.Fill(six, ed, ep, xd, xp, b["coupon_inv"], st)

        last_exit = -10**9
        open_f = None
        for i in np.flatnonzero(sig3):
            t = day[i]
            if open_f is None or t >= open_f.exit_day:
                if t - last_exit < 30:
                    continue
                f = try_fill(int(i))
                if f is not None:
                    initials.append(f); open_f = f; last_exit = f.exit_day
            else:
                if sig5[i] and t >= open_f.entry_day + 7 and \
                   not any(a.six == six and a.entry_day > open_f.entry_day for a in adds[-3:]):
                    f = try_fill(int(i))
                    if f is not None and f.entry_day < open_f.exit_day:
                        adds.append(f)
    return initials, adds


# ---------------------------------------------------------- portfolio scores

def score_breakout(bonds):
    def sc(b, t, i):
        day = b["day"]; mid = b["mid"]
        if not (0.015 <= float(b["cs"][i]) <= 0.08):
            return None
        if not (1 <= int(b["mat"][i]) <= 8):
            return None
        j = np.searchsorted(day, day[i] - 250)
        seg = mid[j:i + 1]
        if len(seg) < 20:
            return None
        hi = np.nanmax(seg)
        if not np.isfinite(hi) or mid[i] < hi - 0.5:
            return None
        return float(b["cs"][i])       # among breakout names, rank by carry
    return sc


def score_mom(b, t, i):
    day = b["day"]; mid = b["mid"]
    if not (0.03 <= float(b["cs"][i]) <= 0.10) or not (2 <= int(b["mat"][i]) <= 10):
        return None
    i21 = np.searchsorted(day, t - 21, side="right") - 1
    i180 = np.searchsorted(day, t - 180, side="right") - 1
    if i180 < 0 or i21 <= i180 or day[i180] < t - 240:
        return None
    m0, m1 = float(mid[i180]), float(mid[i21])
    if not (np.isfinite(m0) and np.isfinite(m1)) or m0 <= 0:
        return None
    return m1 / m0 - 1.0


# ---------------------------------------------------------------------- OBV

def sig_obv(b):
    day = b["day"]; med60 = b.get("med60")
    if med60 is None or len(day) < 30:
        return None
    qv = np.nan_to_num(b["qv"])
    signed = np.where(np.isfinite(b["s_px"]) & ~np.isfinite(b["p_px"]), qv,
             np.where(np.isfinite(b["p_px"]) & ~np.isfinite(b["s_px"]), -qv, 0.0))
    cs_ = np.concatenate([[0.0], np.cumsum(signed)])
    j60 = np.searchsorted(day, day - 60)
    flow60 = cs_[np.arange(1, len(day) + 1)] - cs_[j60]
    under = (b["mid"] - med60) <= -1.0
    return (flow60 > 0) & under & np.isfinite(med60)


def show_book(bonds, fills, label):
    if not fills:
        print(f"  {label}: no fills", flush=True)
        return None
    rr = np.array([f.ret for f in fills])
    r = e2.mtm_nav(bonds, fills)
    ps = e2.perf_stats(*r) if r else {}
    print(f"  {label:24} n={len(fills):5} mean={rr.mean()*100:+6.2f}% "
          f"win={(rr>0).mean()*100:3.0f}% cagr={ps.get('cagr',0)*100:+6.2f}% "
          f"sharpe_m={ps.get('sharpe_m',0):5.2f} maxdd={ps.get('maxdd',0)*100:6.1f}%",
          flush=True)
    ps.update({"n": len(fills), "mean_ret": float(rr.mean()),
               "win": float((rr > 0).mean())})
    return ps


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}

    out["atlas"] = atlas(bonds)

    print("\n[PYR] pyramiding (IS):", flush=True)
    initials, adds = pyramid_fills(bonds, IS_LO, IS_HI)
    out["pyr_init"] = show_book(bonds, initials, "initial units (3pt)")
    out["pyr_adds"] = show_book(bonds, adds, "add units (>=5pt)")
    out["pyr_all"] = show_book(bonds, initials + adds, "pyramided book")

    print("\n[OBV] accumulation divergence (IS):", flush=True)
    f = e2.run_events(bonds, sig_obv, min_hold=60, max_hold=240,
                      date_lo=IS_LO, date_hi=IS_HI, extra_gate=gate_mat5)
    ctl = e2.matched_control(bonds, f, min_hold=60, max_hold=240,
                             extra_gate=gate_mat5)
    s = e2.summarize(f, control=ctl)
    out["obv"] = s
    print(f"  n={s.get('n',0)} win={s.get('win_rate',0)*100:.0f}% "
          f"mean={s.get('mean_ret',0)*100:+.2f}% excess={s.get('excess_vs_control',0)*100:+.2f}% "
          f"p={s.get('excess_p_boot',1):.3f}", flush=True)
    if f:
        show_book(bonds, f, "OBV book")

    print("\n[BRK] breakout-carry (IS, monthly top-60):", flush=True)
    closed = pf.run_portfolio(bonds, score_breakout(bonds), IS_LO, IS_HI,
                              top_n=60, hold_until_rank=150)
    out["brk"] = show_book(bonds, pf.positions_to_fills(closed), "breakout-carry")

    print("\n[MOM] bond-level 6-1 momentum, HY band (IS, monthly top-60):", flush=True)
    closed = pf.run_portfolio(bonds, score_mom, IS_LO, IS_HI,
                              top_n=60, hold_until_rank=150)
    out["mom"] = show_book(bonds, pf.positions_to_fills(closed), "hy momentum")

    (ROOT / "research" / "explore4_results.json").write_text(json.dumps(out, default=float))
    print("\nwrote corps/research/explore4_results.json", flush=True)


if __name__ == "__main__":
    main()
