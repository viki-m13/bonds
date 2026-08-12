"""BEDROCK-V drawdown round 4 — no-trade triggers on a CAPITALIZED book
(§8g, pre-registered).

Capital sim: start 1.0; new position sized NAV/K on entry day, taken only if
cash covers it; marks follow the honest mtm path (entry ask -> mids, stale
flat, daily accrual -> exit bid); cash earns 3M T-bill. K in {50, 100}.
Triggers (lagged 1d): T1 = tape stress > q90 & rising; T2 = q75 & rising;
T3 = own NAV >10% below trailing 1y high. Runoff passive; halted signals
missed, not queued.

IS screen first; the ONE OOS look runs ONLY for triggers that pass the IS
gates (maxDD >=8pp better, CAGR give-up <=2pp, Sharpe(m) not lower, per-K).

  python corps/research/bedrock_dd_screen4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bedrock_v import (cl_fills, real_coupons, exit_lagged, build_cs_median,  # noqa: E402
                       gate_value, gate_issuer_curve)
from bedrock_dd_screen import stress_series  # noqa: E402

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)


def fill_path(bonds, f):
    """Daily returns for one fill over (entry_day, exit_day] — same segment
    logic as e2.mtm_nav."""
    b = bonds[f.six]
    day = b["day"]; mid = b["mid"]
    i0 = np.searchsorted(day, f.entry_day, side="right")
    i1 = np.searchsorted(day, f.exit_day, side="left")
    mds = [f.entry_day]; mks = [f.entry_px]
    for i in range(i0, i1):
        if not np.isnan(mid[i]):
            mds.append(int(day[i])); mks.append(float(mid[i]))
    mds.append(f.exit_day); mks.append(f.exit_px)
    acc = f.coupon / 100.0 / 365.0 * 100.0
    rets = np.zeros(f.exit_day - f.entry_day)
    for k in range(1, len(mds)):
        gap = mds[k] - mds[k - 1]
        if gap <= 0:
            continue
        tot = (mks[k] + acc * gap) / mks[k - 1] - 1.0
        dr = (1.0 + tot) ** (1.0 / gap) - 1.0
        a = mds[k - 1] - f.entry_day; z = mds[k] - f.entry_day
        rets[a:z] = dr
    return rets


def cap_sim(bonds, fills, K, rf_daily, d0, d1, halt=None, own_eq=False,
            eq_slope=False):
    """halt: bool array over days (True = no new entries), or None.
    own_eq: T3 — halt when yesterday's NAV >10% below trailing 1y high.
    eq_slope (T4): additionally require NAV < NAV 20d ago — halt only while
    own equity is underwater AND still deteriorating."""
    days = np.arange(d0, d1 + 1)
    n = len(days)
    by_entry = {}
    for f in sorted(fills, key=lambda f: f.entry_day):
        by_entry.setdefault(f.entry_day, []).append(f)
    cash = 1.0
    open_pos = []          # [value, rets_array, ptr]
    nav = np.zeros(n)
    hi_roll = -np.inf
    nav_hist = np.zeros(n)
    taken = skipped_halt = skipped_cash = 0
    inv_share = np.zeros(n)
    for t in range(n):
        d = int(days[t])
        cash *= 1.0 + rf_daily[t]
        nxt = []
        for pos in open_pos:
            v, rets, ptr = pos
            v *= 1.0 + rets[ptr]
            ptr += 1
            if ptr >= len(rets):
                cash += v
            else:
                nxt.append([v, rets, ptr])
        open_pos = nxt
        cur = cash + sum(p[0] for p in open_pos)
        halted = bool(halt[t]) if halt is not None else False
        if own_eq and t > 0:
            hi_roll = nav_hist[max(0, t - 365):t].max()
            under = nav_hist[t - 1] < 0.90 * hi_roll
            if eq_slope:
                under = under and t > 20 and nav_hist[t - 1] < nav_hist[t - 21]
            halted = halted or under
        for f in by_entry.get(d, []):
            if halted:
                skipped_halt += 1
                continue
            s = cur / K
            if cash < s:
                skipped_cash += 1
                continue
            cash -= s
            open_pos.append([s, fill_path(bonds, f), 0])
            taken += 1
        cur = cash + sum(p[0] for p in open_pos)
        nav[t] = cur
        nav_hist[t] = cur
        inv_share[t] = 1.0 - cash / cur
    daily = np.empty(n)
    daily[0] = nav[0] - 1.0
    daily[1:] = nav[1:] / nav[:-1] - 1.0
    return days, nav, daily, {"taken": taken, "skipped_halt": skipped_halt,
                              "skipped_cash": skipped_cash,
                              "inv_mean": float(inv_share.mean()),
                              "inv_max": float(inv_share.max())}


def stats(days, nav, daily, tag, base=None):
    ps = e2.perf_stats(days, nav, daily)
    verdict = ""
    if base is not None:
        gain = ps["maxdd"] - base["maxdd"]
        ok = (gain >= 0.08 and base["cagr"] - ps["cagr"] <= 0.02
              and ps["sharpe_m"] >= base["sharpe_m"])
        ps["admit"] = bool(ok)
        verdict = f"  ddGain={gain*100:+.1f}pp -> {'ADMIT' if ok else 'reject'}"
    print(f"  {tag:28} cagr={ps['cagr']*100:+6.2f}% sharpe_m={ps['sharpe_m']:5.2f} "
          f"dd={ps['maxdd']*100:6.1f}%{verdict}", flush=True)
    return ps


def concurrency(fills):
    ev = []
    for f in fills:
        ev.append((f.entry_day, 1)); ev.append((f.exit_day, -1))
    ev.sort()
    c = mx = 0
    counts = []
    for _, e_ in ev:
        c += e_; counts.append(c); mx = max(mx, c)
    return mx, int(np.percentile(counts, 90))


def main():
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    s_lo, s_arr, _ = stress_series(bonds)
    is_days = np.arange(IS[0], IS[1] + 1)
    iv = s_arr[np.clip(is_days - s_lo, 0, len(s_arr) - 1)]
    Q90 = float(np.nanquantile(iv, 0.90)); Q75 = float(np.nanquantile(iv, 0.75))
    print(f"FROZEN: q90={Q90*100:.2f}% q75={Q75*100:.2f}%", flush=True)

    base_full = cl_fills(bonds, *(e2.D("2003-01-01"), OOS[1]))
    pipe_all = real_coupons(bonds, exit_lagged(
        bonds, gate_issuer_curve(bonds, issuers, gate_value(bonds, base_full, med))))
    out = {"q90": Q90, "q75": Q75}

    def halt_arr(days, q):
        i = np.clip(days - 1 - s_lo, 0, len(s_arr) - 1)        # lag 1 day
        j = np.clip(i - 20, 0, len(s_arr) - 1)
        a, b = s_arr[i], s_arr[j]
        return np.isfinite(a) & np.isfinite(b) & (a > q) & (a > b)

    def run_window(tag, lo, hi, admits=None):
        fills = [f for f in pipe_all if lo <= f.entry_day <= hi]
        mx, p90 = concurrency(fills)
        print(f"\n=== {tag}: {len(fills)} fills; concurrency max={mx} p90={p90} ===",
              flush=True)
        d0 = min(f.entry_day for f in fills); d1 = max(f.exit_day for f in fills)
        days = np.arange(d0, d1 + 1)
        rf_d = e2.load_rf(days) / 365.0
        res = {"n": len(fills), "conc_max": mx, "conc_p90": p90}
        for K in (50, 100):
            b_days, b_nav, b_daily, b_info = cap_sim(bonds, fills, K, rf_d, d0, d1)
            print(f"[K={K}] base invested mean={b_info['inv_mean']*100:.0f}% "
                  f"max={b_info['inv_max']*100:.0f}% "
                  f"skipped_cash={b_info['skipped_cash']}", flush=True)
            base = stats(b_days, b_nav, b_daily, f"K={K} base (no trigger)")
            res[f"K{K}_base"] = {**base, **b_info}
            trigs = {"T1 q90&rising": ("h", Q90), "T2 q75&rising": ("h", Q75),
                     "T3 own-equity -10%": ("own", None)}
            for tname, (kind, q) in trigs.items():
                if admits is not None and tname not in admits:
                    continue
                if kind == "h":
                    ha = halt_arr(days, q)
                    r = cap_sim(bonds, fills, K, rf_d, d0, d1, halt=ha)
                else:
                    r = cap_sim(bonds, fills, K, rf_d, d0, d1, own_eq=True)
                sdays, snav, sdaily, info = r
                st = stats(sdays, snav, sdaily, f"K={K} {tname}", base)
                res[f"K{K}_{tname}"] = {**st, **info}
                print(f"      taken={info['taken']} halt-skip={info['skipped_halt']} "
                      f"cash-skip={info['skipped_cash']} "
                      f"inv_mean={info['inv_mean']*100:.0f}%", flush=True)
        return res

    out["IS"] = run_window("IS 2003-2015", *IS)

    # one-shot OOS ONLY for IS survivors (per K)
    admits = set()
    for K in (50, 100):
        for tname in ("T1 q90&rising", "T2 q75&rising", "T3 own-equity -10%"):
            if out["IS"].get(f"K{K}_{tname}", {}).get("admit"):
                admits.add(tname)
    print(f"\nIS SURVIVORS (any K): {sorted(admits) if admits else 'NONE'}", flush=True)
    if admits:
        out["OOS"] = run_window("OOS 2016+", *OOS, admits=admits)

    p = ROOT / "research" / "bedrock_dd_screen4.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
