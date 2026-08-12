"""BEDROCK-V drawdown round 3 — phase-selective entry levers, IS SCREEN
(§8f, pre-registered). Stress state: CRISIS = 20d tape stress > IS-q90;
RISING = stress > stress 20 calendar days ago.

  [F]  mechanism diagnostic (IS only): in-episode entries by RISING state
       and value-gap quartile
  [SS] skip entries when CRISIS & RISING (SS75: q75 threshold)
  [FK] in CRISIS require mid(sig) >= mid 5 (or 3) prints earlier
  [DE] CRISIS signals enter 30-44d later if still >=3pts cheap + limit ok
  [CC] in CRISIS skip value-gap > IS-q90 of entries (deep distress)
  [MS] in CRISIS require mat <= 3
  [SO] overlay 0.5x while CRISIS & RISING

Kill gates: IS maxDD >=8pp better, CAGR give-up <=2pp, Sharpe(m) not lower.

  python corps/research/bedrock_dd_screen2.py
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
from bedrock_dd_screen import stress_series, book, report  # noqa: E402
from bedrock_dd_diag import episodes  # noqa: E402

MAXH = 455
IS = (e2.D("2003-01-01"), e2.D("2015-12-31"))


def sig_i(b, f):
    return np.searchsorted(b["day"], f.entry_day, side="left") - 1


def main():
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    s_lo, s_arr, _ = stress_series(bonds)

    def stress_at(day):
        i = int(day - s_lo)
        return s_arr[min(max(i, 0), len(s_arr) - 1)] if i >= 0 else np.nan

    def rising_at(day):
        i = int(day - s_lo)
        j = i - 20
        if j < 0 or i >= len(s_arr):
            return False
        a, b_ = s_arr[i], s_arr[j]
        return bool(np.isfinite(a) and np.isfinite(b_) and a > b_)

    is_days = np.arange(IS[0], IS[1] + 1)
    is_vals = np.array([stress_at(t) for t in is_days])
    Q90 = float(np.nanquantile(is_vals, 0.90))
    Q75 = float(np.nanquantile(is_vals, 0.75))
    print(f"FROZEN: q90={Q90*100:.2f}% q75={Q75*100:.2f}%", flush=True)

    base_f = cl_fills(bonds, *IS)
    v = gate_issuer_curve(bonds, issuers, gate_value(bonds, base_f, med))
    pipe = real_coupons(bonds, exit_lagged(bonds, v))
    days, nav, daily = book(bonds, pipe)
    print("\n[IS base]", flush=True)
    out = {"q90": Q90, "q75": Q75}
    out["base"] = report(days, nav, daily, "base", n=len(pipe))
    base = out["base"]
    rf = e2.load_rf(days) / 365.0

    # value-gap per fill (for F and CC)
    def vgap(f):
        b = bonds[f.six]
        i = sig_i(b, f)
        if i < 0:
            return np.nan
        cs = b["cs"][i]
        if not (np.isfinite(cs) and cs > 0):
            return np.nan
        m = med.get((int(b["day"][i]), int(np.clip(b["mat"][i], 0, 10))))
        return np.log(float(cs)) - m if m is not None and np.isfinite(m) else np.nan

    gaps = np.array([vgap(f) for f in pipe])
    CCQ = float(np.nanquantile(gaps, 0.90))
    print(f"FROZEN CC threshold (IS-q90 value-gap): {CCQ:.2f}", flush=True)

    # ---- [F] mechanism diagnostic (IS only) ---------------------------------
    eps = episodes(days, nav)
    in_ep = np.zeros(len(pipe), bool)
    for e in eps:
        hi = e["rec_day"] if e["rec_day"] else IS[1]
        for k, f in enumerate(pipe):
            if e["peak_day"] < f.entry_day <= hi:
                in_ep[k] = True
    ris = np.array([rising_at(int(bonds[f.six]["day"][sig_i(bonds[f.six], f)]))
                    for f in pipe])
    r = np.array([f.ret for f in pipe])
    print("\n[F] IS in-episode entries by stress slope at signal:", flush=True)
    out["F"] = {}
    for tag, m in (("episode & RISING", in_ep & ris),
                   ("episode & not-rising", in_ep & ~ris),
                   ("normal tape", ~in_ep)):
        if m.sum():
            out["F"][tag] = {"n": int(m.sum()), "mean": float(r[m].mean()),
                             "win": float((r[m] > 0).mean())}
            print(f"  {tag:22} n={m.sum():5} mean={r[m].mean()*100:+6.2f}% "
                  f"win={(r[m]>0).mean()*100:3.0f}%", flush=True)
    qs = np.nanquantile(gaps[in_ep], [0.25, 0.5, 0.75])
    print("  in-episode by value-gap quartile:", flush=True)
    lo_edge = [-np.inf, *qs]; hi_edge = [*qs, np.inf]
    for qi in range(4):
        m = in_ep & (gaps > lo_edge[qi]) & (gaps <= hi_edge[qi])
        if m.sum():
            print(f"    Q{qi+1} n={m.sum():4} mean={r[m].mean()*100:+6.2f}% "
                  f"win={(r[m]>0).mean()*100:3.0f}%", flush=True)
            out["F"][f"gapQ{qi+1}"] = {"n": int(m.sum()), "mean": float(r[m].mean())}

    # ---- helper to evaluate a filtered book ---------------------------------
    def run_filter(keep_mask, tag):
        kept = [f for f, k in zip(pipe, keep_mask) if k]
        d2, n2, dl2 = book(bonds, kept)
        st = report(d2, n2, dl2, tag, base, n=len(kept))
        rr = np.array([f.ret for f in kept])
        st["mean"] = float(rr.mean()); st["win"] = float((rr > 0).mean())
        return st

    crisis = np.array([stress_at(int(bonds[f.six]["day"][sig_i(bonds[f.six], f)]))
                       for f in pipe])
    print("\n[SS]", flush=True)
    out["SS"] = run_filter(~((crisis > Q90) & ris), "SS  skip crisis&rising")
    out["SS75"] = run_filter(~((crisis > Q75) & ris), "SS75 skip q75&rising")

    print("\n[FK]", flush=True)
    for k in (5, 3):
        keep = np.ones(len(pipe), bool)
        for idx, f in enumerate(pipe):
            if crisis[idx] > Q90:
                b = bonds[f.six]
                i = sig_i(b, f)
                j = i - k
                if i < 0 or j < 0 or not (np.isfinite(b["mid"][i])
                                          and np.isfinite(b["mid"][j])):
                    continue
                if b["mid"][i] < b["mid"][j]:
                    keep[idx] = False
        out[f"FK{k}"] = run_filter(keep, f"FK  mid>=mid[-{k}] in crisis")

    print("\n[CC]", flush=True)
    out["CC"] = run_filter(~((crisis > Q90) & (gaps > CCQ)),
                           "CC  drop deep distress")

    print("\n[MS]", flush=True)
    keep = np.ones(len(pipe), bool)
    for idx, f in enumerate(pipe):
        if crisis[idx] > Q90:
            b = bonds[f.six]
            i = sig_i(b, f)
            if i >= 0 and b["mat"][i] > 3:
                keep[idx] = False
    out["MS"] = run_filter(keep, "MS  mat<=3 in crisis")

    # ---- [DE] deferred entry (rebuild entries from gated pre-exit fills) ----
    print("\n[DE]", flush=True)
    de = []
    n_moved = n_dropped = 0
    for f in v:
        b = bonds[f.six]
        i = sig_i(b, f)
        sd = int(b["day"][i]) if i >= 0 else f.entry_day
        if not (stress_at(sd) > Q90):
            de.append(f); continue
        day = b["day"]
        m60 = b.get("med60")
        sm = ~np.isnan(b["s_px"]); s_day = day[sm]; s_px = b["s_px"][sm]
        j0 = np.searchsorted(s_day, sd + 30, side="left")
        moved = None
        for j in range(j0, len(s_day)):
            if s_day[j] > sd + 44:
                break
            ed = int(s_day[j]); ep = float(s_px[j])
            ii = np.searchsorted(day, ed, side="left") - 1
            if ii < 0 or m60 is None or not (np.isfinite(b["mid"][ii])
                                             and np.isfinite(m60[ii])):
                continue
            if ep > b["mid"][ii] + 0.25:          # limit rule at new entry
                continue
            if (ep - float(m60[ii])) > -3.0:      # must still be >=3pts cheap
                continue
            moved = e2.Fill(f.six, ed, ep, f.exit_day, f.exit_px, f.coupon, f.stale)
            break
        if moved is not None:
            de.append(moved); n_moved += 1
        else:
            n_dropped += 1
    print(f"  moved {n_moved}, dropped {n_dropped} of {len(v)} gated entries",
          flush=True)
    de_pipe = real_coupons(bonds, exit_lagged(bonds, de))
    d2, n2, dl2 = book(bonds, de_pipe)
    st = report(d2, n2, dl2, "DE  crisis entries +30-44d", base, n=len(de_pipe))
    rr = np.array([f.ret for f in de_pipe])
    st["mean"] = float(rr.mean()); st["win"] = float((rr > 0).mean())
    out["DE"] = st

    # ---- [SO] overlay -------------------------------------------------------
    print("\n[SO]", flush=True)
    st_d = np.array([stress_at(int(t)) for t in days])
    ris_d = np.array([rising_at(int(t)) for t in days])
    expo = np.where((st_d > Q90) & ris_d, 0.5, 1.0)
    expo = np.roll(expo, 1); expo[0] = 1.0
    dl = expo * daily + (1 - expo) * rf
    out["SO"] = report(days, np.cumprod(1 + dl), dl, "SO  0.5x crisis&rising", base)

    p = ROOT / "research" / "bedrock_dd_screen2.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
