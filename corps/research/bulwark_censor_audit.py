"""BULWARK audit — the right-censoring path in e2.run_events, which matters
FAR more for a default-focused book than for the reversion books.

engine2.run_events exit logic: first customer-bid print in [entry+365,
entry+455]; otherwise the LAST bid print at or before entry+455, dated AT
entry+455 and flagged stale. That fallback never checks the print is after
the entry. So a bond that defaults and stops printing can be booked as:
  * exit price from BEFORE the entry (a mark that never existed post-entry),
  * dated 455 days later, with 455 days of COUPON ACCRUED on a bond that
    stopped paying.
Both flatter defaults. This quantifies it on the BULWARK baseline book.

  python corps/research/bulwark_censor_audit.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bulwark_v1 import (u_mask, build_xsec, sig_factory, real_coupons, IS,  # noqa: E402
                        MAXH, limit_filter, issuer_cap_filter)


def diagnose(bonds, fills, lab):
    rows = []
    for f in fills:
        b = bonds[f.six]
        pm = ~np.isnan(b["p_px"]); p_day = b["day"][pm]; p_px = b["p_px"][pm]
        after = p_day[(p_day > f.entry_day) & (p_day <= f.entry_day + MAXH)]
        last_real = int(after.max()) if len(after) else None
        rows.append({"six": f.six, "entry": f.entry_day, "exit": f.exit_day,
                     "stale": f.stale, "ret": f.ret,
                     "has_post_entry_bid": last_real is not None,
                     "last_real_bid": last_real,
                     "gap_days": (f.exit_day - last_real) if last_real else None,
                     "bust": bool(lab.get(f.six, (False, 0))[0])})
    return pd.DataFrame(rows)


def honest_exit(bonds, fills, recovery=40.0):
    """Exit at the first bid >=365d; else the LAST REAL bid strictly after
    entry (at its own date, coupon accrued only to that date); else the
    position is a credit loss booked at min(last mid after entry, recovery)."""
    out, n_last, n_rec = [], 0, 0
    for f in fills:
        b = bonds[f.six]
        day = b["day"]
        pm = ~np.isnan(b["p_px"]); p_day = day[pm]; p_px = b["p_px"][pm]
        k = np.searchsorted(p_day, f.entry_day + 365, side="left")
        if k < len(p_day) and p_day[k] <= f.entry_day + MAXH:
            xd, xp, st = int(p_day[k]), float(p_px[k]), False
        else:
            m = (p_day > f.entry_day) & (p_day <= f.entry_day + MAXH)
            if m.any():
                j = np.flatnonzero(m)[-1]
                xd, xp, st = int(p_day[j]), float(p_px[j]), True
                n_last += 1
            else:
                mm = ~np.isnan(b["mid"])
                md = day[mm]; mv = b["mid"][mm]
                sel = (md > f.entry_day) & (md <= f.entry_day + MAXH)
                px = float(mv[sel][-1]) if sel.any() else recovery
                xd = int(md[sel][-1]) if sel.any() else f.entry_day + MAXH
                xp, st = min(px, recovery), True
                n_rec += 1
        out.append(e2.Fill(f.six, f.entry_day, f.entry_px, xd, xp, f.coupon, st))
    return out, n_last, n_rec


def main():
    bonds = e2.load_cache()
    print(f"loaded {len(bonds)} bonds", flush=True)
    ldf = pd.read_parquet(ROOT / "research" / "bulwark_labels.parquet")
    lab = {r.six: (bool(r.bust), int(r.last_day)) for r in ldf.itertuples()}
    q97, amed = build_xsec(bonds)
    raw = e2.run_events(bonds, sig_factory(bonds, q97, amed), min_hold=365,
                        max_hold=MAXH, date_lo=IS[0], date_hi=IS[1])
    raw = issuer_cap_filter(limit_filter(bonds, raw, cap=0.25), cap=1)
    fills = real_coupons(bonds, raw)
    d = diagnose(bonds, fills, lab)
    print(f"\nbaseline IS book: n={len(d)}")
    print(f"  stale-exit fills            : {d.stale.sum()} ({d.stale.mean()*100:.1f}%)")
    bad = d[~d.has_post_entry_bid]
    print(f"  NO bid print after entry    : {len(bad)}  <-- exit price predates entry")
    print(f"     of which labelled bust   : {int(bad.bust.sum())}")
    print(f"     their booked mean return : {bad.ret.mean()*100:+.2f}%" if len(bad) else "")
    st = d[d.stale & d.has_post_entry_bid]
    if len(st):
        print(f"  stale w/ real bid earlier   : {len(st)}, median gap "
              f"{st.gap_days.median():.0f}d (coupon accrued over the gap)")
    bt = d[d.bust]
    print(f"  bust-linked fills           : {len(bt)}, booked mean "
          f"{bt.ret.mean()*100:+.2f}%, stale share {bt.stale.mean()*100:.0f}%")

    hf, n_last, n_rec = honest_exit(bonds, raw)
    hf = real_coupons(bonds, hf)
    r0 = np.array([f.ret for f in fills]); r1 = np.array([f.ret for f in hf])
    print(f"\nhonest exit: {n_last} repriced to last real bid, {n_rec} booked "
          f"as credit losses at min(last mid, 40)")
    print(f"  mean/trade  {r0.mean()*100:+.2f}%  ->  {r1.mean()*100:+.2f}%")
    print(f"  win rate    {(r0>0).mean()*100:.0f}%  ->  {(r1>0).mean()*100:.0f}%")
    print(f"  ret<-20%    {(r0<-.2).mean()*100:.1f}% ->  {(r1<-.2).mean()*100:.1f}%")
    bm = d.bust.values
    if bm.sum():
        print(f"  bust-linked mean {r0[bm].mean()*100:+.2f}% -> {r1[bm].mean()*100:+.2f}%")
    for tag, r, fl in (("current", r0, fills), ("honest", r1, hf)):
        days, nav, daily = e2.mtm_nav(bonds, fl)
        ps = e2.perf_stats(days, nav, daily)
        print(f"  [{tag:7}] cagr={ps['cagr']*100:+6.2f}% sharpe={ps['sharpe_m']:5.2f} "
              f"dd={ps['maxdd']*100:6.1f}%")
    out = {"n": len(d), "stale": int(d.stale.sum()),
           "no_post_entry_bid": int(len(bad)),
           "bad_bust": int(bad.bust.sum()),
           "mean_current": float(r0.mean()), "mean_honest": float(r1.mean()),
           "repriced_last_bid": n_last, "booked_loss": n_rec}
    (ROOT / "research" / "bulwark_censor_audit.json").write_text(json.dumps(out, default=float))
    print("\nwrote bulwark_censor_audit.json", flush=True)


if __name__ == "__main__":
    main()
