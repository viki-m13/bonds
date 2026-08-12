"""BEDROCK-V drawdown anatomy — MEASUREMENT ONLY (no strategy changes).

Question (pre-registered in BEDROCK_RESEARCH.md §9): where do the pipeline
book's -44% full / -37% OOS drawdowns come from, and which levers could cut
them WITHOUT giving back the upside?

  [A] Episode table: every drawdown deeper than -10% on the full-window
      pipeline NAV — peak/trough/recovery dates, depth, lengths.
  [B] Decomposition per episode: positions open at the peak (pre-crisis book,
      marked down) vs entries during the fall; eventual REALIZED returns of
      both cohorts — is the drawdown a permanent loss or a mark-to-market
      trough on trades that recover?
  [C] Upside attribution: share of total log NAV growth earned by trades
      ENTERED inside episode windows (peak -> recovery). This is the upside an
      entry-throttle would forfeit.
  [D] Beta structure: monthly book excess returns regressed on LQD and HYG
      excess (full, and episode months only); downside beta. Tells us whether
      a hedge can carry the drawdown.
  [E] Overlay diagnostics (full-window, DISCLOSED as in-sample measurement,
      not a backtest): (1) constant LQD beta-hedge, (2) vol-targeted exposure
      (no leverage), (3) LQD-drawdown de-risk trend filter. Reported only to
      rank levers for the pre-registered IS screen.

  python corps/research/bedrock_dd_diag.py
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
from combos import depth_of  # noqa: E402

MAXH = 455
FULL = (e2.D("2003-01-01"), e2.D("2025-03-31") - MAXH)
OOS = (e2.D("2016-01-01"), e2.D("2025-03-31") - MAXH)
ETF = ROOT / "data" / "etf"


def etf_daily(name, days):
    df = pd.read_csv(ETF / f"{name}.csv.gz", parse_dates=["date"])
    d = df["date"].values.astype("datetime64[D]").astype(np.int64)
    px = df["adjclose"].to_numpy(float)
    # daily return series aligned to `days`: return between consecutive
    # trading days is booked on the later day; non-trading days get 0
    ret = np.zeros(len(days))
    idx = np.searchsorted(d, days)
    for k, t in enumerate(days):
        j = np.searchsorted(d, t)
        if j < len(d) and d[j] == t and j > 0:
            ret[k] = px[j] / px[j - 1] - 1.0
    return ret


def ds(day):
    return str(np.int64(day).astype("datetime64[D]"))


def episodes(days, nav, floor=-0.10):
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    eps = []
    i = 0
    n = len(dd)
    while i < n:
        if dd[i] < 0:
            j = i
            while j < n and dd[j] < 0:
                j += 1
            depth_i = i + int(np.argmin(dd[i:j]))
            if dd[depth_i] <= floor:
                eps.append({"peak_day": int(days[i - 1]) if i > 0 else int(days[0]),
                            "trough_day": int(days[depth_i]),
                            "rec_day": int(days[j - 1]) if j < n else None,
                            "depth": float(dd[depth_i]),
                            "fall_d": int(days[depth_i] - days[i - 1]) if i > 0 else 0,
                            "rec_d": int(days[j - 1] - days[depth_i]) if j < n else None})
            i = j
        else:
            i += 1
    eps.sort(key=lambda e: e["depth"])
    return eps


def main():
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    med = build_cs_median(bonds).to_dict()
    base = cl_fills(bonds, *FULL)
    pipe = real_coupons(bonds, exit_lagged(
        bonds, gate_issuer_curve(bonds, issuers, gate_value(bonds, base, med))))
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in pipe]
    days, nav, daily = e2.mtm_nav(bonds, pipe, weights=w)
    ps = e2.perf_stats(days, nav, daily)
    print(f"pipeline full: n={len(pipe)} cagr={ps['cagr']*100:+.2f}% "
          f"sharpe_m={ps['sharpe_m']:.2f} dd={ps['maxdd']*100:.1f}%", flush=True)
    out = {"base": {**ps, "n": len(pipe)}}

    # [A] episodes ------------------------------------------------------------
    eps = episodes(days, nav)
    print("\n[A] drawdown episodes <= -10%:", flush=True)
    for e in eps:
        print(f"  peak {ds(e['peak_day'])} trough {ds(e['trough_day'])} "
              f"depth {e['depth']*100:6.1f}%  fall {e['fall_d']:4d}d  "
              f"recover {e['rec_d']}d", flush=True)
    out["episodes"] = eps

    # [B] decomposition of the worst 4 ---------------------------------------
    print("\n[B] episode decomposition (eventual realized returns):", flush=True)
    out["decomp"] = []
    for e in eps[:4]:
        pk, tr = e["peak_day"], e["trough_day"]
        open_at_pk = [f for f in pipe if f.entry_day <= pk < f.exit_day]
        entered_fall = [f for f in pipe if pk < f.entry_day <= tr]
        row = {"peak": ds(pk), "depth": e["depth"]}
        for tag, co in (("open_at_peak", open_at_pk), ("entered_in_fall", entered_fall)):
            if co:
                r = np.array([f.ret for f in co])
                row[tag] = {"n": len(co), "mean": float(r.mean()),
                            "win": float((r > 0).mean()),
                            "stale_share": float(np.mean([f.stale for f in co]))}
            else:
                row[tag] = {"n": 0}
        out["decomp"].append(row)
        o, g = row["open_at_peak"], row["entered_in_fall"]
        print(f"  {ds(pk)} ({e['depth']*100:.0f}%): open@peak n={o['n']} "
              f"realized {o.get('mean', 0)*100:+.1f}% (win {o.get('win', 0)*100:.0f}%) | "
              f"entered-in-fall n={g['n']} realized {g.get('mean', 0)*100:+.1f}% "
              f"(win {g.get('win', 0)*100:.0f}%)", flush=True)

    # [C] upside attribution to episode windows -------------------------------
    tot = float(np.sum(np.array([f.ret for f in pipe]) *
                       np.array(w) / np.sum(w)))
    in_ep = np.zeros(len(pipe), bool)
    for e in eps:
        hi = e["rec_day"] if e["rec_day"] else FULL[1]
        for k, f in enumerate(pipe):
            if e["peak_day"] < f.entry_day <= hi:
                in_ep[k] = True
    r_all = np.array([f.ret for f in pipe]); w_all = np.array(w)
    contrib_ep = float(np.sum(r_all[in_ep] * w_all[in_ep]) / np.sum(w_all))
    out["upside"] = {"trades_in_episode_windows": int(in_ep.sum()),
                     "share_of_trades": float(in_ep.mean()),
                     "wmean_ret_in_ep": float(np.average(r_all[in_ep], weights=w_all[in_ep])),
                     "wmean_ret_out_ep": float(np.average(r_all[~in_ep], weights=w_all[~in_ep])),
                     "contrib_share": contrib_ep / tot if tot else None}
    u = out["upside"]
    print(f"\n[C] {u['trades_in_episode_windows']} trades ({u['share_of_trades']*100:.0f}%) "
          f"entered inside episode windows; wmean ret {u['wmean_ret_in_ep']*100:+.2f}% vs "
          f"{u['wmean_ret_out_ep']*100:+.2f}% outside; contribution share "
          f"{u['contrib_share']*100:.0f}% of total", flush=True)

    # [D] beta structure -------------------------------------------------------
    print("\n[D] monthly beta of book excess vs ETF excess:", flush=True)
    rf = e2.load_rf(days) / 365.0
    ts = pd.Series(daily - rf, index=pd.to_datetime(days, unit="D"))
    bm = (1 + ts).resample("ME").prod() - 1
    out["beta"] = {}
    for name in ("LQD", "HYG"):
        er = etf_daily(name, days) - rf
        es = pd.Series(er, index=ts.index)
        em = (1 + es).resample("ME").prod() - 1
        m = pd.concat([bm, em], axis=1, keys=["b", "e"]).dropna()
        m = m[(m != 0).any(axis=1)]
        if len(m) < 12:
            continue
        cov = np.cov(m["b"], m["e"])
        beta = float(cov[0, 1] / cov[1, 1])
        corr = float(np.corrcoef(m["b"], m["e"])[0, 1])
        dn = m[m["e"] < 0]
        dncov = np.cov(dn["b"], dn["e"])
        out["beta"][name] = {"beta": beta, "corr": corr, "r2": corr ** 2,
                             "months": len(m),
                             "beta_dn": float(dncov[0, 1] / dncov[1, 1]),
                             "alpha_m": float(m["b"].mean() - beta * m["e"].mean())}
        d = out["beta"][name]
        print(f"  {name}: beta={d['beta']:.2f} (down {d['beta_dn']:.2f}) "
              f"corr={d['corr']:.2f} R2={d['r2']:.2f} alpha={d['alpha_m']*100:+.2f}%/mo "
              f"({d['months']} mo)", flush=True)

    # [E] overlay diagnostics (disclosed full-window measurement) --------------
    print("\n[E] overlay diagnostics (NOT a backtest — lever ranking only):", flush=True)
    out["overlay"] = {}

    def stats_of(dl, tag):
        nv = np.cumprod(1.0 + dl)
        p2 = e2.perf_stats(days, nv, dl)
        out["overlay"][tag] = p2
        print(f"  {tag:34} cagr={p2['cagr']*100:+6.2f}% sharpe_m={p2['sharpe_m']:5.2f} "
              f"dd={p2['maxdd']*100:6.1f}%", flush=True)

    lqd = etf_daily("LQD", days)
    for h in (0.5, 1.0, 1.5):
        stats_of(daily - h * (lqd - rf), f"hedge LQD beta*{h:.1f}")

    # vol targeting: trailing 63d realized vol of the book, cap exposure at 1
    dser = pd.Series(daily, index=pd.to_datetime(days, unit="D"))
    vol = dser.rolling(63).std() * np.sqrt(365)
    for tgt in (0.10, 0.12, 0.15):
        expo = (tgt / vol).clip(upper=1.0).shift(1).fillna(1.0).to_numpy()
        stats_of(expo * daily + (1 - expo) * rf, f"vol-target {int(tgt*100)}% (cap 1x)")

    # LQD trend de-risk: if LQD below its own trailing 200d max by >5%, half expo
    lqdpx = np.cumprod(1.0 + lqd)
    roll = pd.Series(lqdpx).rolling(200, min_periods=1).max().to_numpy()
    derisk = (lqdpx / roll - 1.0) < -0.05
    expo = np.where(np.roll(derisk, 1), 0.5, 1.0); expo[0] = 1.0
    stats_of(expo * daily + (1 - expo) * rf, "LQD -5% trend de-risk (0.5x)")

    p = ROOT / "research" / "bedrock_dd_diag.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
