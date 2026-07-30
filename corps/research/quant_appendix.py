"""Quant appendix for GRANITE-XL — everything a quant would ask.

Per-trade return decomposition (identity, sums exactly to total):
  total = carry + entry_edge + reversion + exit_cost
    carry      = coupon accrual over the hold / entry price
    entry_edge = (entry-day mid - entry ask) / entry px   [discount captured]
    reversion  = (exit-day mid - entry-day mid) / entry px [the alpha engine]
    exit_cost  = (exit bid - exit-day mid) / entry px      [spread paid out]

Plus: distributions, monthly moments (Sharpe/Sortino/skew/kurt/hit),
drawdown episodes, beta/alpha vs LQD, annual table, concentration, capacity.
Writes docs/quant_corp.json.

  python corps/research/quant_appendix.py
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
from flowml import xl_fills  # noqa: E402
from combos import depth_of  # noqa: E402

DOCS = ROOT.parent / "docs"


def last_mid(b, day_i):
    day = b["day"]
    i = np.searchsorted(day, day_i, side="right") - 1
    while i >= 0 and not np.isfinite(b["mid"][i]):
        i -= 1
    return float(b["mid"][i]) if i >= 0 else np.nan


def decompose(bonds, fills):
    rows = []
    for f in fills:
        b = bonds[f.six]
        em = last_mid(b, f.entry_day)
        xm = last_mid(b, f.exit_day)
        if not (np.isfinite(em) and np.isfinite(xm)):
            continue
        carry = f.coupon / 100 / 365 * f.hold * 100 / f.entry_px
        entry_edge = (em - f.entry_px) / f.entry_px
        reversion = (xm - em) / f.entry_px
        exit_cost = (f.exit_px - xm) / f.entry_px
        rows.append({"ret": f.ret, "carry": carry, "edge": entry_edge,
                     "rev": reversion, "xcost": exit_cost, "hold": f.hold,
                     "year": 1970 + f.entry_day // 365, "six": f.six,
                     "depth": depth_of(bonds, f)})
    return pd.DataFrame(rows)


def dist_stats(r):
    r = np.asarray(r)
    q = np.percentile(r, [1, 5, 25, 50, 75, 95, 99])
    return {"mean": float(r.mean()), "std": float(r.std()),
            "skew": float(pd.Series(r).skew()), "kurt": float(pd.Series(r).kurt()),
            "win": float((r > 0).mean()),
            "p1": float(q[0]), "p5": float(q[1]), "p25": float(q[2]),
            "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
            "p99": float(q[6]), "cvar5": float(r[r <= q[1]].mean()),
            "best": float(r.max()), "worst": float(r.min())}


def drawdown_episodes(days, nav, top=3):
    ts = pd.Series(nav, index=pd.to_datetime(days, unit="D"))
    peak = ts.cummax()
    dd = ts / peak - 1
    eps = []
    in_dd = False
    for t, v in dd.items():
        if v < -0.005 and not in_dd:
            in_dd = True; start = t; trough = t; depth = v
        elif in_dd:
            if v < depth:
                depth = v; trough = t
            if v >= 0:
                eps.append((start, trough, t, depth)); in_dd = False
    if in_dd:
        eps.append((start, trough, None, depth))
    eps.sort(key=lambda e: e[3])
    return [{"peak": s.strftime("%Y-%m"), "trough": tr.strftime("%Y-%m"),
             "recovered": (rec.strftime("%Y-%m") if rec is not None else "open"),
             "depth": float(d)} for s, tr, rec, d in eps[:top]]


def monthly_stats(days, nav, daily, bench_m=None):
    ts = pd.Series(daily, index=pd.to_datetime(days, unit="D"))
    m = (1 + ts).resample("ME").prod() - 1
    rf = pd.Series((1 + e2.load_rf(days) / 365).cumprod(),
                   index=pd.to_datetime(days, unit="D")).resample("ME").last().pct_change().fillna(0)
    ex = (m - rf).dropna()
    dn = ex[ex < 0]
    out = {"vol_ann": float(m.std() * np.sqrt(12)),
           "sharpe": float(ex.mean() / ex.std() * np.sqrt(12)),
           "sortino": float(ex.mean() / dn.std() * np.sqrt(12)) if len(dn) > 2 else None,
           "skew": float(m.skew()), "kurt": float(m.kurt()),
           "hit": float((m > 0).mean()),
           "best_m": float(m.max()), "worst_m": float(m.min()),
           "roll12_best": float(((1 + m).rolling(12).apply(np.prod) - 1).max()),
           "roll12_worst": float(((1 + m).rolling(12).apply(np.prod) - 1).min())}
    if bench_m is not None:
        j = m.index.intersection(bench_m.index)
        a, bb = m.reindex(j), bench_m.reindex(j)
        beta = float(np.cov(a, bb)[0, 1] / np.var(bb))
        out["beta"] = beta
        out["corr"] = float(np.corrcoef(a, bb)[0, 1])
        out["alpha_ann"] = float((a.mean() - beta * bb.mean()) * 12)
    return out


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    fills = xl_fills(bonds)
    w = [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fills]
    df = decompose(bonds, fills)
    print(f"XL fills {len(fills)}, decomposed {len(df)}", flush=True)

    decomp = {k: {"mean": float(df[k].mean()),
                  "share": float(df[k].mean() / df["ret"].mean())}
              for k in ("carry", "edge", "rev", "xcost")}
    decomp["total"] = {"mean": float(df["ret"].mean()), "share": 1.0}
    print("decomposition:", {k: round(v["mean"] * 100, 2) for k, v in decomp.items()}, flush=True)

    days, nav, daily = e2.mtm_nav(bonds, fills, weights=w)
    lqd = (pd.read_csv(ROOT / "data" / "etf" / "LQD.csv.gz", parse_dates=["date"])
           .set_index("date")["adjclose"])
    lqd_m = lqd.resample("ME").last().pct_change().dropna()
    ms = monthly_stats(days, nav, daily, bench_m=lqd_m)
    dds = drawdown_episodes(days, nav)

    ann = []
    ts = pd.Series(daily, index=pd.to_datetime(days, unit="D"))
    yr_nav = (1 + ts).resample("YE").prod() - 1
    for y, v in yr_nav.items():
        sub = df[df["year"] == y.year]
        ann.append({"year": int(y.year), "nav_ret": float(v),
                    "n": int(len(sub)),
                    "mean": float(sub["ret"].mean()) if len(sub) else None})

    by_iss = df.groupby(df["six"].str[:6]).size()
    conc = {"n_issuers": int(by_iss.size),
            "top_share": float(by_iss.max() / len(df)),
            "top5_share": float(by_iss.nlargest(5).sum() / len(df))}
    qv = []
    for f in fills:
        b = bonds[f.six]
        i = min(max(np.searchsorted(b["day"], f.entry_day), 0), len(b["day"]) - 1)
        qv.append(float(b["qv90"][i]))
    capacity = {"med_qv90_mm": float(np.nanmedian(qv)),
                "p75_qv90_mm": float(np.nanpercentile(qv, 75))}
    entry_traits = {"depth_med": float(df["depth"].median()),
                    "hold_med": float(df["hold"].median()),
                    "hold_mean": float(df["hold"].mean())}

    out = {"decomp": decomp, "dist": dist_stats(df["ret"]),
           "monthly": ms, "drawdowns": dds, "annual": ann,
           "concentration": conc, "capacity": capacity,
           "entry": entry_traits, "n": int(len(df))}
    (DOCS / "quant_corp.json").write_text(json.dumps(out, default=float))
    print("monthly:", {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in ms.items()}, flush=True)
    print("wrote docs/quant_corp.json", flush=True)


if __name__ == "__main__":
    main()
