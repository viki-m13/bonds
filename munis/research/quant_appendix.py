"""Quant appendix for KEYSTONE-XL (munis) — same decomposition and diagnostics
as the corporate appendix, on the muni engine's conventions (NAV via linear
intra-trade attribution; benchmark MUB). Writes docs/quant_muni.json.

  python munis/research/quant_appendix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
from strategies import FACTORIES  # noqa: E402
from limit_transfer import load_bonds, limit_filter  # noqa: E402
from keystone_xl import issuer_cap, recovery_exit, IS_LO, OOS_HI  # noqa: E402
from xl_equity import nav_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parents[2] / "docs"


def last_mid(g, ts):
    s = g.set_index("date")["mid"]
    try:
        v = s.asof(ts)
    except Exception:
        return np.nan
    return float(v) if np.isfinite(v) else np.nan


def main():
    bonds = load_bonds()
    print(f"{len(bonds)} muni bonds", flush=True)
    fn = FACTORIES["price_discount"](discount=3.0)
    base = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                         date_lo=IS_LO, date_hi=OOS_HI)
    fills = recovery_exit(bonds, issuer_cap(limit_filter(bonds, base)))
    rows = []
    for f in fills:
        g = bonds[f.six]
        em = last_mid(g, f.entry_date); xm = last_mid(g, f.exit_date)
        if not (np.isfinite(em) and np.isfinite(xm)):
            continue
        carry = f.coupon / 100 / 365 * f.hold_days * 100 / f.entry_px
        rows.append({"ret": f.ret, "carry": carry,
                     "edge": (em - f.entry_px) / f.entry_px,
                     "rev": (xm - em) / f.entry_px,
                     "xcost": (f.exit_px - xm) / f.entry_px,
                     "hold": f.hold_days, "year": f.entry_date.year,
                     "six": f.six})
    df = pd.DataFrame(rows)
    print(f"XL fills {len(fills)}, decomposed {len(df)}", flush=True)

    decomp = {k: {"mean": float(df[k].mean()),
                  "share": float(df[k].mean() / df["ret"].mean())}
              for k in ("carry", "edge", "rev", "xcost")}
    decomp["total"] = {"mean": float(df["ret"].mean()), "share": 1.0}
    print("decomposition:", {k: round(v["mean"] * 100, 2) for k, v in decomp.items()}, flush=True)

    r = df["ret"].to_numpy()
    q = np.percentile(r, [1, 5, 25, 50, 75, 95, 99])
    dist = {"mean": float(r.mean()), "std": float(r.std()),
            "skew": float(df["ret"].skew()), "kurt": float(df["ret"].kurt()),
            "win": float((r > 0).mean()), "p1": float(q[0]), "p5": float(q[1]),
            "p25": float(q[2]), "p50": float(q[3]), "p75": float(q[4]),
            "p95": float(q[5]), "p99": float(q[6]),
            "cvar5": float(r[r <= q[1]].mean()),
            "best": float(r.max()), "worst": float(r.min())}

    nav = nav_series(fills)
    m = nav.resample("ME").last().pct_change().dropna()
    mub = (pd.read_csv(ROOT / "data" / "mub_daily.csv.gz", parse_dates=["date"])
           .set_index("date").iloc[:, 0])
    mub_m = mub.resample("ME").last().pct_change().dropna()
    j = m.index.intersection(mub_m.index)
    a, bb = m.reindex(j), mub_m.reindex(j)
    beta = float(np.cov(a, bb)[0, 1] / np.var(bb))
    dn = m[m < 0]
    monthly = {"vol_ann": float(m.std() * np.sqrt(12)),
               "sharpe_gross": float(m.mean() / m.std() * np.sqrt(12)),
               "sortino": float(m.mean() / dn.std() * np.sqrt(12)) if len(dn) > 2 else None,
               "skew": float(m.skew()), "kurt": float(m.kurt()),
               "hit": float((m > 0).mean()), "best_m": float(m.max()),
               "worst_m": float(m.min()), "beta": beta,
               "corr": float(np.corrcoef(a, bb)[0, 1]),
               "alpha_ann": float((a.mean() - beta * bb.mean()) * 12),
               "note": "linear-attribution NAV (munis do not print daily) — vol/Sharpe optimistic; per-trade rows are realized fills"}
    peak = nav.cummax(); dd = nav / peak - 1
    eps = []
    in_dd = False
    for t, v in dd.items():
        if v < -0.002 and not in_dd:
            in_dd = True; start = t; trough = t; depth = v
        elif in_dd:
            if v < depth:
                depth = v; trough = t
            if v >= 0:
                eps.append((start, trough, t, depth)); in_dd = False
    if in_dd:
        eps.append((start, trough, None, depth))
    eps.sort(key=lambda e: e[3])
    dds = [{"peak": s.strftime("%Y-%m"), "trough": tr.strftime("%Y-%m"),
            "recovered": rec.strftime("%Y-%m") if rec is not None else "open",
            "depth": float(d)} for s, tr, rec, d in eps[:3]]

    yr_nav = nav.resample("YE").last().pct_change()
    yr_nav.iloc[0] = nav.resample("YE").last().iloc[0] - 1
    ann = []
    for y, v in yr_nav.items():
        sub = df[df["year"] == y.year]
        ann.append({"year": int(y.year), "nav_ret": float(v),
                    "n": int(len(sub)),
                    "mean": float(sub["ret"].mean()) if len(sub) else None})

    by_iss = df.groupby(df["six"].str[:6]).size()
    conc = {"n_issuers": int(by_iss.size),
            "top_share": float(by_iss.max() / len(df)),
            "top5_share": float(by_iss.nlargest(5).sum() / len(df))}
    entry = {"hold_med": float(df["hold"].median()),
             "hold_mean": float(df["hold"].mean())}

    out = {"decomp": decomp, "dist": dist, "monthly": monthly,
           "drawdowns": dds, "annual": ann, "concentration": conc,
           "entry": entry, "n": int(len(df))}
    (DOCS / "quant_muni.json").write_text(json.dumps(out, default=float))
    print("monthly:", {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in monthly.items()}, flush=True)
    print("wrote docs/quant_muni.json", flush=True)


if __name__ == "__main__":
    main()
