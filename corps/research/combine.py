"""Sleeve combination layer (the MOSAIC architecture, frozen rules).

Inputs: per-sleeve fill lists (saved as JSON by the sleeve runners via
save_fills), each already honest (ask entries, bid exits). This layer:

1. Computes each sleeve's daily MTM return stream (engine2.mtm_nav) and
   resamples monthly.
2. Combines with inverse-vol weights on trailing 24 monthly marks (12-month
   minimum history; fixed priors before that), renormalized monthly.
3. Applies a book-level volatility target (5% ann.) with scale=min(1, tgt/vol),
   hard no-leverage cap; residual capital earns the 3M T-bill.
4. Reports honest stats: monthly Sharpe vs T-bill, CAGR, maxDD, per-sleeve
   correlation matrix.

The RULES travel to OOS; fitted weights never do.

  python corps/research/combine.py fills_a.json:0.4 fills_b.json:0.35 ...
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


def save_fills(fills, path):
    rows = [{"six": f.six, "ed": int(f.entry_day), "ep": float(f.entry_px),
             "xd": int(f.exit_day), "xp": float(f.exit_px),
             "cp": float(f.coupon), "st": bool(f.stale)} for f in fills]
    Path(path).write_text(json.dumps(rows))


def load_fills(path):
    rows = json.loads(Path(path).read_text())
    return [e2.Fill(r["six"], r["ed"], r["ep"], r["xd"], r["xp"], r["cp"], r["st"])
            for r in rows]


def sleeve_monthly(bonds, fills):
    r = e2.mtm_nav(bonds, fills)
    if r is None:
        return None
    days, nav, daily = r
    ts = pd.Series(daily, index=pd.to_datetime(days, unit="D"))
    return (1 + ts).resample("ME").prod() - 1


def combine(monthlies: dict[str, pd.Series], priors: dict[str, float],
            vol_target=0.05, lookback=24, min_hist=12):
    idx = None
    for s in monthlies.values():
        idx = s.index if idx is None else idx.union(s.index)
    M = pd.DataFrame({k: s.reindex(idx) for k, s in monthlies.items()})
    rf_daily = e2.load_rf(idx.values.astype("datetime64[D]").astype(np.int64))
    rf_m = pd.Series((1 + rf_daily / 365.0) ** 30 - 1, index=idx)
    book = []
    weights_hist = []
    for i, t in enumerate(idx):
        hist = M.iloc[max(0, i - lookback):i]
        w = {}
        for k in M.columns:
            h = hist[k].dropna()
            if len(h) >= min_hist and h.std() > 1e-9:
                w[k] = 1.0 / h.std()
            else:
                w[k] = priors.get(k, 1.0)
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}
        # book pre-scale return: sleeves with no obs this month => their weight
        # sits in cash at rf
        r = 0.0
        cash_w = 0.0
        for k, wk in w.items():
            v = M.at[t, k]
            if pd.isna(v):
                cash_w += wk
            else:
                r += wk * v
        r += cash_w * rf_m.iloc[i]
        # vol targeting on trailing book vol
        if len(book) >= min_hist:
            bv = np.std([b for b in book[-lookback:]]) * np.sqrt(12)
            scale = min(1.0, vol_target / bv) if bv > 1e-9 else 1.0
        else:
            scale = 1.0
        rs = scale * r + (1 - scale) * rf_m.iloc[i]
        book.append(rs)
        weights_hist.append(w)
    bs = pd.Series(book, index=idx)
    ex = bs - rf_m
    yrs = len(bs) / 12.0
    nav = (1 + bs).cumprod()
    peak = nav.cummax()
    out = {
        "cagr": float(nav.iloc[-1] ** (1 / yrs) - 1),
        "sharpe_m": float(ex.mean() / ex.std() * np.sqrt(12)) if ex.std() > 0 else None,
        "maxdd": float((nav / peak - 1).min()),
        "vol_ann": float(bs.std() * np.sqrt(12)),
        "corr": M.corr().round(2).to_dict(),
        "avg_weights": {k: float(np.mean([wh[k] for wh in weights_hist]))
                        for k in M.columns},
    }
    return out, bs


def main():
    bonds = e2.load_cache()
    monthlies, priors = {}, {}
    for arg in sys.argv[1:]:
        path, _, pr = arg.partition(":")
        name = Path(path).stem
        fills = load_fills(path)
        m = sleeve_monthly(bonds, fills)
        if m is None:
            print(f"  {name}: no NAV", flush=True)
            continue
        monthlies[name] = m
        priors[name] = float(pr) if pr else 1.0
        print(f"  {name}: {len(fills)} fills, {len(m)} months", flush=True)
    out, series = combine(monthlies, priors)
    print(json.dumps({k: v for k, v in out.items() if k != "corr"}, indent=1,
                     default=float))
    print("corr:", out["corr"])


if __name__ == "__main__":
    main()
