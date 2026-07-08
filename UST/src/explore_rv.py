#!/usr/bin/env python3
"""IS-only exploration: is there REAL (non-bid-ask-bounce) relative-value
convergence in individual Treasuries?

For a cheapness signal (yield residual vs a local/curve fit), we measure the
forward return of a duration-neutral long-cheap/short-rich book at several
holding horizons and, crucially, at several EXECUTION LAGS.

The honesty test: lag=1 captures next-day reversion (partly uncapturable
bid-ask bounce in EOD marks). If the edge only exists at lag=1 and collapses
at lag=2/3 (skip a day before trading), it was mark noise, not tradeable RV.
An edge that persists at lag>=2 over multi-day horizons is real convergence.

Runs on IS data (<=2019-12-31) only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
ROOT = SRC.parent
import strategies as st

IS_END = pd.Timestamp("2019-12-31")


def local_residual(ss: st.SignalSet, k: int = 6) -> pd.DataFrame:
    """Yield minus the duration-weighted mean yield of the k nearest-maturity
    coupon neighbours on each date. Pure LOCAL cheapness (no curve model).
    Positive = cheap."""
    ytm = ss.ytm
    tsy = ss.tsy
    out = pd.DataFrame(np.nan, index=ytm.index, columns=ytm.columns)
    for t in ytm.index:
        y = ytm.loc[t].dropna()
        if len(y) < k + 2:
            continue
        m = tsy.loc[t].reindex(y.index)
        order = m.sort_values()
        ys = y.reindex(order.index).values
        # rolling neighbourhood mean excluding self
        n = len(ys)
        res = np.full(n, np.nan)
        for i in range(n):
            lo, hi = max(0, i - k), min(n, i + k + 1)
            idx = [j for j in range(lo, hi) if j != i]
            res[i] = ys[i] - np.mean(ys[idx])
        out.loc[t, order.index] = res
    return out


def fwd_return(ss: st.SignalSet, sig: pd.DataFrame, dates, lag: int, horizon: int,
               frac: float = 0.2) -> pd.Series:
    """Daily-equivalent return series of a duration-neutral L/S book formed on
    `sig` at date t, entered `lag` days later, held `horizon` days, using
    per-name daily returns. Overlapping books are averaged (like h independent
    tranches)."""
    ret = ss.ret
    dur = ss.dur
    trad = ss.tradeable_liq
    rebs = list(dates)
    # build target weights at each formation date, shifted by lag
    date_arr = np.array(sorted(ret.index))
    pos = {d: i for i, d in enumerate(date_arr)}
    daily = pd.Series(0.0, index=date_arr)
    count = pd.Series(0.0, index=date_arr)
    for t in rebs:
        if t not in pos:
            continue
        i0 = pos[t] + lag
        if i0 + horizon >= len(date_arr):
            continue
        s = sig.loc[t]
        ok = trad.loc[t].fillna(False).astype(bool) if t in trad.index else None
        if ok is None:
            continue
        s = s[ok.reindex(s.index).fillna(False)].dropna()
        if len(s) < 20:
            continue
        d = dur.loc[t].reindex(s.index)
        n = max(int(len(s) * frac), 5)
        longs, shorts = s.nlargest(n).index, s.nsmallest(n).index

        def side(names, sign):
            dd = d.reindex(names).clip(lower=0.5)
            w = (1.0 / dd); w = w / w.sum()
            return sign * w
        wl, ws = side(longs, 1.0), side(shorts, -1.0)
        dl = float((wl * d.reindex(wl.index)).sum())
        dsh = float(-(ws * d.reindex(ws.index)).sum())
        if dsh > 0:
            ws = ws * (dl / dsh)
        w = pd.concat([wl, ws]); w = w.groupby(w.index).sum()
        g = w.abs().sum()
        if g > 0:
            w = w * (2.0 / g)
        # accrue this tranche's daily pnl over the holding window
        for hh in range(horizon):
            di = date_arr[i0 + hh]
            r = ret.loc[di].reindex(w.index).fillna(0.0)
            daily.loc[di] += float((w * r).sum()) / horizon
            count.loc[di] += 1.0 / horizon
    d = daily[count > 0]
    return d


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if r.std() == 0 or len(r) < 50:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(252))


def main() -> int:
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    fits = pd.read_parquet(ROOT / "data" / "processed" / "cache_full" / "curve_fits.parquet")
    params = pd.read_parquet(ROOT / "data" / "processed" / "cache_full" / "curve_params.parquet")
    ss = st.SignalSet(panel, fits, params)

    is_dates = ss.ret.index[ss.ret.index <= IS_END]
    reb = st.weekly_rebalance_dates(is_dates)

    print("computing signals (IS)...")
    sig_curve = ss.resid.loc[is_dates]                 # cheap vs NSS curve
    sig_local = local_residual(ss).loc[is_dates]       # cheap vs maturity neighbours

    print(f"\n{'signal':10} {'frac':>4} {'lag':>3} {'horizon':>7} {'IS_Sharpe':>10} {'ann_ret':>8}")
    for name, sig in [("curve", sig_curve), ("local", sig_local)]:
        for frac in (0.1, 0.2):
            for lag in (1, 2, 3):
                for horizon in (1, 5, 10, 21):
                    r = fwd_return(ss, sig, reb, lag, horizon, frac)
                    sh = sharpe(r)
                    ar = r.mean() * 252 if len(r) else np.nan
                    print(f"{name:10} {frac:>4} {lag:>3} {horizon:>7} {sh:>10.2f} {ar:>8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
