#!/usr/bin/env python3
"""IS-only test of the specialness signal.

Economic hypothesis: a security that is special/scarce (heavily borrowed from
the Fed) is richly priced in cash and will CHEAPEN relative to peers as the
specialness fades (it ages off-the-run, a new on-the-run is issued). So a high
specialness score predicts NEGATIVE duration-neutral relative return.

Because specialness is NOT in the price data, any predictability at execution
lag >= 1 cannot be bid-ask bounce (the failure mode that killed the pure-price
signals). That is the whole point of bringing in this dataset.

Measures, IS (<=2019) only:
  1. Information coefficient: rank-corr(special_t, forward duration-neutral
     idiosyncratic return over t+lag .. t+lag+h). Negative = hypothesis holds.
  2. A duration-neutral L/S book (long low-special, short high-special) forward
     return and gross Sharpe, across lag and horizon.

No costs here — this is a signal-existence test. Costs + repo carry are
charged in the full backtest only if a signal survives this stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
ROOT = SRC.parent

IS_END = pd.Timestamp("2019-12-31")


def duration_neutral_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (date,cusip) idiosyncratic return: cross-sectional residual of daily
    return regressed on modified duration (+ constant) each day. Removes the
    level/curve move so what's left is bond-specific rich/cheap drift."""
    df = panel.dropna(subset=["ret", "mod_dur"]).copy()
    out = np.full(len(df), np.nan)
    i = 0
    idx_all = []
    res_all = []
    for date, day in df.groupby("date"):
        r = day["ret"].values
        d = day["mod_dur"].values
        X = np.column_stack([np.ones_like(d), d])
        try:
            beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        except Exception:
            continue
        resid = r - X @ beta
        idx_all.append(day.index.values)
        res_all.append(resid)
    ridx = np.concatenate(idx_all)
    rval = np.concatenate(res_all)
    s = pd.Series(rval, index=ridx)
    panel = panel.copy()
    panel["r_idio"] = s.reindex(panel.index)
    return panel


def main() -> int:
    p = pd.read_parquet(ROOT / "data" / "processed" / "special_panel.parquet")
    p = p[p["sec_type"].isin(["MARKET BASED NOTE", "MARKET BASED BOND"])].copy()
    p = p[p["tsy_years"] >= 1.0]
    p = p.sort_values(["cusip", "date"]).reset_index(drop=True)

    print("computing duration-neutral idiosyncratic returns...")
    p = duration_neutral_returns(p)

    # wide matrices
    special = p.pivot_table(index="date", columns="cusip", values="special", aggfunc="first")
    ridio = p.pivot_table(index="date", columns="cusip", values="r_idio", aggfunc="first")
    dates = special.index
    is_mask = dates <= IS_END

    # forward cumulative idio return
    print(f"\n{'lag':>3} {'horizon':>7} {'IC(rank)':>9} {'LS_grossSharpe':>15} {'ann_ret':>8}")
    for lag in (1, 2, 5):
        for h in (5, 10, 21, 63):
            fwd = ridio.shift(-lag).rolling(h).sum().shift(-(h - 1))  # sum of t+lag..t+lag+h-1
            fwd = fwd.reindex(dates)
            # IC per day then average (IS)
            ics = []
            for t in dates[is_mask]:
                a = special.loc[t]
                b = fwd.loc[t]
                d = pd.concat([a, b], axis=1).dropna()
                if len(d) >= 20:
                    ics.append(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
            ic = float(np.nanmean(ics)) if ics else np.nan

            # weekly L/S book: long bottom-quintile special, short top-quintile
            wed = pd.Series(dates, index=dates)
            reb = pd.DatetimeIndex(wed.groupby(wed.index.to_period("W-WED")).max().values)
            reb = reb[reb <= IS_END]
            daily = {}
            for t in reb:
                if t not in special.index:
                    continue
                s = special.loc[t].dropna()
                if len(s) < 25:
                    continue
                n = max(int(len(s) * 0.2), 5)
                shorts = s.nlargest(n).index   # most special -> short (expect cheapen)
                longs = s.nsmallest(n).index    # least special -> long
                pos = list(dates).index(t)
                for hh in range(h):
                    j = pos + lag + hh
                    if j >= len(dates):
                        break
                    dj = dates[j]
                    rl = ridio.loc[dj].reindex(longs).mean()
                    rs = ridio.loc[dj].reindex(shorts).mean()
                    daily[dj] = daily.get(dj, 0.0) + (rl - rs) / h
            ds = pd.Series(daily).sort_index()
            sh = float(ds.mean() / ds.std() * np.sqrt(252)) if ds.std() > 0 else np.nan
            print(f"{lag:>3} {h:>7} {ic:>9.3f} {sh:>15.2f} {ds.mean()*252:>8.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
