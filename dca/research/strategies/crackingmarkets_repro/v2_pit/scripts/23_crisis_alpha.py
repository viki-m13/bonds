"""Sleeve D — crisis-alpha trend sleeve on defensive assets, with costs.

Original spec: TLT/IEF/GLD long when 12-1 momentum > 0, equal weight,
monthly. Extensions explored (documented trend-following, not fitted):
  - broader defensive set incl. UUP (dollar) when available
  - execution at signal close t (MOC) vs next close (trend is slow; the lag
    should NOT matter -- that's the point of measuring it)
Costs: ETF auction fills (spread ~1bp) + commission; turnover is tiny.
Uses Tiingo adjCloses (total return) 2003+; ETF OHLC csvs only run 2005+.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_tiingo, riskfree_daily, stats, fmt, OUT

t0 = time.time()
ac, _ = load_tiingo()
ASSETS = ["TLT", "IEF", "GLD"]
px = ac[ASSETS].astype("float64").dropna(how="all").loc["2003":]
rets = px.pct_change(fill_method=None)

month_ends = px.groupby(px.index.to_period("M")).tail(1).index
COST_BPS = 2.0  # commission+auction noise per side, ETFs


def run(assets=ASSETS, lag=0, costs=True, start="2005-11-30"):
    me = month_ends[month_ends >= pd.Timestamp(start)]
    daily, expo = [], []
    prev_w = pd.Series(0.0, index=assets)
    for k in range(1, len(me)):
        d_sig, d_next = me[k - 1], me[k]
        p = px.index.searchsorted(d_sig)
        mom = px.iloc[p - 21][assets] / px.iloc[p - 252][assets] - 1
        on = mom > 0
        w = pd.Series(np.where(on, 1.0 / len(assets), 0.0), index=assets)
        i_from = px.index.searchsorted(d_sig) + 1 + lag
        i_to = min(px.index.searchsorted(d_next) + lag, len(px) - 1)
        seg = rets.iloc[i_from:i_to + 1][assets].fillna(0)
        r = (seg * w).sum(axis=1)
        if costs and len(r):
            turn = (w - prev_w).abs().sum()
            r.iloc[0] -= turn * COST_BPS / 1e4
        daily.append(r)
        expo.append(pd.Series(w.sum(), index=r.index))
        prev_w = w
    r = pd.concat(daily)
    e = pd.concat(expo)
    return r[~r.index.duplicated()], e[~e.index.duplicated()]


if __name__ == "__main__":
    rf = riskfree_daily(px.index)
    print("Sleeve D: crisis-alpha trend (2005-2026)")
    res, expos = {}, {}
    for label, kw in [
        ("TLT/IEF/GLD MOC t, costed",      dict(lag=0)),
        ("TLT/IEF/GLD close t+1, costed",  dict(lag=1)),
        ("TLT/IEF/GLD FREE",               dict(costs=False)),
    ]:
        r, e = run(**kw)
        st = stats(r, rf, label)
        print(fmt(st) + f"  expo {e.mean()*100:.0f}%")
        res[label] = r
        expos[label] = e
    keep = pd.DataFrame({"crisis": res["TLT/IEF/GLD MOC t, costed"],
                         "crisis_lag": res["TLT/IEF/GLD close t+1, costed"]})
    keep.to_parquet(os.path.join(OUT, "sleeveD_crisis.parquet"))
    pd.DataFrame({"crisis": expos["TLT/IEF/GLD MOC t, costed"]}) \
        .to_parquet(os.path.join(OUT, "sleeveD_crisis_expo.parquet"))
    print(f"saved -> out/sleeveD_crisis.parquet  t={time.time()-t0:.0f}s")
