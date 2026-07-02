"""New sleeve F — multi-asset time-series momentum (Moskowitz/Faber style),
long-flat, monthly, inverse-vol weights, on liquid ETFs, with costs.

Assets: US/intl equities, small caps, bonds, credit, gold, silver, broad
commodities, oil, REITs, dollar, China. Long when 12-1 momentum > 0 at the
month-end signal close, else flat that asset. Inverse-63d-vol weights over
the "on" assets, 25% single-asset cap, no leverage.

Execution: MOC t vs next close (should be insensitive -- measured).
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_tiingo, riskfree_daily, stats, fmt, OUT

ASSETS = ["SPY", "IWM", "EFA", "EEM", "FXI", "VNQ", "TLT", "IEF", "LQD",
          "HYG", "GLD", "SLV", "DBC", "USO", "UUP"]
COST_BPS = 2.0


def run(lag=0, costs=True, start="2008-01-31"):
    ac, _ = load_tiingo()
    px = ac[ASSETS].astype("float64").loc["2003":]
    rets = px.pct_change(fill_method=None)
    vol63 = rets.rolling(63, min_periods=40).std()
    me = px.groupby(px.index.to_period("M")).tail(1).index
    me = me[me >= pd.Timestamp(start)]
    daily, prev_w, expo = [], pd.Series(0.0, index=ASSETS), []
    for k in range(1, len(me)):
        d_sig, d_next = me[k - 1], me[k]
        p = px.index.searchsorted(d_sig)
        mom = px.iloc[p - 21] / px.iloc[p - 252] - 1
        on = mom[mom > 0].index
        on = [a for a in on if np.isfinite(vol63.iloc[p][a])]
        if len(on) == 0:
            w = pd.Series(0.0, index=ASSETS)
        else:
            iv = 1.0 / vol63.iloc[p][on]
            w = (iv / iv.sum()).clip(upper=0.25)
            w = w.reindex(ASSETS).fillna(0.0)
        i_from = p + 1 + lag
        i_to = min(px.index.searchsorted(d_next) + lag, len(px) - 1)
        seg = rets.iloc[i_from:i_to + 1].fillna(0)
        r = (seg * w).sum(axis=1)
        if costs and len(r):
            r.iloc[0] -= (w - prev_w).abs().sum() * COST_BPS / 1e4
        daily.append(r)
        expo.append(pd.Series(w.sum(), index=r.index))
        prev_w = w
    r = pd.concat(daily)
    e = pd.concat(expo)
    return r[~r.index.duplicated()], e[~e.index.duplicated()]


if __name__ == "__main__":
    t0 = time.time()
    res, expos = {}, {}
    for label, kw in [("tsmom 15-ETF MOC t", dict(lag=0)),
                      ("tsmom 15-ETF close t+1", dict(lag=1)),
                      ("tsmom FREE", dict(costs=False))]:
        r, e = run(**kw)
        rf = riskfree_daily(r.index)
        print(fmt(stats(r, rf, label)) + f"  expo {e.mean()*100:.0f}%")
        res[label] = r
        expos[label] = e
    pd.DataFrame({"tsmom": res["tsmom 15-ETF MOC t"],
                  "tsmom_lag": res["tsmom 15-ETF close t+1"]}) \
        .to_parquet(os.path.join(OUT, "sleeveF_tsmom.parquet"))
    pd.DataFrame({"tsmom": expos["tsmom 15-ETF MOC t"]}) \
        .to_parquet(os.path.join(OUT, "sleeveF_tsmom_expo.parquet"))
    print(f"saved -> out/sleeveF_tsmom.parquet  t={time.time()-t0:.0f}s")
