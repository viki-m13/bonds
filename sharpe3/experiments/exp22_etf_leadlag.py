"""Exp22: ETF late-day flow -> next-day beta cross-section on stocks.

Signal: QQQ/SPY 14:30->15:55 move (5-min data). Position: if late-day move up,
long high-beta vs short low-beta S&P names next day (flow continuation);
also the reverse (fade). Next-open execution, 5bps.
"""
import os, sys, json
import datetime as dtm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, open_, member = P["close"], P["open"], P["member"]
r1 = close.pct_change(fill_method=None)
mkt = r1.where(member).mean(axis=1)
cov = r1.rolling(60).cov(mkt); var = mkt.rolling(60).var()
beta = cov.div(var, axis=0)

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

bz = zs(beta.where(member))
w_beta = bt.norm_ls(bz, member, 0.2, 0.2, 2.0)

out = {}
for etf in ("QQQ", "SPY"):
    df = pd.read_csv(os.path.join(ROOT, "..", "data", "intraday_5min", f"{etf}.csv"), parse_dates=["ts"])
    df["date"] = df["ts"].dt.normalize()
    df["t"] = df["ts"].dt.time
    pv = df.pivot_table(index="date", columns="t", values="close")
    late = (pv[dtm.time(15, 55)] / pv[dtm.time(14, 30)] - 1).reindex(close.index)
    sgn = np.sign(late).fillna(0.0)
    for nm, s in ((f"{etf}_late_cont", sgn), (f"{etf}_late_fade", -sgn)):
        w = w_beta.mul(s, axis=0)
        res = bt.run(w, P, mode="open", cost_bps=5.0)
        m = bt.metrics(res["net"])
        m["gross_sharpe"] = bt.sharpe(res["gross"])
        m.update(bt.is_oos(res["net"], split="2022-01-01"))
        out[nm] = m
        print(f"{nm:18s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} 16-21={m['IS']:6.2f} 22+={m['OOS']:6.2f}")

json.dump(out, open(os.path.join(ROOT, "results", "exp22_etf_leadlag.json"), "w"), indent=1)
print("saved")
