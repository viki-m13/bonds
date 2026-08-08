"""Exp03: 8-K event strategies on the S&P 500 PIT panel.

- PEAD: earnings (item 2.02) reaction close(e-1)->close(e+1), signal live at
  close(e+1), held H days (LS by reaction z).
- News/no-news conditioned reversal: reversal only on names with no 8-K in
  the past 5 days; news-momentum on names with news.
Timing is conservative: a filing dated e is only used from close(e+1).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, open_, volp, member = P["close"], P["open"], P["volume"], P["member"]
r1 = close.pct_change(fill_method=None)
vol20 = r1.rolling(20).std()

ek = pd.read_parquet(os.path.join(ROOT, "..", "dca", "research", "data", "sec", "8k_items.parquet"))
ek["date"] = pd.to_datetime(ek["date"])
ek = ek[ek.tk.isin(close.columns)]

def event_matrix(sub):
    m = pd.DataFrame(False, index=close.index, columns=close.columns)
    g = sub.groupby("tk")["date"]
    for tk, dates in g:
        # snap each filing date to the next trading day if non-trading
        idx = close.index.searchsorted(dates.values)
        idx = idx[idx < len(close.index)]
        m.iloc[idx, m.columns.get_loc(tk)] = True
    return m

earn = event_matrix(ek[ek["items"].str.contains("2.02", na=False)])
anyk = event_matrix(ek)
print("earnings events on panel:", int(earn.values.sum()))

# reaction: close(e-1) -> close(e+1); known at close(e+1)
reac = (close.shift(-1) / close.shift(1) - 1)  # centered at e; NOT causal at e
# make causal: value at row e+1 = reaction of event at e
reac_causal = reac.shift(1).where(earn.shift(1).fillna(False))
reac_z = reac_causal.div(vol20 * np.sqrt(2))

def hold(sig, H):
    """Extend an event-day signal forward H days (decayed equal weight)."""
    return sig.rolling(H, min_periods=1).mean()

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

out = {}
def ev(name, w, cost=5.0):
    res = bt.run(w, P, mode="open", cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    out[name] = m
    print(f"{name:28s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}")

for H in (3, 5, 10):
    sig = hold(reac_z, H)
    # LS on event names only: top/bottom by reaction among names with signal
    w = bt.norm_ls(sig, member & sig.notna(), 0.2, 0.2, 2.0)
    ev(f"pead_H{H}", w)

# news/no-news reversal
news5 = anyk.rolling(5, min_periods=1).sum() > 0
intraday = close / open_ - 1
rev5 = -zs(intraday.rolling(5).sum())
ev("rev5_nonews", bt.norm_ls(rev5.where(~news5), member, 0.1, 0.1, 2.0))
ev("rev5_newsonly", bt.norm_ls(rev5.where(news5), member, 0.1, 0.1, 2.0))
newsmom = zs(r1.rolling(2).sum().where(anyk.rolling(2, min_periods=1).sum() > 0))
ev("newsmom_2d", bt.norm_ls(newsmom, member, 0.2, 0.2, 2.0))

with open(os.path.join(ROOT, "results", "exp03_events.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
