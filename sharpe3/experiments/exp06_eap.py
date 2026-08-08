"""Exp06: Earnings Announcement Premium (EAP) with PIT-predicted windows.

Each stock's next earnings date is predicted from its OWN past 8-K (2.02)
cadence: expected next = last 2.02 date + median gap (~91d). A stock enters
the "announcement window" [pred-3, pred+3]. Strategy: long stocks in window,
vs short stocks far from window (or vs beta-matched market short).

Fully PIT: prediction uses only filings dated <= signal date.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, open_, member = P["close"], P["open"], P["member"]
r1 = close.pct_change(fill_method=None)
dates = close.index

ek = pd.read_parquet(os.path.join(ROOT, "..", "dca", "research", "data", "sec", "8k_items.parquet"))
ek["date"] = pd.to_datetime(ek["date"])
ek = ek[ek["items"].str.contains("2.02", na=False) & ek.tk.isin(close.columns)]
ek = ek.sort_values("date")

# days-to-predicted-announcement matrix
dtpa = pd.DataFrame(np.nan, index=dates, columns=close.columns, dtype=np.float32)
for tk, g in ek.groupby("tk"):
    ds = g["date"].values
    if len(ds) < 3:
        continue
    col = np.full(len(dates), np.nan, dtype=np.float32)
    # for each trading day, last announcement <= d and median historical gap
    gaps = np.diff(ds).astype("timedelta64[D]").astype(float)
    for i, d in enumerate(ds[:-1]):
        # prediction active from just after announcement i until next one
        med = np.median(gaps[max(0, i - 7):i + 1])
        if not (60 <= med <= 130):
            med = 91.0
        pred = d + np.timedelta64(int(round(med)), "D")
        lo = dates.searchsorted(d + np.timedelta64(1, "D"))
        hi = dates.searchsorted(ds[i + 1] + np.timedelta64(1, "D"))
        rng = np.arange(lo, hi)
        col[rng] = (pred - dates.values[rng]).astype("timedelta64[D]").astype(float)
    dtpa[tk] = col
print("dtpa built; coverage:", float(dtpa.notna().mean().mean()).__round__(3))

in_window = (dtpa >= -1) & (dtpa <= 5)   # pred-5d .. pred+1d
far = (dtpa > 20) | (dtpa < -10)

out = {}
def ev(name, w, mode="open", cost=5.0):
    res = bt.run(w, P, mode=mode, cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    m["yearly"] = bt.yearly_sharpes(res["net"])
    out[name] = m
    print(f"{name:26s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}")

memw = member & r1.notna()
wl = in_window & memw
wl_w = wl.div(wl.sum(axis=1).clip(lower=1), axis=0)
ws = far & memw
ws_w = ws.div(ws.sum(axis=1).clip(lower=1), axis=0)
ev("eap_LS", wl_w - ws_w)
ev("eap_longonly", wl_w)
# how many names in window per day
print("avg names in window:", float(wl.sum(axis=1).mean()).__round__(1))

with open(os.path.join(ROOT, "results", "exp06_eap.json"), "w") as f:
    json.dump(out, f, indent=1)
print("saved")
