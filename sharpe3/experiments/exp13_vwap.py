"""Exp13: close-vs-VWAP (end-of-day order-imbalance proxy) on the 118-stock
intraday-derived daily files (2016+). Signal: (close/vwap - 1), z-scored;
closing above VWAP = buying pressure into the close. Test both continuation
and reversal, 1-5d, next-open execution. Also gap-vs-vwap variants.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDIR = os.path.join(ROOT, "..", "data", "intraday_daily")
frames = {}
for fn in sorted(os.listdir(DDIR)):
    tk = fn[:-4]
    df = pd.read_csv(os.path.join(DDIR, fn), parse_dates=["ts"])
    df["date"] = df["ts"].dt.tz_localize(None).dt.normalize()
    frames[tk] = df.set_index("date")
print("stocks:", len(frames))

fields = {}
for f in ("open", "high", "low", "close", "volume", "vwap"):
    fields[f] = pd.DataFrame({tk: d[f] for tk, d in frames.items()})

close, open_, vwap = fields["close"], fields["open"], fields["vwap"]
high, low = fields["high"], fields["low"]
r1 = close.pct_change(fill_method=None)
vol20 = r1.rolling(20).std()
member = close.notna() & vwap.notna()

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

cv = (close / vwap - 1)
# intraday range position: where close sits in [low, high]
rngpos = ((close - low) / (high - low).replace(0, np.nan) - 0.5)

sigs = {
    "cv_cont": zs(cv / vol20),
    "cv_rev": -zs(cv / vol20),
    "cv5_cont": zs(cv.rolling(5).mean() / vol20),
    "cv5_rev": -zs(cv.rolling(5).mean() / vol20),
    "rngpos_cont": zs(rngpos),
    "rngpos_rev": -zs(rngpos),
    "rngpos5_rev": -zs(rngpos.rolling(5).mean()),
}

panel = {"close": close, "open": open_}
out = {}
for name, sig in sigs.items():
    w = bt.norm_ls(sig, member, 0.15, 0.15, 2.0)
    res = bt.run(w, panel, mode="open", cost_bps=3.0)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m.update(bt.is_oos(res["net"], split="2022-01-01"))
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    out[name] = m
    print(f"{name:14s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS(16-21)={m['IS']:6.2f} OOS(22+)={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}")

json.dump(out, open(os.path.join(ROOT, "results", "exp13_vwap.json"), "w"), indent=1)
print("saved")
