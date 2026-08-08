"""Exp15: what does intraday (5-min) data offer? ETF timing sleeves (SPY, QQQ,
IWM, DIA, GLD, TLT, XLF), 2016-2026. Not stock picking — this measures whether
the *frequency* dimension (absent for single stocks in this repo) is where
higher Sharpe lives. Costs: 1 bp/side (ETF spreads ~0.3-1bp + auction).

Sleeves:
  a) overnight drift: long close->open
  b) intraday momentum: first-30-min return sign -> hold rest of day
  c) last-30-min momentum: day-so-far sign -> hold last 30 min
  d) open-drive reversal: fade big gaps intraday
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import bt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDIR = os.path.join(ROOT, "..", "data", "intraday_5min")

out = {}
for fn in sorted(os.listdir(DDIR)):
    tk = fn[:-4]
    df = pd.read_csv(os.path.join(DDIR, fn), parse_dates=["ts"])
    df["date"] = df["ts"].dt.normalize()
    df["t"] = df["ts"].dt.time
    piv_c = df.pivot_table(index="date", columns="t", values="close")
    piv_o = df.pivot_table(index="date", columns="t", values="open")
    import datetime as dtm
    T = lambda h, m: dtm.time(h, m)
    op = piv_o.get(T(9, 30))
    c10 = piv_c.get(T(9, 55))   # ~first 30 min (9:30-10:00 bar close at 9:55+5m)
    c330 = piv_c.get(T(15, 25))  # 3:30pm
    cl = piv_c.get(T(15, 55))    # last bar close ~ 4:00
    if op is None or cl is None:
        continue
    on = op / cl.shift(1) - 1                 # overnight
    fh = c10 / op - 1                         # first 30 min
    rest = cl / c10 - 1                       # 10:00 -> close
    dayso = c330 / op - 1                     # open -> 3:30
    last30 = cl / c330 - 1                    # 3:30 -> close
    c = 1e-4
    sleeves = {
        "overnight_long": on - 2 * c,
        "intraday_mom": np.sign(fh) * rest - 2 * c,
        "last30_mom": np.sign(dayso) * last30 - 2 * c,
        "gap_fade": -np.sign(on) * (cl / op - 1) - 2 * c,
    }
    for nm, ret in sleeves.items():
        ret = ret.dropna()
        if len(ret) < 500:
            continue
        m = bt.metrics(ret)
        m.update(bt.is_oos(ret, split="2022-01-01"))
        out[f"{tk}_{nm}"] = m

for k, v in sorted(out.items(), key=lambda kv: -kv[1]["sharpe"]):
    print(f"{k:24s} SR={v['sharpe']:6.2f} IS={v['IS']:6.2f} OOS22+={v['OOS']:6.2f} dd={v['maxdd']:.2f}")
json.dump(out, open(os.path.join(ROOT, "results", "exp15_intraday_etf.json"), "w"), indent=1)
