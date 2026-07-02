"""Intraday momentum (Gao-Han-Li-Zhou 2018) on the committed 5-min ETF bars.

Effect: the first half-hour return (measured from the previous close, so it
includes the overnight gap) predicts the last half-hour return. Also
documented: the second-to-last half-hour (15:00-15:30) as a second predictor.

Implementation (PIT: everything known at 15:30):
  r_fh  = close(10:00 bar i.e. 09:55 bar) / prev_session_close - 1
  r_pen = close(15:30, i.e. 15:25 bar) / close(15:00, i.e. 14:55 bar) - 1
  signal at 15:30: sign(r_fh) [variant: agree = trade only when both agree]
  position: +/-1x notional, enter at the 15:30 bar OPEN, exit at session
  close (15:55 bar close). One round trip per day.

Costs per side: best 0.25bp (QQQ/SPY) / 0.5bp others; conservative 1bp all.
Leverage variants use intraday margin (free) — still one trade/day.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import stats, fmt, riskfree_daily, OUT

ID = "/home/user/bonds/data/intraday_5min"
ETFS = ["QQQ", "SPY", "IWM", "DIA", "XLF", "GLD", "TLT"]


def build_day_table(tk):
    df = pd.read_csv(os.path.join(ID, f"{tk}.csv"), parse_dates=["ts"])
    df["date"] = df["ts"].dt.normalize()
    df["t"] = df["ts"].dt.time
    g = df.set_index(["date", "t"])
    def at(tt, col):
        return g[col].xs(pd.Timestamp(tt).time(), level="t")
    tab = pd.DataFrame({
        "c0955": at("09:55", "close"),    # first half-hour end
        "c1455": at("14:55", "close"),    # 15:00
        "c1525": at("15:25", "close"),    # 15:30
        "o1530": at("15:30", "open"),     # entry price
        "sess_close": df.groupby("date")["close"].last(),
        "prev_close": df.groupby("date")["close"].last().shift(1),
    })
    tab["r_fh"] = tab["c0955"] / tab["prev_close"] - 1
    tab["r_pen"] = tab["c1525"] / tab["c1455"] - 1
    tab["r_last"] = tab["sess_close"] / tab["o1530"] - 1
    return tab.dropna(subset=["r_fh", "r_pen", "r_last", "o1530"])


def imom(tab, sig="fh", cost_bps=0.25, lev=1.0):
    if sig == "fh":
        d = np.sign(tab["r_fh"])
    elif sig == "pen":
        d = np.sign(tab["r_pen"])
    elif sig == "agree":
        s1, s2 = np.sign(tab["r_fh"]), np.sign(tab["r_pen"])
        d = np.where(s1 == s2, s1, 0.0)
    elif sig == "either":                     # pen overrides only if fh flat
        d = np.sign(tab["r_pen"] + tab["r_fh"])
    r = pd.Series(d, index=tab.index) * tab["r_last"]
    traded = pd.Series(d, index=tab.index).abs() > 0
    net = (r - traded * 2 * cost_bps / 1e4) * lev
    return net, traded


if __name__ == "__main__":
    t0 = time.time()
    tabs = {tk: build_day_table(tk) for tk in ETFS}
    print(f"tables built t={time.time()-t0:.0f}s\n")
    print("=== intraday momentum, last half-hour, 1x, cost 0.25-0.5bp/side ===")
    keep = {}
    for tk in ETFS:
        c = 0.25 if tk in ("QQQ", "SPY") else 0.5
        for sig in ["fh", "pen", "agree"]:
            r, traded = imom(tabs[tk], sig, c)
            rf = riskfree_daily(r.index)
            st = stats(r, rf, f"{tk} {sig}")
            if sig == "agree":
                keep[tk] = r
            print(fmt(st) + f"  traded {traded.mean()*100:.0f}%")
        print()
    print("=== QQQ 'agree' at cost levels & leverage ===")
    for c, lev in [(0.25, 1), (1.0, 1), (2.0, 1), (0.25, 4), (1.0, 4)]:
        r, _ = imom(tabs["QQQ"], "agree", c, lev)
        rf = riskfree_daily(r.index)
        print(fmt(stats(r, rf, f"QQQ agree {c}bp {lev}x")))
    print()
    # cross-ETF equal-weight ensemble of 'agree' signals
    ens = pd.DataFrame(keep).fillna(0)
    print("=== cross-ETF 'agree' ensembles (1x total notional, split) ===")
    for cols, nm in [(ETFS, "all-7 EW"), (["QQQ", "SPY", "IWM", "DIA"], "eq-4 EW"),
                     (["QQQ", "IWM", "GLD", "TLT"], "div-4 EW")]:
        r = ens[cols].mean(axis=1)
        rf = riskfree_daily(r.index)
        print(fmt(stats(r, rf, nm)))
        corr = ens[cols].corr()
        print(f"  avg pairwise corr {corr.values[np.triu_indices(len(cols),1)].mean():.2f}")
    ens.to_parquet(os.path.join(OUT, "sleeveH_imom.parquet"))
    print(f"\nsaved -> out/sleeveH_imom.parquet  t={time.time()-t0:.0f}s")
