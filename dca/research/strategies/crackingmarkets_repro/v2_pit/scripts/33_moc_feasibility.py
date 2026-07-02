"""How wrong is a signal computed at 3:55pm vs the actual close?

Uses committed 5-minute bars (SPY/QQQ/TLT/GLD/IWM/DIA/XLF, 2016+) to
measure the 3:55->4:00 drift, then asks: what fraction of RSI(2)<10 /
RSI(5)<20 / "down >3%" signals computed on the 3:55 price would FLIP by the
close? That is the honest cost of same-close (MOC) execution.
"""
import os
import numpy as np
import pandas as pd

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from lib import wilder_rsi

ID = "/home/user/bonds/data/intraday_5min"
for tk in ["SPY", "QQQ", "IWM", "TLT", "GLD"]:
    df = pd.read_csv(os.path.join(ID, f"{tk}.csv"), parse_dates=["ts"])
    df["date"] = df["ts"].dt.date
    days = df.groupby("date")
    # bars are labeled by START time; the 15:55 bar's close IS the 16:00
    # close. The price you'd see at ~3:55pm is the close of the 15:50 bar.
    c1555 = days.apply(lambda g: g[g["ts"].dt.time <= pd.Timestamp("15:50").time()]["close"].iloc[-1]
                       if len(g[g["ts"].dt.time <= pd.Timestamp("15:50").time()]) else np.nan,
                       include_groups=False)
    c1600 = days.apply(lambda g: g["close"].iloc[-1], include_groups=False)
    drift = (c1600 / c1555 - 1).dropna()
    close = pd.Series(c1600.values, index=pd.to_datetime(c1600.index))
    # signal flip rates using proxy close = 3:55 price
    prox = pd.Series(c1555.values, index=pd.to_datetime(c1555.index))
    for nm, n, th in [("RSI2<10", 2, 10), ("RSI5<20", 5, 20)]:
        r_true = wilder_rsi(close, n) < th
        hist = close.shift(1)
        prox_series = close.copy()
        # recompute rsi replacing today's close with the 3:55 price, day by day
        # (vectorized approx: use prox for the last diff only)
        d_true = close.diff()
        d_prox = prox - close.shift(1)
        # Wilder rsi with substituted last diff
        ru = d_true.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean().shift(1)
        rd = (-d_true).clip(lower=0).ewm(alpha=1/n, min_periods=n).mean().shift(1)
        a = 1.0 / n
        ru_p = (1 - a) * ru + a * d_prox.clip(lower=0)
        rd_p = (1 - a) * rd + a * (-d_prox).clip(lower=0)
        rsi_p = 100 - 100 / (1 + ru_p / rd_p.replace(0, np.nan))
        s_prox = rsi_p < th
        both = r_true.notna() & rsi_p.notna()
        fire_true = r_true[both]
        fire_prox = s_prox[both]
        flip = (fire_true != fire_prox).mean()
        cond_flip = (fire_true != fire_prox)[fire_true | fire_prox].mean()
        print(f"{tk} {nm}: |drift| median {drift.abs().median()*1e4:.1f}bp "
              f"p90 {drift.abs().quantile(.9)*1e4:.1f}bp | flip(all days) "
              f"{flip*100:.2f}%  flip(signal days) {cond_flip*100:.1f}%")
