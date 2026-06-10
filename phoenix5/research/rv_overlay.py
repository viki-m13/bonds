"""Different method test: intraday realized-vol (RV) accelerated overlay.

PHOENIX's overlay estimates portfolio vol with a 60d trailing std — it takes
weeks to register a regime shift. Intraday 5-min bars give a vol estimate that
reacts in days. Test: speed up the vol-target denominator multiplicatively:

    sigma_hat = sigma_60d(blend) * clip( RV_fast / RV_slow , 0.6, 2.5 )

where RV_fast = 5d mean of market intraday RV (SPY/QQQ/TLT average),
RV_slow = 60d mean of the same. Everything lagged 1 day. The blend's own 60d
vol anchors the level; the intraday ratio supplies speed. Data 2016+ only, so
IS for this test = 2016-2018, OOS = 2019+.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(ROOT / "phoenix5"))
import phoenix5x as X  # noqa: E402


def intraday_rv(t):
    df = pd.read_csv(ROOT / f"data/intraday_5min/{t}.csv", parse_dates=["ts"])
    df["date"] = df["ts"].dt.normalize()
    r = df.groupby("date")["close"].apply(lambda s: (np.log(s / s.shift(1)) ** 2).sum())
    return np.sqrt(r * 252)  # annualized daily RV


def main():
    df = X.load_sleeves()
    raw = (df @ pd.Series(X.W_PROD)).loc["2015-06-01":]
    bil = X.px("BIL").pct_change().reindex(raw.index).fillna(0)

    rv_mkt = pd.concat([intraday_rv(t) for t in ["SPY", "QQQ", "TLT"]], axis=1).mean(axis=1)
    rv_mkt = rv_mkt.reindex(raw.index).ffill()
    ratio = (rv_mkt.rolling(5).mean() / rv_mkt.rolling(60).mean()).clip(0.6, 2.5)

    def run(speed_on, gate_lvl=0.5):
        sig60 = raw.rolling(60).std() * np.sqrt(252)
        sigma = sig60 * ratio if speed_on else sig60
        vol_mult = (0.15 / sigma).clip(0.25, 1.0).shift(1).fillna(1.0)
        scaled = raw * vol_mult
        cum = (1 + scaled).cumprod()
        hwm = cum.rolling(252, min_periods=30).max()
        dd_mult = (1.0 + (cum / hwm - 1) / -0.10).clip(0, 1).shift(1).fillna(1.0)
        sv = scaled.rolling(60).std()
        thr = sv.rolling(252, min_periods=60).quantile(0.99)
        ok = (sv <= thr).shift(1).fillna(True).astype(float)
        gate = ok + (1 - ok) * gate_lvl
        total = (vol_mult * dd_mult * gate).ewm(span=3).mean().clip(0, 1.0)
        tc = total.diff().abs().fillna(0) * (10 / 1e4)
        return (raw * total + (1 - total).clip(lower=0) * bil - tc).loc["2016-06-01":]

    base = run(False)
    fast = run(True)
    print("2016-06 onward (intraday data era), idle->BIL, 3d smooth, gate 0.5:")
    for name, r in [("baseline 60d vol", base), ("intraday-RV accelerated", fast)]:
        o = X.metrics(r.loc["2019":])
        i = X.metrics(r.loc[:"2018"])
        print(f"  {name:26s} IS16-18: SR={i['sr']:4.2f} MDD={i['mdd']*100:5.1f}% | "
              f"OOS: SR={o['sr']:4.2f} CAGR={o['cagr']*100:5.1f}% vol={o['vol']*100:4.1f}% "
              f"MDD={o['mdd']*100:6.1f}%")

    # where does it differ? top drawdown episodes
    for name, r in [("base", base), ("fast", fast)]:
        c = (1 + r.loc["2019":]).cumprod()
        dd = c / c.cummax() - 1
        print(f"  {name} worst dd date: {dd.idxmin().date()} {dd.min()*100:.1f}%  "
              f"2020Q1: {((1+r.loc['2020-02-15':'2020-04-15']).prod()-1)*100:+.1f}%  "
              f"2022: {((1+r.loc['2022']).prod()-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
