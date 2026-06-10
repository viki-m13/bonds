"""More candidates: crypto cross-sectional momentum, calendar effects.

XSCRYPTO — weekly cross-sectional momentum across liquid alts, BTC-hedged option.
TOM      — turn-of-month equity effect (long QQQ/SPY last day + first 3 days).
All: signal lag >= 1d, costs included (crypto 20bp/side, ETF 1bp/side).
"""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRY = ROOT / "data/crypto"
ETF = ROOT / "data/etfs"


def sr(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(len_per_year(r))) if len(r) > 60 and r.std() > 0 else np.nan


def len_per_year(r):
    # crypto trades 365d/yr; equities 252
    if r.index.dayofweek.max() >= 5:
        return 365
    return 252


def stats(r, label):
    r = r.dropna()
    ann = len_per_year(r)
    m = {w: (s.mean() / s.std() * np.sqrt(ann) if len(s) > 30 and s.std() > 0 else np.nan)
         for w, s in [("IS", r.loc[:"2018"]), ("OOS", r.loc["2019":]), ("full", r)]}
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    print(f"  {label:46s} IS={m['IS']:5.2f} OOS={m['OOS']:5.2f} full={m['full']:5.2f} mdd={mdd*100:5.1f}%")
    return r


def xs_crypto():
    print("=" * 90)
    print("XSCRYPTO — weekly cross-sectional momentum, top-3 of liquid alts, 20bp/side")
    print("=" * 90)
    closes = {}
    for f in CRY.glob("*_USD.csv"):
        t = f.stem.replace("_USD", "")
        d = pd.read_csv(f, parse_dates=["Date"], index_col="Date")["Close"]
        d = d[~d.index.duplicated()].sort_index()
        if len(d) > 365 * 2:
            closes[t] = d
    px = pd.concat(closes, axis=1, sort=True)
    print(f"  universe {px.shape[1]} coins")
    r1 = px.pct_change()
    # liquidity proxy: require 1y history; momentum = 28d return skipping last 7d
    mom = px.shift(7).pct_change(21)
    vol = r1.rolling(30).std()
    score = mom / (vol * np.sqrt(28))
    n_avail = score.notna().sum(axis=1)
    rank = score.rank(axis=1, ascending=False)
    w = ((rank <= 3) / 3.0)[n_avail >= 8]
    # weekly rebalance (Mondays), hold 7d, lag 1 day
    monday = pd.Series(w.index.dayofweek == 0, index=w.index)
    w = w.where(monday, np.nan).ffill(limit=7).shift(1).fillna(0)
    # absolute filter: only long coins above their own 50d MA
    above = (px > px.rolling(50).mean()).shift(1).fillna(False)
    w = w * above
    gross = (w * r1).sum(axis=1)
    to = w.diff().abs().sum(axis=1)
    net = gross - to * 0.002
    stats(net, "XS top-3 mom, trend-gated, long-only")
    # vol-targeted 30%:
    rv = net.rolling(30, min_periods=15).std() * np.sqrt(365)
    net_vt = net * (0.30 / rv.clip(lower=0.05)).clip(0.1, 3).shift(1)
    stats(net_vt.dropna(), "  same, vol-targeted 30%")
    # BTC-only TSMOM benchmark for reference
    btc = px["BTC"].pct_change()
    sig = (px["BTC"] > px["BTC"].rolling(100).mean()).shift(1).astype(float)
    stats((sig * btc - sig.diff().abs().fillna(0) * 0.002).dropna(), "BTC 100d trend benchmark")
    return net_vt


def tom():
    print("=" * 90)
    print("TOM — turn-of-month long QQQ (last td + first 3 td), 1bp/side")
    print("=" * 90)
    for t in ["SPY", "QQQ", "IWM"]:
        d = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")["Close"]
        d = d[~d.index.duplicated()].sort_index()
        r = d.pct_change()
        idx = r.index
        mon = idx.to_period("M")
        is_last = pd.Series(mon != np.roll(mon, -1), index=idx)
        is_last.iloc[-1] = True
        pos_in_month = pd.Series(idx, index=idx).groupby(mon).cumcount()
        in_win = (is_last | (pos_in_month <= 2)).astype(float)
        # trade at close before window: shift not needed for calendar (known in advance)
        ret = in_win * r - in_win.diff().abs().fillna(0) * 0.0001
        stats(ret.loc["2010":], f"{t} TOM window")
        # complement: rest of month
        ret2 = (1 - in_win) * r
        stats(ret2.loc["2010":], f"{t} rest-of-month")


if __name__ == "__main__":
    xv = xs_crypto()
    xv.to_csv(ROOT / "phoenix5/results/xscrypto_candidate.csv")
    tom()
