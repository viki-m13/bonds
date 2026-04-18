"""Step 1: Explore return/Sharpe landscape of leveraged & momentum ETFs
with a simple regime filter (SPY > 200dma).

Goal: find the raw ingredients that could plausibly give Sharpe 3 with 20%+ return.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data")
ETF = DATA / "etfs"


def load(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def gate_spy(dates):
    spy = load("SPY").reindex(dates).ffill()
    ma = spy.rolling(200).mean()
    return (spy > ma).shift(1).fillna(False).astype(float)


def metrics(r):
    r = r.loc[r.ne(0).idxmax():] if (r != 0).any() else r
    if r.std() == 0 or len(r) == 0: return None
    ar, av = r.mean() * 252, r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return (ar/av if av > 0 else 0, ar, av, dd, len(r)/252)


tickers = ["TQQQ", "UPRO", "TMF", "SOXL", "TECL", "FAS", "SSO", "QLD", "UGL",
           "SVXY", "BITO", "TLT", "QQQ", "SPY", "GLD", "DBMF", "KMLM", "CTA",
           "EWZ", "FXI", "INDA", "EEM", "XLK", "XLE", "XLF"]

print(f"{'Ticker':<7}{'RAW SR':>8}{'Ret':>7}{'Vol':>7}{'MDD':>8}   "
      f"{'GATED SR':>9}{'Ret':>7}{'Vol':>7}{'MDD':>8}{'Years':>7}")
for t in tickers:
    p = load(t)
    if p is None: continue
    r = p.pct_change().fillna(0)
    r = r.loc["2014-01-01":]
    raw = metrics(r)
    if raw is None: continue
    # Gated version: only hold when SPY > 200dma
    g = gate_spy(r.index)
    rg = r * g
    gated = metrics(rg)
    if gated is None: continue
    print(f"{t:<7}{raw[0]:8.2f}{raw[1]*100:6.1f}%{raw[2]*100:6.1f}%{raw[3]*100:7.1f}%   "
          f"{gated[0]:9.2f}{gated[1]*100:6.1f}%{gated[2]*100:6.1f}%{gated[3]*100:7.1f}%{gated[4]:7.1f}")
