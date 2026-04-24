"""Shared utilities for CRYPTO-APEX strategy.

Applies APEX methodology to a broad crypto universe (20 coins, 4 dead/delisted).
Key difference vs APEX: no LETFs, no synthetic pre-inception, higher target vol
(crypto is 3-5x more volatile than equity). Survivorship bias explicitly handled.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data/crypto")
ETF = Path("/home/user/bonds/data/etfs")
FRED = Path("/home/user/bonds/data/fred")
OUT = Path("/home/user/bonds/data/crypto_apex")
OUT.mkdir(parents=True, exist_ok=True)

DPY = 365  # crypto trades 7 days/week

# Coin universe with survivorship metadata
SURVIVORS = ["BTC", "ETH", "SOL", "ADA", "DOGE", "LTC", "BCH", "XRP", "LINK",
             "DOT", "AVAX", "ATOM", "XLM", "TRX", "ALGO"]
DEAD = ["LUNA1", "USTC", "FTT", "MATIC", "UNI"]
ALL_COINS = SURVIVORS + DEAD


def load_prices(coins=None):
    if coins is None:
        coins = ALL_COINS
    frames = []
    for c in coins:
        fp = DATA / f"{c}_USD.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
        frames.append(df["Close"].astype(float).rename(c))
    cp = pd.concat(frames, axis=1)
    # Daily freq (crypto is 7d/week)
    cp = cp.sort_index()
    cp.index = pd.to_datetime(cp.index)
    cp = cp.loc[~cp.index.duplicated(keep="last")]
    return cp


def load_macro(idx):
    """Load SPY + VIX for macro gates."""
    def _etf(t):
        fp = ETF / f"{t}.csv"
        if not fp.exists():
            return pd.Series(np.nan, index=idx)
        df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
        return df["Close"].astype(float).reindex(idx).ffill()

    def _fred(n):
        fp = FRED / f"{n}.csv"
        if not fp.exists():
            return pd.Series(np.nan, index=idx)
        df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
        return df[df.columns[0]].astype(float).reindex(idx).ffill()

    return {
        "spy": _etf("SPY"),
        "vix": _fred("VIXCLS"),
        "dxy": _etf("UUP"),  # DXY proxy
    }


def metrics(r: pd.Series, rf: float = 0.0) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"sharpe": 0, "cagr": 0, "vol": 0, "mdd": 0, "calmar": 0, "hit": 0, "nav": 1.0, "n": 0}
    mu = r.mean() * DPY
    sd = r.std() * np.sqrt(DPY)
    sharpe = (mu - rf) / sd if sd > 0 else 0
    nav = (1 + r).cumprod()
    years = len(r) / DPY
    cagr = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    hwm = nav.cummax()
    mdd = (nav / hwm - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else 0
    # Sortino
    dn = r[r < 0]
    dsd = dn.std() * np.sqrt(DPY) if len(dn) > 0 else 0
    sortino = (mu - rf) / dsd if dsd > 0 else 0
    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 4),
        "vol": round(sd, 4),
        "mdd": round(mdd, 4),
        "calmar": round(calmar, 4),
        "hit": round((r > 0).mean(), 4),
        "nav": round(float(nav.iloc[-1]), 3),
        "n": len(r),
    }


def regime_slice(r: pd.Series, start: str, end: str) -> pd.Series:
    return r.loc[start:end]


def summarize(r, label=""):
    m = metrics(r)
    print(f"  {label:24s} SR={m['sharpe']:>5.2f}  CAGR={m['cagr']*100:>6.1f}%  "
          f"Vol={m['vol']*100:>5.1f}%  MDD={m['mdd']*100:>6.1f}%  NAV={m['nav']:.2f}")


def safe_returns(cp: pd.DataFrame, cap: float = 0.30) -> pd.DataFrame:
    """Daily returns capped at ±`cap` (realistic — stops would fire at ~30% intraday loss).

    Also clamps extreme positive prints to the same cap (prevents LUNA's
    dead-cat bounces from $0.00006 → $0.0002 from polluting the backtest).
    """
    r = cp.pct_change()
    return r.clip(lower=-cap, upper=cap)


def _weights_to_ret(W: pd.DataFrame, cp: pd.DataFrame, tc_bps: float = 30.0,
                     ret_cap: float = 0.30) -> pd.Series:
    """Convert weight DataFrame to daily returns with TC drag.

    tc_bps: 30bps round-trip (crypto is higher-cost than LETFs).
    ret_cap: cap single-day returns at ±30% (stops-would-fire assumption).
    """
    rets = safe_returns(cp, cap=ret_cap).reindex_like(W).fillna(0.0)
    gross = (W.shift(1).fillna(0.0) * rets).sum(axis=1)
    dw = W.diff().abs().fillna(W.abs())
    drag = dw.sum(axis=1).shift(1).fillna(0.0) * tc_bps / 1e4
    return gross - drag
