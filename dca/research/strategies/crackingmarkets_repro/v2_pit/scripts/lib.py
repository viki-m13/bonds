"""Shared data / metrics / cost layer for the v2 PIT-honest study.

Everything runs off COMMITTED repo data (no API key needed):
  - data/pit/n100_panel_*.parquet       PIT NDX-100, adjusted OHLCV + member mask
  - dca/research/data/tiingo/prices/    survivorship-clean adjClose+volume, 24k names
  - data/etfs*/                         adjusted OHLCV ETF csvs
  - data/fred/                          DGS3MO (risk-free), FEDFUNDS (financing), VIXCLS

Cost model (per fill, in bps of trade notional unless stated):
  commission      $0.005/share (IBKR tiered)  -> bps = 0.05/price*100
  auction fills   (MOO/MOC) no spread paid; +1 bp auction noise + impact
  marketable      +half-spread (ADV tier) + impact
  passive limit   no spread (we provide liquidity); optional fill-through
                  requirement handles optimistic-fill bias instead
  impact          25 bps * sqrt(trade_notional / dollar_ADV)   (sqrt law;
                  ~0.8 bp at 0.1% ADV, 2.5 bp at 1% ADV)
  financing       leverage borrows at FEDFUNDS + 150 bp, daily accrual;
                  cash earns nothing (conservative)
"""
import os
import numpy as np
import pandas as pd

ROOT = "/home/user/bonds"
PIT = os.path.join(ROOT, "data", "pit")
TII = os.path.join(ROOT, "dca", "research", "data", "tiingo")
OUT = os.path.join(ROOT, "dca", "research", "strategies", "crackingmarkets_repro",
                   "v2_pit", "out")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- data loaders
def load_n100():
    """PIT NDX-100 panels: open/high/low/close/volume (adjusted) + member mask."""
    p = {f: pd.read_parquet(os.path.join(PIT, f"n100_panel_{f}.parquet"))
         for f in ["open", "high", "low", "close", "volume", "member"]}
    return p


_TII_CACHE = {}


def load_tiingo(min_days=250):
    """Survivorship-clean adjClose + volume wide panels (24k names, 1990-2026).

    Grafts AAPL/ADP/BKNG (missing from the tiingo pull) from the committed
    Yahoo-adjusted csvs in data/stocks so megacaps aren't silently absent.
    """
    if "ac" in _TII_CACHE:
        return _TII_CACHE["ac"], _TII_CACHE["vol"]
    pr = os.path.join(TII, "prices")
    ac = pd.concat([pd.read_parquet(os.path.join(pr, f))
                    for f in sorted(os.listdir(pr)) if f.startswith("ac_")],
                   axis=1, sort=True)
    vol = pd.concat([pd.read_parquet(os.path.join(pr, f))
                     for f in sorted(os.listdir(pr)) if f.startswith("vol_")],
                    axis=1, sort=True)
    ac = ac.loc[:, ~ac.columns.duplicated()]
    vol = vol.loc[:, ~vol.columns.duplicated()]
    ac.index = pd.to_datetime(ac.index)
    vol.index = pd.to_datetime(vol.index)
    for t in ["AAPL", "ADP", "BKNG"]:
        if t not in ac.columns:
            f = os.path.join(ROOT, "data", "stocks", f"{t}.csv")
            if os.path.exists(f):
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                df = df[~df.index.duplicated()]
                ac[t] = df["Close"].reindex(ac.index).astype("float32")
                vol[t] = df["Volume"].reindex(ac.index).astype("float32")
    _TII_CACHE["ac"], _TII_CACHE["vol"] = ac, vol
    return ac, vol


def load_etf(ticker, extended=True):
    """Adjusted OHLCV for an ETF."""
    d = "etfs_extended" if extended else "etfs"
    f = os.path.join(ROOT, "data", d, f"{ticker}.csv")
    if not os.path.exists(f):
        f = os.path.join(ROOT, "data", "etfs", f"{ticker}.csv")
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated()]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def load_fred(series):
    f = os.path.join(ROOT, "data", "fred", f"{series}.csv")
    s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def riskfree_daily(index):
    """Daily risk-free return series (DGS3MO, act/252) aligned to index."""
    rf = load_fred("DGS3MO").reindex(index).ffill().fillna(0) / 100.0
    return rf / 252.0


def financing_daily(index, spread_bps=150):
    """Daily borrow-rate for margin debit: FEDFUNDS + spread."""
    ff = load_fred("FEDFUNDS").reindex(index).ffill().fillna(0) / 100.0
    return (ff + spread_bps / 1e4) / 252.0


# ------------------------------------------------------------------- indicators
def wilder_rsi(close, n):
    d = close.diff()
    ru = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    rd = (-d).clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))


def atr_simple(high, low, close, n=5):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()]) \
           .groupby(level=0).max()
    return tr.rolling(n, min_periods=n).mean()


# ------------------------------------------------------------------------ costs
def half_spread_bps(dollar_adv):
    """Half-spread estimate by liquidity tier (one-way, bps)."""
    if not np.isfinite(dollar_adv) or dollar_adv <= 0:
        return 20.0
    if dollar_adv > 500e6:
        return 1.5
    if dollar_adv > 100e6:
        return 2.5
    if dollar_adv > 25e6:
        return 5.0
    if dollar_adv > 5e6:
        return 10.0
    return 20.0


def impact_bps(trade_notional, dollar_adv):
    if not np.isfinite(dollar_adv) or dollar_adv <= 0:
        return 25.0
    return 25.0 * np.sqrt(min(trade_notional / dollar_adv, 0.25))


def fill_cost_bps(style, price, trade_notional, dollar_adv,
                  commission_per_share=0.005):
    """Total one-way cost in bps of notional for a fill of the given style.

    style: 'auction' (MOO/MOC), 'marketable' (crossing the spread),
           'passive' (resting limit filled).
    """
    comm = commission_per_share / price * 1e4 if price > 0 else 5.0
    imp = impact_bps(trade_notional, dollar_adv)
    if style == "auction":
        return comm + imp + 1.0
    if style == "marketable":
        return comm + imp + half_spread_bps(dollar_adv)
    if style == "passive":
        return comm  # liquidity provider: no spread, no impact
    raise ValueError(style)


# ----------------------------------------------------------------------- metrics
def stats(returns, rf=None, label="", periods=252):
    """Daily-basis stats. returns = daily simple returns (net)."""
    r = pd.Series(returns).dropna()
    if len(r) < 10:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(periods)
    sharpe = r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan
    if rf is not None:
        ex = r - pd.Series(rf).reindex(r.index).fillna(0)
        sharpe_ex = ex.mean() / ex.std() * np.sqrt(periods)
    else:
        sharpe_ex = np.nan
    dd = (eq / eq.cummax() - 1).min()
    mr = (1 + r).resample("ME").prod() - 1 if isinstance(r.index, pd.DatetimeIndex) else None
    sharpe_m = mr.mean() / mr.std() * np.sqrt(12) if mr is not None and mr.std() > 0 else np.nan
    return {"label": label, "CAGR": cagr, "vol": vol, "Sharpe_d": sharpe,
            "Sharpe_ex": sharpe_ex, "Sharpe_m": sharpe_m, "maxDD": dd,
            "years": yrs}


def fmt(st):
    if not st:
        return "insufficient data"
    return (f"{st['label']:42s} CAGR {st['CAGR']*100:6.1f}%  vol {st['vol']*100:5.1f}%  "
            f"Sh(d) {st['Sharpe_d']:5.2f}  Sh(ex) {st['Sharpe_ex']:5.2f}  "
            f"Sh(m) {st['Sharpe_m']:5.2f}  maxDD {st['maxDD']*100:6.1f}%")
