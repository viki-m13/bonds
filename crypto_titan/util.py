"""CRYPTO-TITAN utilities — standalone crypto strategy, independent of APEX/PHOENIX.

Design philosophy (distinct from crypto_apex):
  * Weekly rebalance (not daily) — dramatically lowers TC
  * Volatility-managed exposure (Moreira-Muir) as the primary Sharpe booster
  * Risk-parity across trend-filtered survivors (not top-K inverse-vol)
  * Breadth-based master kill-switch (not 200MA/DD-based)
  * Survivorship bias handled by including dead coins (LUNA1/USTC/FTT/MATIC/UNI)
    and masking each coin ineligible from its last-valid date onward.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data/crypto")
ETF = Path("/home/user/bonds/data/etfs")
FRED = Path("/home/user/bonds/data/fred")
OUT = Path("/home/user/bonds/data/crypto_titan")
OUT.mkdir(parents=True, exist_ok=True)

DPY = 365  # crypto trades 7 days/week

# Full universe — survivors and dead together (survivorship-aware by design).
SURVIVORS = ["BTC", "ETH", "SOL", "ADA", "DOGE", "LTC", "BCH", "XRP", "LINK",
             "DOT", "AVAX", "ATOM", "XLM", "TRX", "ALGO"]
DEAD = ["LUNA1", "USTC", "FTT", "MATIC", "UNI"]
ALL_COINS = SURVIVORS + DEAD


def load_prices(coins=None) -> pd.DataFrame:
    if coins is None:
        coins = ALL_COINS
    frames = []
    for c in coins:
        fp = DATA / f"{c}_USD.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
        frames.append(df["Close"].astype(float).rename(c))
    cp = pd.concat(frames, axis=1).sort_index()
    cp.index = pd.to_datetime(cp.index)
    cp = cp.loc[~cp.index.duplicated(keep="last")]
    return cp


def load_macro(idx: pd.DatetimeIndex) -> dict:
    """Load SPY / VIX / DXY aligned to the crypto index (ffill over weekends)."""
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
        "uup": _etf("UUP"),
        "vix": _fred("VIXCLS"),
        "ust10": _fred("DGS10"),
    }


# Historical BTC halvings (deterministic, known in advance).
HALVING_DATES = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19"]


def safe_returns(cp: pd.DataFrame, cap: float = 0.25) -> pd.DataFrame:
    """Daily returns clipped to ±cap. A realistic stop-loss would fire near
    these magnitudes; clipping also prevents dead-cat bounces in delisted
    coins (LUNA at $0.00006 → $0.0002) from polluting the backtest."""
    return cp.pct_change().clip(lower=-cap, upper=cap)


def eligibility(cp: pd.DataFrame, min_history: int = 180,
                 catastrophe_dd: float = -0.35,
                 dd_window: int = 60) -> pd.DataFrame:
    """True when coin:
      (1) has ≥`min_history` days of prior data,
      (2) has not yet delisted (last_valid_index in the future), AND
      (3) is not in a catastrophic drawdown (>35% off `dd_window`-day high).

    (3) is the critical survivorship-aware stop-loss — it pulls LUNA, FTT,
    and any coin mid-collapse out of the basket BEFORE the compound damage.
    Applied with shift(1) at the signal layer to avoid look-ahead.
    """
    has = cp.notna()
    age = has.cumsum()
    mask = (age >= min_history).astype(float)
    for c in cp.columns:
        lv = cp[c].last_valid_index()
        if lv is not None:
            mask.loc[mask.index > lv, c] = 0.0

    hwm = cp.rolling(dd_window, min_periods=20).max()
    dd = cp / hwm - 1
    not_crashing = (dd > catastrophe_dd).astype(float)
    mask = mask * not_crashing.fillna(0.0)
    return mask.astype(float)


def metrics(r: pd.Series, rf: float = 0.0) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"sharpe": 0, "sortino": 0, "cagr": 0, "vol": 0, "mdd": 0,
                "calmar": 0, "hit": 0, "nav": 1.0, "n": 0}
    mu = r.mean() * DPY
    sd = r.std() * np.sqrt(DPY)
    sharpe = (mu - rf) / sd if sd > 0 else 0
    nav = (1 + r).cumprod()
    years = len(r) / DPY
    cagr = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    hwm = nav.cummax()
    mdd = (nav / hwm - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else 0
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


def summarize(r: pd.Series, label: str = "") -> None:
    m = metrics(r)
    print(f"  {label:28s} SR={m['sharpe']:>5.2f}  CAGR={m['cagr']*100:>7.1f}%  "
          f"Vol={m['vol']*100:>5.1f}%  MDD={m['mdd']*100:>6.1f}%  "
          f"Calmar={m['calmar']:>4.2f}  NAV={m['nav']:>8.2f}")


def weights_to_ret(W: pd.DataFrame, cp: pd.DataFrame,
                    tc_bps: float = 25.0, ret_cap: float = 0.25) -> pd.Series:
    """Weight DataFrame → daily net returns with TC drag.

    Weights are assumed already shifted (i.e., W[t] is the position held going
    into day t). Caller should pass shifted weights.
    """
    rets = safe_returns(cp, cap=ret_cap).reindex_like(W).fillna(0.0)
    gross = (W * rets).sum(axis=1)
    dw = W.diff().abs().fillna(W.abs())
    drag = dw.sum(axis=1) * tc_bps / 1e4
    return gross - drag


def to_weekly_weights(W_daily: pd.DataFrame, rebal_day: int = 2) -> pd.DataFrame:
    """Rebalance weekly on `rebal_day` (0=Mon ... 6=Sun). Between rebalances,
    hold the most recent rebalance-day weights (piecewise constant).

    Reduces turnover ~5x vs daily, a major Sharpe lever.
    """
    is_rebal = pd.Series(W_daily.index.dayofweek == rebal_day, index=W_daily.index)
    Wr = W_daily.where(is_rebal, other=np.nan).ffill()
    return Wr.fillna(0.0)
