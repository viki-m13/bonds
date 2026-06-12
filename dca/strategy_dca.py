"""SUMMIT — biweekly DCA stock-selection strategy (final spec).

Identity: accumulate the index's leaders.
  * RISK-ON  (SPY >= 200dma AND breadth healthy): buy the top-k names by
    multi-horizon momentum with a strong mega-cap (dollar-volume) tilt —
    "the largest stocks that are also leading".
  * RISK-OFF (SPY < 200dma OR <40% of members above their 200dma): keep
    buying every period, but switch to "quality rebounders with size" —
    large names whose long-term uptrend is intact (above 400dma or positive
    24-month return) trading 30-60% below their all-time high, deeper
    discount preferred.
  * Never sell. Every contribution is a permanent lot. No recovery triggers,
    no panic exits (both tested and rejected — see research/results_bear.md).

Causality: a score row at date d uses data through the close of d only;
execution at the next session's open. All inputs (stock panel, SPY, breadth)
are reindexed to the panel's own dates, so hard-truncating the panel
truncates every input — verified by audit.audit_builder.

Cadence: every 10 trading days. k=2 primary, k=3 conservative. 5 bps/trade.
"""
import numpy as np
import pandas as pd

import data as data_mod

BULL_HORIZONS = (63, 126, 189, 252)   # formation windows, each skipping 21d
SKIP = 21                             # skip most recent month (reversal zone)
SIZE_WIN = 63                         # dollar-volume average window
W_SIZE_BULL = 5.0
W_SIZE_BEAR = 1.0
DD_LO, DD_HI = -0.60, -0.30           # rebound discount band vs ATH
BREADTH_FLOOR = 0.40


def _xrank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def _spy_close(index: pd.DatetimeIndex) -> pd.Series:
    spy = data_mod.load_benchmark("SPY")["Close"]
    return spy.reindex(index).ffill()


def risk_off(P: dict) -> pd.Series:
    """True on risk-off days. Causal: 200d rolling stats, reindexed to the
    panel index (truncating the panel truncates these too)."""
    close, member = P["close"], P["member"]
    idx = close.index
    spy = _spy_close(idx)
    spy_bear = spy < spy.rolling(200).mean()
    above = close > close.rolling(200).mean()
    memb = member & close.notna()
    breadth = (above & memb).sum(axis=1) / memb.sum(axis=1).replace(0, np.nan)
    breadth_weak = breadth.rolling(10).mean() < BREADTH_FLOOR
    return (spy_bear | breadth_weak).fillna(False)


def bull_scores(P: dict) -> pd.DataFrame:
    close, vol = P["close"], P["volume"]
    mom = sum(_xrank(close.shift(SKIP).pct_change(h - SKIP, fill_method=None))
              for h in BULL_HORIZONS)
    size = _xrank((close * vol).rolling(SIZE_WIN).mean())
    return _xrank(mom) + W_SIZE_BULL * size


def bear_scores(P: dict) -> pd.DataFrame:
    close, vol = P["close"], P["volume"]
    ath = close.cummax()
    dd = close / ath - 1
    above400 = close > close.rolling(400).mean()
    pos24m = close.pct_change(504, fill_method=None) > 0
    quality = above400 | pos24m
    in_band = (dd >= DD_LO) & (dd <= DD_HI)
    discount = _xrank(-dd.where(quality & in_band))      # deeper = higher
    size = _xrank((close * vol).rolling(SIZE_WIN).mean())
    return discount + W_SIZE_BEAR * size.where(discount.notna())


def build_scores(P: dict) -> pd.DataFrame:
    """The strategy: bull scores on risk-on days, bear scores on risk-off."""
    bull = bull_scores(P)
    bear = bear_scores(P)
    off = risk_off(P).to_numpy()[:, None]
    out = np.where(off, bear.to_numpy(float), bull.to_numpy(float))
    return pd.DataFrame(out, index=bull.index, columns=bull.columns)


def current_picks(k: int = 2) -> pd.DataFrame:
    """Live helper: today's signal (to execute at the next open)."""
    P = data_mod.build_panel()
    s = build_scores(P)
    member = P["member"]
    enough = P["close"].notna().rolling(252).count() >= 252
    row = s.iloc[-1].where(member.iloc[-1]).where(enough.iloc[-1]).dropna()
    picks = row.sort_values(ascending=False).head(k)
    regime = "RISK-OFF (rebound sleeve)" if risk_off(P).iloc[-1] \
        else "RISK-ON (momentum sleeve)"
    return pd.DataFrame({"score": picks}), regime
