"""Faithful replication of THE ROTATOR (v2.0) for our DCA evaluation harness.

Spec (from the published factsheet):
  * Universe: large-cap US equities, point-in-time membership.
  * LEADERSHIP SCORE = average(3-month return, 6-month return)
                       * fraction of rising weeks over the last 6 months.
    "Strong AND steady beats strong alone."
  * MAX buy-screen: exclude any name whose largest 1-day gain in the past
    month ranks in the top 5% of the universe that day.
  * Hold the top 3 (equal weight). Retain an existing holding while it stays
    in the top 8; sell it once it drops below rank 8.
  * Bear switch: if SPY closes below its 210-day average, go fully to CASH;
    re-enter at the first rebalance after SPY reclaims the 210dma.
  * Biweekly cadence, next-open fills.

Rendering into our DCA harness (so it is directly comparable to SUMMIT):
  * `build_scores(P)` -> leadership score, NaN where MAX-screened or on bear
    days (no buys in cash regime). The engine buys the top-k (k=3) each
    biweekly contribution.
  * `build_sell(P)` -> boolean: True where a *held* name should be liquidated
    at the next open = (not in the top 8) OR (bear regime). The engine only
    acts on names actually held, so marking all-non-top-8 names sell-eligible
    reproduces the rank-8 retention buffer and the bear-to-cash exit.

Everything is trailing-only (audited via audit.audit_builder).
"""
import numpy as np
import pandas as pd

import data as data_mod

R3, R6 = 63, 126          # 3- and 6-month look-backs (trading days)
RISING_WEEKS = 26         # 6 months of weeks
MAX_LB = 21               # 1-month window for the MAX screen
MAX_PCTL = 0.95           # exclude top 5% MAX
TOP_HOLD = 3              # positions
TOP_KEEP = 8              # retention buffer
SPY_MA = 210


def _rising_week_frac(close: pd.DataFrame) -> pd.DataFrame:
    """Fraction of rising weeks over the last 26 completed weeks, as a daily
    series (each day uses the most recent completed Friday — causal)."""
    weekly = close.resample("W-FRI").last()
    wret = weekly.pct_change(fill_method=None)
    frac = (wret > 0).rolling(RISING_WEEKS).mean()
    return frac.reindex(close.index, method="ffill")


def leadership(P: dict) -> pd.DataFrame:
    close = P["close"]
    r3 = close.pct_change(R3, fill_method=None)
    r6 = close.pct_change(R6, fill_method=None)
    avg = (r3 + r6) / 2.0
    return avg * _rising_week_frac(close)


def _max_screen(P: dict) -> pd.DataFrame:
    """True where the name's max 1-day gain over the past month is in the top
    5% of the (member) universe that day -> excluded from buying."""
    close, member = P["close"], P["member"]
    ret = close.pct_change(fill_method=None)
    max1d = ret.rolling(MAX_LB).max()
    rk = max1d.where(member).rank(axis=1, pct=True)
    return rk > MAX_PCTL


def _spy_bear(index: pd.DatetimeIndex) -> pd.Series:
    spy = data_mod.load_benchmark("SPY")["Close"].reindex(index).ffill()
    return spy < spy.rolling(SPY_MA).mean()


def build_scores(P: dict) -> pd.DataFrame:
    score = leadership(P)
    score = score.where(~_max_screen(P))          # drop screened names
    bear = _spy_bear(P["close"].index).to_numpy()
    arr = score.to_numpy(copy=True)
    arr[bear, :] = np.nan                          # no buys in cash regime
    return pd.DataFrame(arr, index=score.index, columns=score.columns)


def build_sell(P: dict) -> pd.DataFrame:
    """Liquidate a holding if it falls below rank 8, or sell everything in the
    bear regime. Ranks computed among members on raw leadership (the screen
    governs buying, not holding — a held name that later trips the MAX screen
    is still ranked normally for retention)."""
    close, member = P["close"], P["close"].notna() & P["member"]
    score = leadership(P).where(member)
    rank = score.rank(axis=1, ascending=False)
    in_top8 = (rank <= TOP_KEEP).fillna(False)
    arr = (~in_top8).to_numpy(copy=True)           # not in top 8 -> sell-eligible
    bear = _spy_bear(close.index).to_numpy()
    arr[bear, :] = True                            # bear -> sell all
    return pd.DataFrame(arr, index=rank.index, columns=rank.columns)


def current_picks(k: int = TOP_HOLD):
    P = data_mod.build_panel()
    s = build_scores(P)
    member = P["member"]
    enough = P["close"].notna().rolling(252).count() >= 252
    row = (s.iloc[-1].where(member.iloc[-1]).where(enough.iloc[-1])
           .dropna().sort_values(ascending=False))
    bear = bool(_spy_bear(P["close"].index).iloc[-1])
    regime = "CASH (SPY below 210dma)" if bear else "INVESTED (top-3 leaders)"
    return list(row.index[:k]), regime
