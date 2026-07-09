"""Strategy signal definitions (fixed ex-ante; parameter grids are small
and pre-specified — IS picks one config per family, OOS runs it once).

Each signal function takes the per-bond daily frame g (sorted by date) and
returns a boolean Series: True on day t means "enter at the next day's
customer-buy print". Only information through day t is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _trailing_med(g: pd.DataFrame, col: str, window_days: int,
                  min_obs: int = 5) -> pd.Series:
    """Median of `col` over the trailing `window_days` calendar days,
    excluding today; NaN when fewer than min_obs observations."""
    s = g.set_index("date")[col]
    med = (s.rolling(f"{window_days}D", min_periods=min_obs)
           .median().shift(1))
    return med.reset_index(drop=True)


def firesale(discount: float, require_sell_pressure: bool = True):
    """Dislocation reversion: today's mid printed `discount` points below
    the trailing 20-day median mid; optionally require that the day's
    customer flow was net selling (p_par > s_par)."""
    def fn(g: pd.DataFrame) -> pd.Series | None:
        if "p_px" not in g or "s_px" not in g:
            return None
        ref = _trailing_med(g, "mid", 20)
        sig = (g["mid"] - ref) <= -discount
        if require_sell_pressure:
            p_par = g.get("p_par")
            s_par = g.get("s_par")
            if p_par is None:
                return None
            pressure = p_par.fillna(0) > s_par.fillna(0) if s_par is not None \
                else p_par.notna()
            sig &= pressure
        return sig.fillna(False)
    return fn


def momentum(window_days: int, thresh: float):
    """Trailing price momentum: mid gained >= thresh points over the
    trailing window (vs the median mid at the start of the window)."""
    def fn(g: pd.DataFrame) -> pd.Series | None:
        if "mid" not in g:
            return None
        s = g.set_index("date")["mid"]
        past = (s.rolling(f"{window_days}D", min_periods=5).median()
                .shift(1))
        recent = s.rolling("10D", min_periods=1).median()
        sig = (recent - past) >= thresh
        return sig.reset_index(drop=True).fillna(False)
    return fn


def new_issue(max_age_days: int):
    """Buy in the first `max_age_days` after a bond's first-ever print
    (new-issue concession capture)."""
    def fn(g: pd.DataFrame) -> pd.Series | None:
        age = (g["date"] - g["date"].iloc[0]).dt.days
        # fire across the post-issue age band; the engine's per-bond cooldown
        # and first-available-S-print rule collapse this to one entry.
        return ((age >= 1) & (age <= max_age_days)).fillna(False)
    return fn


def dealer_roundtrip():
    """Upper-bound diagnostic, NOT tradable by a customer: 'enter' whenever
    both sides printed (buy at P, sell at S same day would be the dealer's
    round trip). Used only to size the spread pool in FINDINGS."""
    def fn(g: pd.DataFrame) -> pd.Series | None:
        if "p_px" not in g or "s_px" not in g:
            return None
        return (g["p_px"].notna() & g["s_px"].notna()).fillna(False)
    return fn


GRIDS = {
    "firesale": [
        {"discount": 1.0, "require_sell_pressure": True},
        {"discount": 1.5, "require_sell_pressure": True},
        {"discount": 2.5, "require_sell_pressure": True},
        {"discount": 1.5, "require_sell_pressure": False},
    ],
    "momentum": [
        {"window_days": 60, "thresh": 1.0},
        {"window_days": 120, "thresh": 2.0},
    ],
    "new_issue": [
        {"max_age_days": 10},
        {"max_age_days": 30},
    ],
}

FACTORIES = {
    "firesale": firesale,
    "momentum": momentum,
    "new_issue": new_issue,
}

MIN_HOLDS = {"firesale": 10, "momentum": 20, "new_issue": 30}
