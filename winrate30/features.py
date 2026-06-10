"""Feature/condition computation on the price panel.

Everything is computed on wide (dates x tickers) frames and exposed as
boolean numpy arrays so rule evaluation is a chain of cheap elementwise ANDs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import HORIZON, MIN_PRICE
from universe import MARKET_TICKER, VIX_TICKER

# Condition groups searched by the rule grid. None = "no constraint".
# "market" has no None option: spy_above_200 is a hard prespecified gate.
# Broken-market rebound rules looked great in-sample but failed
# catastrophically out-of-sample (March 2020: 13% hit rate), so the tool
# never recommends while the S&P 500 is below its 200-day average.
CONDITION_GROUPS: dict[str, list[str | None]] = {
    "market": ["spy_above_200"],
    "vix":    [None, "vix_lt15", "vix_lt20", "vix_gt30"],
    "trend":  [None, "above_200", "golden"],
    "rsi":    [None, "rsi_lt25", "rsi_lt30", "rsi_lt35"],
    "dd":     [None, "dd_gt_-05", "dd_gt_-10", "dd_-10_-25", "dd_lt_-25"],
    "vol":    [None, "vol_vlow", "vol_low", "vol_high"],
    "base":   [None, "base_hi70", "base_hi75"],
    "mom":    [None, "mom_pos"],
}

DESCRIPTIONS = {
    "spy_above_200": "S&P 500 above its 200-day average (market uptrend)",
    "spy_below_200": "S&P 500 below its 200-day average (market downtrend)",
    "vix_lt15": "VIX below 15 (very calm market)",
    "vix_lt20": "VIX below 20 (calm market)",
    "vix_gt30": "VIX above 30 (panic)",
    "above_200": "stock above its 200-day average",
    "golden": "stock above 200-day avg and 50-day avg above 200-day avg",
    "rsi_lt25": "RSI(14) below 25 (extremely oversold)",
    "rsi_lt30": "RSI(14) below 30 (deeply oversold)",
    "rsi_lt35": "RSI(14) below 35 (oversold)",
    "dd_gt_-05": "within 5% of its 52-week high",
    "dd_gt_-10": "within 10% of its 52-week high",
    "dd_-10_-25": "10-25% below its 52-week high",
    "dd_lt_-25": "more than 25% below its 52-week high",
    "vol_vlow": "volatility in the bottom fifth of its own 1-year range",
    "vol_low": "volatility in the bottom third of its own 1-year range",
    "vol_high": "volatility in the top third of its own 1-year range",
    "base_hi70": "steady compounder: >70% of its past-3y 1-month windows were positive",
    "base_hi75": "steady compounder: >75% of its past-3y 1-month windows were positive",
    "mom_pos": "positive 12-month momentum (excluding last month)",
}


@dataclass
class Panel:
    index: pd.DatetimeIndex
    tickers: list[str]
    close: np.ndarray                      # float (days x tickers)
    fwd: np.ndarray                        # forward HORIZON-day return
    valid_hist: np.ndarray                 # enough history + price filter
    conds: dict[str, np.ndarray] = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(default_factory=dict)  # rsi, dd, ... for display

    @property
    def valid(self) -> np.ndarray:
        """Valid for backtesting: history ok AND forward return known."""
        return self.valid_hist & ~np.isnan(self.fwd)


def _rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, min_periods=period).mean()
    avg_dn = dn.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_up / avg_dn
    return 100 - 100 / (1 + rs)


def compute_panel(prices: pd.DataFrame) -> Panel:
    spy = prices[MARKET_TICKER]
    vix = prices[VIX_TICKER].ffill()
    close = prices.drop(columns=[MARKET_TICKER, VIX_TICKER])
    tickers = list(close.columns)
    n_days, n_tk = close.shape

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = _rsi(close)
    high252 = close.rolling(252, min_periods=200).max()
    dd = close / high252 - 1.0
    ret1 = close.pct_change(fill_method=None)
    vol21 = ret1.rolling(21).std() * np.sqrt(252)
    vol_pct = vol21.rolling(252).rank(pct=True)
    mom = close.shift(HORIZON) / close.shift(252) - 1.0
    fwd = close.shift(-HORIZON) / close - 1.0

    # Trailing base rate: fraction of the stock's own past-3y 21-day windows
    # that were positive (causal: uses returns ending today or earlier).
    past21 = close / close.shift(HORIZON) - 1.0
    pos_ind = (past21 > 0).astype(float).where(past21.notna())
    base_rate = pos_ind.rolling(756, min_periods=500).mean()

    spy_sma200 = spy.rolling(200).mean()
    spy_above = (spy > spy_sma200) & spy_sma200.notna()
    spy_below = (spy < spy_sma200) & spy_sma200.notna()

    def bcast(s: pd.Series) -> np.ndarray:
        return np.repeat(s.to_numpy(dtype=bool)[:, None], n_tk, axis=1)

    conds = {
        "spy_above_200": bcast(spy_above),
        "spy_below_200": bcast(spy_below),
        "vix_lt15": bcast((vix < 15).fillna(False)),
        "vix_lt20": bcast((vix < 20).fillna(False)),
        "vix_gt30": bcast((vix > 30).fillna(False)),
        "above_200": (close > sma200).to_numpy(),
        "golden": ((close > sma200) & (sma50 > sma200)).to_numpy(),
        "rsi_lt25": (rsi < 25).to_numpy(),
        "rsi_lt30": (rsi < 30).to_numpy(),
        "rsi_lt35": (rsi < 35).to_numpy(),
        "dd_gt_-05": (dd > -0.05).to_numpy(),
        "dd_gt_-10": (dd > -0.10).to_numpy(),
        "dd_-10_-25": ((dd <= -0.10) & (dd > -0.25)).to_numpy(),
        "dd_lt_-25": (dd <= -0.25).to_numpy(),
        "vol_vlow": (vol_pct < 0.20).to_numpy(),
        "vol_low": (vol_pct < 1 / 3).to_numpy(),
        "vol_high": (vol_pct > 2 / 3).to_numpy(),
        "base_hi70": (base_rate > 0.70).to_numpy(),
        "base_hi75": (base_rate > 0.75).to_numpy(),
        "mom_pos": (mom > 0).to_numpy(),
    }

    valid_hist = (
        sma200.notna() & vol_pct.notna() & mom.notna()
        & (close > MIN_PRICE)
    ).to_numpy()

    extras = {
        "rsi": rsi.to_numpy(),
        "dd": dd.to_numpy(),
        "vol_pct": vol_pct.to_numpy(),
        "mom": mom.to_numpy(),
    }

    return Panel(
        index=close.index, tickers=tickers,
        close=close.to_numpy(), fwd=fwd.to_numpy(),
        valid_hist=valid_hist, conds=conds, extras=extras,
    )


def rule_mask(panel: Panel, rule: list[str]) -> np.ndarray:
    """Boolean (days x tickers) mask where all conditions of the rule hold."""
    mask = panel.valid_hist.copy()
    for name in rule:
        mask &= panel.conds[name]
    return mask


def describe_rule(rule: list[str]) -> str:
    if not rule:
        return "any stock, any day"
    return "; ".join(DESCRIPTIONS[c] for c in rule)
