"""Transparent, causal candidate scores (days x tickers).

Every score at row d uses only closes through day d (strict: shift where a
window would otherwise peek). Higher score = more attractive to BUY under the
underwater-avoidance objective. These are the honest yardsticks the HF model
must beat.

Hypothesis: a name that rarely trades below a recent purchase is one in a
*smooth, low-volatility uptrend*. So beyond plain momentum we score trend
quality (drift / risk) and trend smoothness (how linear the log-price path is).
"""
import numpy as np
import pandas as pd

from data import load_panel


def _logclose() -> pd.DataFrame:
    return np.log(load_panel()["close"])


def mom_12_1() -> pd.DataFrame:
    """12-1 momentum: 252d return skipping the most recent 21d. Repo champion
    for the *return* objective; a yardstick here."""
    c = load_panel()["close"]
    return c.shift(21) / c.shift(252) - 1.0


def ret_126() -> pd.DataFrame:
    c = load_panel()["close"]
    return c / c.shift(126) - 1.0


def low_vol(win: int = 126) -> pd.DataFrame:
    """Negative trailing realized vol (low-vol anomaly direction)."""
    r = _logclose().diff()
    return -r.rolling(win).std()


def trend_quality(win: int = 126) -> pd.DataFrame:
    """Trailing Sharpe-like trend: mean daily log-return / its std over `win`.
    High drift per unit risk -> spends little time below a recent entry."""
    r = _logclose().diff()
    mu = r.rolling(win).mean()
    sd = r.rolling(win).std()
    return mu / sd.replace(0, np.nan)


def downside_trend(win: int = 126) -> pd.DataFrame:
    """Drift divided by *downside* deviation (Sortino-style). Penalizes only
    the down moves, which is exactly what 'goes below entry' is about."""
    r = _logclose().diff()
    mu = r.rolling(win).mean()
    dn = r.where(r < 0, 0.0)
    dd = np.sqrt((dn ** 2).rolling(win).mean())
    return mu / dd.replace(0, np.nan)


def trend_smoothness(win: int = 126) -> pd.DataFrame:
    """Signed R^2 of log-price vs time: how linear the path is, times the sign
    of the slope. A clean, steady climb scores high; a choppy or downward path
    scores low. Computed in closed form per rolling window.

    For y = log price over the window and x = 0..win-1:
        slope = cov(x,y)/var(x);  R^2 = cov(x,y)^2 / (var(x) var(y))
    score = sign(slope) * R^2.
    """
    y = _logclose()
    n = win
    x = np.arange(n, dtype=float)
    xbar = x.mean()
    varx = ((x - xbar) ** 2).sum()
    sy = y.rolling(n).sum()
    # sum of (local x)*y over each window, fully vectorized. With absolute
    # series position p (same across columns) and window start s = p - n + 1,
    # the local weight on row j is (j - s), so
    #   sum_local_xy = RS(p*y, n) - s * RS(y, n),  s taken at the window's end.
    pos = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    py = y.mul(pos, axis=0)
    s_end = pos - (n - 1)                       # absolute start index per row
    sxy = py.rolling(n).sum() - sy.mul(s_end, axis=0)
    ybar = sy / n
    cov_xy = sxy - n * xbar * ybar
    # var(y)*n = sum(y^2) - n*ybar^2
    sy2 = (y * y).rolling(n).sum()
    var_y = sy2 - n * ybar * ybar
    denom = (varx * var_y).replace(0, np.nan)
    r2 = (cov_xy ** 2) / denom
    slope = cov_xy / varx
    return np.sign(slope) * r2


def low_maxdd(win: int = 252) -> pd.DataFrame:
    """Negative trailing max drawdown over `win`. Names with shallow recent
    drawdowns have been climbing steadily."""
    c = load_panel()["close"]
    roll_max = c.rolling(win, min_periods=win // 2).max()
    dd = c / roll_max - 1.0          # current dd vs window high (<=0)
    worst = dd.rolling(win, min_periods=win // 2).min()
    return -worst


def self_underwater(win: int = 252, h: int = 63) -> pd.DataFrame:
    """Direct in-sample proxy of the objective: over the trailing window, the
    fraction of days the price sat below where it was `h` days earlier; negated
    so 'rarely-underwater-historically' scores high. Causal: uses closes <= d."""
    c = load_panel()["close"]
    below = (c < c.shift(h)).astype(float)
    return -below.rolling(win).mean()


def random_scores(seed: int = 0) -> pd.DataFrame:
    """Reproducible random control aligned to the panel."""
    c = load_panel()["close"]
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(c.shape)
    return pd.DataFrame(arr, index=c.index, columns=c.columns)


REGISTRY = {
    "mom_12_1": mom_12_1,
    "ret_126": ret_126,
    "low_vol": low_vol,
    "trend_quality": trend_quality,
    "downside_trend": downside_trend,
    "trend_smoothness": trend_smoothness,
    "low_maxdd": low_maxdd,
    "self_underwater": self_underwater,
    "random": random_scores,
}


def build(name: str) -> np.ndarray:
    """Return a causal score matrix as numpy (days x tickers)."""
    return REGISTRY[name]().to_numpy(float)
