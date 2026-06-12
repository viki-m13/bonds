"""Momentum / trend / 52-week-high signal family for the biweekly DCA picker.

Every builder takes the panels dict (open/high/low/close/volume/member) and
returns a scores DataFrame (dates x tickers). Causality: every operation is a
trailing rolling window or a non-negative shift, so scores.loc[d] uses data
through the close of d only. Cross-sectional ranks are within-row only.

Auditable via audit.audit_builder(builder).
"""
import numpy as np
import pandas as pd


# ---------- 1. plain momentum ----------------------------------------------

def mom(panels, lb=126, skip=0):
    """Trailing total return over `lb` days, skipping the last `skip` days."""
    c = panels["close"]
    return c.shift(skip) / c.shift(lb) - 1.0


def mom_63(p):   return mom(p, 63)
def mom_126(p):  return mom(p, 126)
def mom_189(p):  return mom(p, 189)
def mom_252(p):  return mom(p, 252)
def mom_12_1(p): return mom(p, 252, skip=21)          # classic 12-1
def mom_int(p):  return mom(p, 252, skip=126)         # intermediate 12-7


# ---------- 2. momentum acceleration ----------------------------------------

def mom_accel(panels):
    """63d return minus the previous (non-overlapping) 63d return."""
    c = panels["close"]
    r63 = c / c.shift(63) - 1.0
    return r63 - r63.shift(63)


def mom_accel2(panels):
    """Second derivative: change in acceleration over consecutive quarters."""
    c = panels["close"]
    r63 = c / c.shift(63) - 1.0
    a = r63 - r63.shift(63)
    return a - a.shift(63)


def mom_accel_pos(panels):
    """Acceleration gated on positive 6m momentum (accelerating winners)."""
    c = panels["close"]
    r63 = c / c.shift(63) - 1.0
    acc = r63 - r63.shift(63)
    m126 = c / c.shift(126) - 1.0
    return acc.where(m126 > 0)


# ---------- 3. 52-week-high proximity ---------------------------------------

def high52_prox(panels):
    """close / rolling 252d max close (trailing, includes today)."""
    c = panels["close"]
    return c / c.rolling(252, min_periods=252).max()


def high52_fresh(panels):
    """Negative days since the close last printed a 252d rolling high.

    Fewer days since a fresh 52w high => higher score.
    """
    c = panels["close"]
    rmax = c.rolling(252, min_periods=252).max()
    is_high = (c >= rmax - 1e-12) & rmax.notna()
    pos = pd.DataFrame(
        np.broadcast_to(np.arange(len(c))[:, None], c.shape).copy(),
        index=c.index, columns=c.columns, dtype=float)
    last = pos.where(is_high).ffill()
    days_since = pos - last
    return -days_since.where(rmax.notna())


# ---------- 4. trend quality -------------------------------------------------

def frac_above_50dma(panels, frac_lb=126):
    """Share of the last `frac_lb` days the stock closed above its 50d MA."""
    c = panels["close"]
    ma50 = c.rolling(50, min_periods=50).mean()
    above = (c > ma50).where(c.notna() & ma50.notna())
    return above.rolling(frac_lb, min_periods=frac_lb).mean()


def frac_above_50dma_252(p):
    return frac_above_50dma(p, 252)


def _trend_corr(c, W):
    """Rolling Pearson corr of log price with time over trailing W days.

    Monotonic transform of the regression t-stat; sign = trend direction.
    Vectorized via rolling sums (x = within-window position 0..W-1).
    """
    y = np.log(c)
    n = np.arange(len(c), dtype=float)
    z = y.mul(n, axis=0)                       # global-index-weighted price
    Sy = y.rolling(W, min_periods=W).sum()
    Szy = z.rolling(W, min_periods=W).sum()
    # within-window x starts at global index (t - W + 1)
    start = pd.Series(n - W + 1, index=c.index)
    Sxy = Szy.sub(Sy.mul(start, axis=0))       # sum_i i * y_i, i=0..W-1
    sx = W * (W - 1) / 2.0
    sxx = (W - 1) * W * (2 * W - 1) / 6.0
    var_x = sxx / W - (sx / W) ** 2
    mean_y = Sy / W
    cov = Sxy / W - (sx / W) * mean_y
    std_y = y.rolling(W, min_periods=W).std(ddof=0)
    corr = cov / (np.sqrt(var_x) * std_y)
    return corr, cov / var_x                    # corr, slope (log/day)


def trend_tstat_126(panels):
    """Signed trend strength: corr(log price, time) over 126d."""
    corr, _ = _trend_corr(panels["close"], 126)
    return corr


def clenow_126(panels):
    """Clenow momentum: annualized log-slope * R^2 over 126d."""
    corr, slope = _trend_corr(panels["close"], 126)
    return slope * 252.0 * corr ** 2


def clenow_252(panels):
    corr, slope = _trend_corr(panels["close"], 252)
    return slope * 252.0 * corr ** 2


def fip_126(panels):
    """Frog-in-the-pan style: 126d momentum * share of up days."""
    c = panels["close"]
    ret = c.pct_change(fill_method=None)
    up = (ret > 0).where(ret.notna())
    up_share = up.rolling(126, min_periods=126).mean()
    m = c / c.shift(126) - 1.0
    return m * up_share


# ---------- 5. MA alignment ---------------------------------------------------

def ma_align(panels):
    """price>50dma>100dma>200dma score (0-4) + clipped distance tiebreak."""
    c = panels["close"]
    ma50 = c.rolling(50, min_periods=50).mean()
    ma100 = c.rolling(100, min_periods=100).mean()
    ma200 = c.rolling(200, min_periods=200).mean()
    ok = c.notna() & ma200.notna()
    align = ((c > ma50).astype(float) + (ma50 > ma100).astype(float)
             + (ma100 > ma200).astype(float) + (c > ma200).astype(float))
    dist = (c / ma200 - 1.0).clip(-0.9, 0.9)
    return (align + dist).where(ok)


def dist_200dma(panels):
    """Distance of close above its 200d MA."""
    c = panels["close"]
    ma200 = c.rolling(200, min_periods=200).mean()
    return c / ma200 - 1.0


# ---------- 6. combos ----------------------------------------------------------

def mom_x_smooth(panels):
    """Cross-sectional rank(12-1 momentum) + rank(126d trend corr)."""
    m = mom(panels, 252, skip=21)
    corr, _ = _trend_corr(panels["close"], 126)
    return m.rank(axis=1, pct=True) + corr.rank(axis=1, pct=True)


def high52_gated_mom(panels):
    """52w-high proximity, eligible only when 6m momentum is positive."""
    prox = high52_prox(panels)
    m126 = mom(panels, 126)
    return prox.where(m126 > 0)


def mom126_x_high52(panels):
    """rank(6m momentum) + rank(52w-high proximity)."""
    m = mom(panels, 126)
    prox = high52_prox(panels)
    return m.rank(axis=1, pct=True) + prox.rank(axis=1, pct=True)


BUILDERS = {
    "mom_ret63": mom_63,
    "mom_ret126": mom_126,
    "mom_ret189": mom_189,
    "mom_ret252": mom_252,
    "mom_12_1": mom_12_1,
    "mom_int_12_7": mom_int,
    "mom_accel": mom_accel,
    "mom_accel2": mom_accel2,
    "mom_accel_pos": mom_accel_pos,
    "mom_high52_prox": high52_prox,
    "mom_high52_fresh": high52_fresh,
    "mom_frac50dma_126": frac_above_50dma,
    "mom_frac50dma_252": frac_above_50dma_252,
    "mom_trend_tstat126": trend_tstat_126,
    "mom_clenow126": clenow_126,
    "mom_clenow252": clenow_252,
    "mom_fip126": fip_126,
    "mom_ma_align": ma_align,
    "mom_dist200": dist_200dma,
    "mom_x_smooth": mom_x_smooth,
    "mom_high52_gated": high52_gated_mom,
    "mom126_x_high52": mom126_x_high52,
}


# ---------- sensitivity variants around 12-1 ---------------------------------

def mom_126_21(p):  return mom(p, 126, skip=21)       # 6-1
def mom_189_21(p):  return mom(p, 189, skip=21)       # 9-1
def mom_252_10(p):  return mom(p, 252, skip=10)
def mom_252_42(p):  return mom(p, 252, skip=42)       # 12-2
def mom_231_21(p):  return mom(p, 231, skip=21)

SENSITIVITY = {
    "mom_6_1": mom_126_21,
    "mom_9_1": mom_189_21,
    "mom_12_0p5": mom_252_10,
    "mom_12_2": mom_252_42,
    "mom_11_1": mom_231_21,
}
