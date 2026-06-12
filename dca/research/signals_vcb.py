"""Volatility-compression / breakout (VCB) signal family for the biweekly
DCA picker.

Every builder takes the panels dict from `data.build_panel()` and returns a
scores DataFrame (dates x tickers).  Causality: every value at row d uses
only trailing windows ending at the close of d (rolling .rank(pct=True) is
the rank of the LAST element inside a trailing window -> point-in-time).
NaN = "not a candidate today" (the engine skips NaN scores).
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- helpers
def _ret(close, n):
    return close / close.shift(n) - 1.0


def _rv(close, n):
    return close.pct_change().rolling(n).std()


def _uptrend(close, ma=200, mom=126):
    """Uptrend gate: price above its 200dma AND positive 6m return."""
    return (close > close.rolling(ma).mean()) & (_ret(close, mom) > 0)


def _trail_pct(df, window=252, min_periods=126):
    """Trailing percentile of today's value vs the stock's own last
    `window` days (no full-sample stats)."""
    return df.rolling(window, min_periods=min_periods).rank(pct=True)


def _true_range(P):
    h, l, c = P["high"], P["low"], P["close"]
    pc = c.shift(1)
    a = (h - l).to_numpy(float)
    b = (h - pc).abs().to_numpy(float)
    d = (l - pc).abs().to_numpy(float)
    tr = np.fmax(a, np.fmax(b, d))
    return pd.DataFrame(tr, index=c.index, columns=c.columns)


# ---------------------------------------------------------------- 1. vol compression
def vcb_volcomp(P, short=20, long=120):
    """Realized-vol compression rv(short)/rv(long); low ratio = squeeze.
    Score = 1/ratio, only inside an uptrend (>200dma, +6m)."""
    c = P["close"]
    ratio = _rv(c, short) / _rv(c, long)
    return (1.0 / ratio).where(_uptrend(c))


def vcb_volcomp_rank(P, short=20, long=120, mom=126):
    """Cross-sectional rank combo: rank(compression) + rank(6m momentum),
    gated by uptrend."""
    c = P["close"]
    ratio = _rv(c, short) / _rv(c, long)
    comp_rk = (-ratio).rank(axis=1, pct=True)
    mom_rk = _ret(c, mom).rank(axis=1, pct=True)
    return (comp_rk + mom_rk).where(_uptrend(c))


# ---------------------------------------------------------------- 2. range contraction
def vcb_range(P, short=20, long=120):
    """(short high-low range)/(long high-low range); score = inverse ratio,
    uptrend-gated."""
    h, l, c = P["high"], P["low"], P["close"]
    rng_s = h.rolling(short).max() - l.rolling(short).min()
    rng_l = h.rolling(long).max() - l.rolling(long).min()
    ratio = rng_s / rng_l
    return (1.0 / ratio).where(_uptrend(c))


def vcb_bbw(P, n=20, lookback=252):
    """Bollinger bandwidth percentile vs own trailing year (low = squeeze),
    uptrend-gated. Score = 1 - trailing percentile."""
    c = P["close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    bbw = 4.0 * sd / ma
    pct = _trail_pct(bbw, lookback)
    return (1.0 - pct).where(_uptrend(c))


# ---------------------------------------------------------------- 3. Donchian freshness
def vcb_donchian(P, lookback=252, near_pct=0.02, voln=60, terc=1 / 3):
    """Close within `near_pct` of the `lookback`-day high AND `voln`-day
    realized vol in the bottom tercile of its own trailing year.
    Score = 1 - vol percentile (quietest leaders first)."""
    c, h = P["close"], P["high"]
    hi = h.rolling(lookback).max()
    near = c >= hi * (1.0 - near_pct)
    vol_pct = _trail_pct(_rv(c, voln), 252)
    cond = near & (vol_pct <= terc)
    return (1.0 - vol_pct).where(cond)


# ---------------------------------------------------------------- 4. ATR contraction
def vcb_atr(P, short=20, long=120, mom=126):
    """ATR(short)/ATR(long) contraction with positive momentum gate."""
    c = P["close"]
    tr = _true_range(P)
    ratio = tr.rolling(short).mean() / tr.rolling(long).mean()
    gate = (_ret(c, mom) > 0) & (c > c.rolling(200).mean())
    return (1.0 / ratio).where(gate)


# ---------------------------------------------------------------- 5. base breakout
def vcb_basebreak(P, fresh=10, base_min=63, lookback=252, depth_w=True):
    """New `lookback`-day closing high within the last `fresh` days, after
    spending >= `base_min` days below the prior high (a "base").  Score =
    base length in days, optionally x (1 + base depth).  Score stays alive
    for `fresh` days after the breakout."""
    c = P["close"]
    prior_hi = c.shift(1).rolling(lookback).max()
    nh = (c > prior_hi)                      # new 252d closing high today
    pos = np.arange(len(c))[:, None].astype(float)
    last_nh = pd.DataFrame(np.where(nh.to_numpy(bool), pos, np.nan),
                           index=c.index, columns=c.columns).ffill()
    gap = pos - last_nh.shift(1).to_numpy()  # days since previous new high
    gap = pd.DataFrame(gap, index=c.index, columns=c.columns)
    brk = nh & (gap >= base_min)
    score = gap.where(brk)
    if depth_w:
        depth = 1.0 - c.rolling(126).min() / prior_hi   # base depth proxy
        score = score * (1.0 + depth.clip(lower=0.0))
    # keep the entry window open `fresh` days, while still in uptrend
    score = score.ffill(limit=fresh - 1).where(_uptrend(c))
    return score


# ---------------------------------------------------------------- 6. squeeze->expansion
def vcb_sqz_exp(P, lag=21, sqz_pct=0.20, n=20, lookback=252):
    """Bollinger-bandwidth percentile was in the bottom `sqz_pct` of its
    trailing year `lag` days ago AND the last `lag`-day return is positive
    (the expansion has begun).  Score = 1-month return among qualifiers."""
    c = P["close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    bbw = 4.0 * sd / ma
    pct = _trail_pct(bbw, lookback)
    r = _ret(c, lag)
    cond = (pct.shift(lag) <= sqz_pct) & (r > 0)
    return r.where(cond)


def vcb_sqz_exp_tight(P, lag=21, sqz_pct=0.20, n=20, lookback=252):
    """Same condition, but score by how tight the squeeze was (1 - lagged
    percentile) instead of expansion strength."""
    c = P["close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    bbw = 4.0 * sd / ma
    pct = _trail_pct(bbw, lookback)
    r = _ret(c, lag)
    cond = (pct.shift(lag) <= sqz_pct) & (r > 0)
    return (1.0 - pct.shift(lag)).where(cond)


BUILDERS = {
    "vcb_volcomp": vcb_volcomp,
    "vcb_volcomp_rank": vcb_volcomp_rank,
    "vcb_range": vcb_range,
    "vcb_bbw": vcb_bbw,
    "vcb_donchian": vcb_donchian,
    "vcb_atr": vcb_atr,
    "vcb_basebreak": vcb_basebreak,
    "vcb_sqz_exp": vcb_sqz_exp,
    "vcb_sqz_exp_tight": vcb_sqz_exp_tight,
}
