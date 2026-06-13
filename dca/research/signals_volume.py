"""Volume / accumulation signal family for the biweekly DCA picker.

Every builder takes the panels dict from `data.build_panel()` and returns a
scores DataFrame (dates x tickers). Causality: every computation is a
TRAILING rolling window or a cross-sectional transform within the row date,
so scores.loc[d] uses information through the close of d only.

Conventions
-----------
* higher score = better; NaN = ineligible at that date
* `_xrank` = cross-sectional percentile rank within the row (allowed by the
  causality contract)
* gated signals fall back to a heavily penalized secondary score so the
  engine can still fill k slots when few names qualify
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- helpers
def _xrank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank within each row (trailing-safe)."""
    return df.rank(axis=1, pct=True)


def _ret(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close.pct_change(n, fill_method=None)


def _clv(P: dict) -> pd.DataFrame:
    """Close location value in the daily range, in [-1, 1]."""
    h, l, c = P["high"], P["low"], P["close"]
    rng = (h - l)
    clv = ((c - l) - (h - c)) / rng.where(rng > 0)
    return clv.fillna(0.0).where(c.notna())


def _vol(P: dict) -> pd.DataFrame:
    """Volume, NaN where close is NaN (keeps rolling stats honest)."""
    return P["volume"].where(P["close"].notna())


# ------------------------------------------------- 1. high-volume premium
def hv_premium(P: dict, vol_win: int = 20, base_win: int = 120,
               ret_win: int = 20, mode: str = "gate") -> pd.DataFrame:
    """Gervais et al. high-volume return premium.

    Abnormal volume = mean(vol, vol_win) / mean(vol, base_win), interacted
    with recent return. mode='gate': rank abnormal volume only among names
    with positive ret_win return. mode='interact': rank(abn) * rank(ret).
    """
    v = _vol(P)
    abn = (v.rolling(vol_win, min_periods=vol_win).mean()
           / v.rolling(base_win, min_periods=base_win).mean())
    r = _ret(P["close"], ret_win)
    if mode == "gate":
        return _xrank(abn.where(r > 0))
    return _xrank(abn) * _xrank(r)


# ------------------------------------------------- 2. up/down volume ratio
def updown_ratio(P: dict, win: int = 63, dollar: bool = False) -> pd.DataFrame:
    """log(up-day volume / down-day volume) over `win` days.

    dollar=True uses dollar volume (close * shares).
    """
    c = P["close"]
    v = _vol(P)
    if dollar:
        v = v * c
    dr = c.diff()
    up = v.where(dr > 0, 0.0).where(c.notna())
    dn = v.where(dr < 0, 0.0).where(c.notna())
    su = up.rolling(win, min_periods=win).sum()
    sd = dn.rolling(win, min_periods=win).sum()
    return np.log((su + 1.0) / (sd + 1.0))


# --------------------------------------------------------- 3. OBV trend
def obv_trend(P: dict, win: int = 63) -> pd.DataFrame:
    """63d change of cumulative signed volume, normalized by total volume.

    Equivalent to (up_vol - down_vol) / total_vol over the window: bounded
    [-1, 1], cross-sectionally comparable.
    """
    c = P["close"]
    v = _vol(P)
    sv = (np.sign(c.diff()) * v).where(c.notna())
    num = sv.rolling(win, min_periods=win).sum()
    den = v.rolling(win, min_periods=win).sum()
    return num / den.where(den > 0)


def obv_divergence(P: dict, win: int = 126) -> pd.DataFrame:
    """Accumulation divergence: OBV near its trailing high while price isn't.

    Position of OBV in its trailing `win`-day [min, max] range minus the
    position of price in its own trailing range; positive = OBV stronger
    than price (stealth accumulation).
    """
    c = P["close"]
    v = _vol(P)
    obv = (np.sign(c.diff()) * v).fillna(0.0).cumsum().where(c.notna())

    def rel_pos(x):
        lo = x.rolling(win, min_periods=win).min()
        hi = x.rolling(win, min_periods=win).max()
        return (x - lo) / (hi - lo).where(hi > lo)

    return rel_pos(obv) - rel_pos(c)


# --------------------------------------------------------- 4. money flow
def chaikin_flow(P: dict, win: int = 63) -> pd.DataFrame:
    """Chaikin-style accumulation/distribution over `win` days.

    sum(CLV * volume) / sum(volume); CLV = ((c-l)-(h-c))/(h-l). Bounded
    [-1, 1].
    """
    v = _vol(P)
    mfv = _clv(P) * v
    num = mfv.rolling(win, min_periods=win).sum()
    den = v.rolling(win, min_periods=win).sum()
    return num / den.where(den > 0)


# ------------------------------------------- 5. volume dry-up in uptrend
def dryup_uptrend(P: dict, dry_win: int = 10, base_win: int = 126,
                  near_high: float = 0.95) -> pd.DataFrame:
    """Supply exhaustion: low recent volume vs 6m average while price holds
    near trailing highs, gated on an uptrend (close > 100d SMA).

    Among qualifying names, lower recent relative volume scores higher.
    Non-qualifying names get a deeply penalized closeness-to-high fallback
    so the engine can still fill k slots on thin dates.
    """
    c = P["close"]
    v = _vol(P)
    dry = (v.rolling(dry_win, min_periods=dry_win).mean()
           / v.rolling(base_win, min_periods=base_win).mean())
    near = c / c.rolling(base_win, min_periods=base_win).max()
    uptrend = c > c.rolling(100, min_periods=100).mean()
    gate = (near >= near_high) & uptrend & dry.notna()
    # rank dryness only within the gated subset
    dry_rank = dry.where(gate).rank(axis=1, pct=True)
    score = (2.0 - dry_rank).where(gate)         # in (1, 2]: drier = higher
    fallback = _xrank(near) - 10.0               # always < any gated score
    return score.fillna(fallback).where(c.notna())


# -------------------------------------------- 6. big-money footprint count
def footprints(P: dict, win: int = 63, mult: float = 2.0,
               base_win: int = 120) -> pd.DataFrame:
    """Net accumulation-day count: days in last `win` with volume > mult x
    `base_win`-day average AND close in top third of the daily range, minus
    distribution days (same volume spike, close in bottom third).
    """
    c = P["close"]
    v = _vol(P)
    base = v.rolling(base_win, min_periods=base_win).mean()
    big = v > mult * base
    pos = (c - P["low"]) / (P["high"] - P["low"]).where(P["high"] > P["low"])
    acc = (big & (pos >= 2 / 3)).astype(float).where(c.notna() & base.notna())
    dst = (big & (pos <= 1 / 3)).astype(float).where(c.notna() & base.notna())
    return (acc.rolling(win, min_periods=win).sum()
            - dst.rolling(win, min_periods=win).sum())


# ------------------------------------------- 7. interactions with momentum
def accum_x_momentum(P: dict, accum: str = "footprints",
                     mom_win: int = 126) -> pd.DataFrame:
    """rank(accumulation signal) * rank(6m momentum)."""
    builders = {
        "footprints": lambda: footprints(P),
        "chaikin": lambda: chaikin_flow(P, 63),
        "updown": lambda: updown_ratio(P, 63),
        "obv_div": lambda: obv_divergence(P),
        "hv": lambda: hv_premium(P, mode="interact"),
    }
    a = builders[accum]()
    mom = _ret(P["close"], mom_win)
    return _xrank(a) * _xrank(mom)


def accum_plus_momentum(P: dict, accum: str = "chaikin",
                        mom_win: int = 126) -> pd.DataFrame:
    """rank(accumulation) + rank(6m momentum) (additive blend)."""
    builders = {
        "footprints": lambda: footprints(P),
        "chaikin": lambda: chaikin_flow(P, 63),
        "updown": lambda: updown_ratio(P, 63),
        "obv_div": lambda: obv_divergence(P),
    }
    return _xrank(builders[accum]()) + _xrank(_ret(P["close"], mom_win))


# ----------------------- momentum-dominant blends (volume as tilt / veto)
def _accum_builder(P: dict, accum: str) -> pd.DataFrame:
    return {
        "footprints": lambda: footprints(P, 63, 1.5),
        "chaikin": lambda: chaikin_flow(P, 63),
        "updown": lambda: updown_ratio(P, 63),
        "obv_div": lambda: obv_divergence(P),
        "hv": lambda: hv_premium(P, mode="interact"),
    }[accum]()


def mom_tilt(P: dict, accum: str = "updown", w: float = 0.25,
             mom_win: int = 126) -> pd.DataFrame:
    """rank(6m momentum) + w * rank(accumulation): volume as a tiebreaker."""
    return (_xrank(_ret(P["close"], mom_win))
            + w * _xrank(_accum_builder(P, accum)))


def mom_veto(P: dict, accum: str = "updown",
             mom_win: int = 126) -> pd.DataFrame:
    """6m momentum, but names under net distribution (accum <= 0) are
    pushed below all net-accumulated names (volume as a veto)."""
    a = _accum_builder(P, accum)
    mom = _xrank(_ret(P["close"], mom_win))
    return mom.where(a > 0, mom - 10.0).where(a.notna())


def mom_gate_accum(P: dict, accum: str = "updown", q: float = 0.8,
                   mom_win: int = 126) -> pd.DataFrame:
    """Among the top-(1-q) momentum names, rank by accumulation; everything
    else falls back to a penalized momentum rank."""
    mom = _xrank(_ret(P["close"], mom_win))
    a = _accum_builder(P, accum)
    gated = a.where(mom >= q).rank(axis=1, pct=True)
    return (1.0 + gated).where(mom >= q).fillna(mom - 10.0).where(mom.notna())
