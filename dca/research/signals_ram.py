"""Risk-adjusted momentum / volatility-structure signal builders.

Each builder takes the panels dict from data.build_panel() and returns a
scores DataFrame (dates x tickers).  Every computation is TRAILING-only:
rolling windows ending at row date d, cross-sectional stats within row d.
NaN score = ineligible at that date.
"""
import numpy as np
import pandas as pd

import data as data_mod


# ---------------------------------------------------------------- helpers

def _rets(P):
    return P["close"].pct_change()


def _spy_rets(P):
    spy = data_mod.load_benchmark("SPY")["Close"].pct_change()
    return spy.reindex(P["close"].index)


def _xrank(df, member):
    """Cross-sectional percentile rank within row d, members only."""
    return df.where(member).rank(axis=1, pct=True)


# ---------------------------------------------------------------- 1. Sharpe / Sortino momentum

def sharpe_mom(P, window=126):
    r = _rets(P)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std()
    return mu / sd


def sortino_mom(P, window=126):
    """Mean return / downside deviation (sqrt of mean squared neg return)."""
    r = _rets(P)
    mu = r.rolling(window).mean()
    dd = np.sqrt((r.clip(upper=0.0) ** 2).rolling(window).mean())
    return mu / dd.replace(0.0, np.nan)


# ---------------------------------------------------------------- 2. Vol-scaled momentum

def volscaled_mom(P, mom_window=126, vol_window=126):
    """Compound trailing return divided by realized vol over vol_window."""
    r = _rets(P)
    mom = P["close"] / P["close"].shift(mom_window) - 1.0
    vol = r.rolling(vol_window).std() * np.sqrt(252)
    return mom / vol.replace(0.0, np.nan)


# ---------------------------------------------------------------- 3. Residual (idiosyncratic) momentum

def residual_mom(P, beta_window=252, mom_window=126, scaled=False):
    """Rolling-beta (cov/var vs SPY, trailing beta_window) residual returns,
    cumulated over the last mom_window days.  scaled=True divides by the
    rolling std of residuals (idiosyncratic Sharpe)."""
    r = _rets(P)
    m = _spy_rets(P)
    r_mu = r.rolling(beta_window).mean()
    m_mu = m.rolling(beta_window).mean()
    cov = r.mul(m, axis=0).rolling(beta_window).mean() - r_mu.mul(m_mu, axis=0)
    var = (m ** 2).rolling(beta_window).mean() - m_mu ** 2
    beta = cov.div(var.replace(0.0, np.nan), axis=0)
    resid = r.sub(beta.mul(m, axis=0))
    score = resid.rolling(mom_window).sum()
    if scaled:
        score = score / (resid.rolling(mom_window).std()
                         * np.sqrt(mom_window)).replace(0.0, np.nan)
    return score


# ---------------------------------------------------------------- 4. Low-vol gate x momentum

def lowvol_gate_mom(P, mom_window=126, vol_window=126):
    """Momentum, but only among the lowest-vol half of the member universe
    at each date (others NaN -> never picked)."""
    r = _rets(P)
    mom = P["close"] / P["close"].shift(mom_window) - 1.0
    vol = r.rolling(vol_window).std().where(P["member"])
    med = vol.median(axis=1)
    gate = vol.le(med, axis=0)
    return mom.where(gate)


def mom_minus_volrank(P, mom_window=126, vol_window=126, penalty=0.5):
    """Momentum percentile rank minus penalty * vol percentile rank."""
    r = _rets(P)
    mom = P["close"] / P["close"].shift(mom_window) - 1.0
    vol = r.rolling(vol_window).std()
    return _xrank(mom, P["member"]) - penalty * _xrank(vol, P["member"])


# ---------------------------------------------------------------- 5. Anti-lottery (MAX effect) filter

def antilottery_mom(P, mom_window=126, max_window=21, cut=0.9):
    """Exclude names whose max single-day return over the last max_window
    days is in the top cross-sectional decile; rank survivors by momentum."""
    r = _rets(P)
    mx = r.rolling(max_window).max().where(P["member"])
    thr = mx.quantile(cut, axis=1)
    keep = mx.lt(thr, axis=0)
    mom = P["close"] / P["close"].shift(mom_window) - 1.0
    return mom.where(keep)


def plain_mom(P, mom_window=126):
    """Unfiltered momentum control (for with/without comparisons)."""
    return P["close"] / P["close"].shift(mom_window) - 1.0


# ---------------------------------------------------------------- 6. Smoothness / path quality

def trend_r2_mom(P, window=126):
    """R^2 of log-price vs time over the trailing window, signed by the
    direction of the move: smooth uptrend -> high positive score."""
    logp = np.log(P["close"])
    t = pd.Series(np.arange(len(logp)), index=logp.index, dtype=float)
    corr = logp.rolling(window).corr(t)
    mom = P["close"] / P["close"].shift(window) - 1.0
    return corr ** 2 * np.sign(mom)


def path_quality(P, window=126):
    """Net trailing return divided by total absolute daily movement:
    +1 would be a perfectly monotone rise."""
    r = _rets(P)
    mom = P["close"] / P["close"].shift(window) - 1.0
    tot = r.abs().rolling(window).sum()
    return mom / tot.replace(0.0, np.nan)


# ---------------------------------------------------------------- 7. Skewness x momentum

def skew_mom(P, mom_window=126, skew_window=126, weight=0.5, prefer_negative=True):
    """Momentum rank +/- weight * skew rank.  prefer_negative=True rewards
    negatively skewed winners (literature direction); False the opposite."""
    r = _rets(P)
    sk = r.rolling(skew_window).skew()
    mr = _xrank(P["close"] / P["close"].shift(mom_window) - 1.0, P["member"])
    sr = _xrank(sk, P["member"])
    return mr - weight * sr if prefer_negative else mr + weight * sr


# ---------------------------------------------------------------- combos (built after screening)

def antilottery_sharpe(P, window=126, max_window=21, cut=0.9):
    """MAX-effect filter applied to Sharpe momentum."""
    r = _rets(P)
    mx = r.rolling(max_window).max().where(P["member"])
    thr = mx.quantile(cut, axis=1)
    keep = mx.lt(thr, axis=0)
    return sharpe_mom(P, window).where(keep)


def lowvol_gate_sharpe(P, window=126, vol_window=126):
    """Low-vol half gate applied to Sharpe momentum."""
    r = _rets(P)
    vol = r.rolling(vol_window).std().where(P["member"])
    med = vol.median(axis=1)
    gate = vol.le(med, axis=0)
    return sharpe_mom(P, window).where(gate)
