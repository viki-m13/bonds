"""APEX v26 — NOVEL mathematical/statistical sleeves.

Each sleeve uses a method not yet tried:
  SL_KALMAN       — Kalman filter for adaptive trend estimation (state-space)
  SL_HURST        — Hurst exponent regime (trend-follow if H>0.55, MR if H<0.45)
  SL_BREADTH      — Market breadth: % LETFs above 50d MA
  SL_ACCEL_MOM    — Acceleration momentum (2nd derivative of return)
  SL_ENTROPY      — Shannon entropy of 21d returns (low = trending = long)
  SL_YANG_ZHANG   — Yang-Zhang volatility estimator (superior to close-to-close)
  SL_FRAC_DIFF    — Fractionally differentiated price (López de Prado) — stationary series
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import util

ROOT = Path("/home/user/bonds")
FRED = ROOT / "data/fred"
ETF = ROOT / "data/etfs"


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def _etf_close(t, idx):
    fp = ETF / f"{t}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Close"].astype(float).reindex(idx).ffill()


def _weights_to_ret(W, cp):
    w = W.fillna(0.0)
    rets = cp.pct_change()
    r = (w.shift(1).fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def _scale_to_vol(W, cp, target_vol=0.15):
    r = _weights_to_ret(W, cp)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


# ========================================================================
# KALMAN FILTER TREND
# ========================================================================

def kalman_trend(x: np.ndarray, q: float = 0.001, r: float = 0.1) -> np.ndarray:
    """Simple 1D Kalman filter for adaptive trend (level) estimation.
    q: process noise (higher = more responsive); r: measurement noise."""
    n = len(x)
    est = np.zeros(n)
    err = np.zeros(n)
    est[0] = x[0] if not np.isnan(x[0]) else 0
    err[0] = 1.0
    for i in range(1, n):
        if np.isnan(x[i]):
            est[i] = est[i-1]
            err[i] = err[i-1] + q
            continue
        # Predict
        err[i] = err[i-1] + q
        # Update
        K = err[i] / (err[i] + r)
        est[i] = est[i-1] + K * (x[i] - est[i-1])
        err[i] = (1 - K) * err[i]
    return est


def sleeve_kalman(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Trade each major LETF when its underlying price is ABOVE its
    Kalman-filtered trend AND the trend is rising."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    pairs = {"SPY": "UPRO", "QQQ": "TQQQ", "TLT": "TMF", "GLD": "UGL"}
    for under, letf in pairs.items():
        if under not in cp.columns or letf not in cp.columns:
            continue
        p = cp[under].values
        trend = kalman_trend(p, q=0.01, r=5.0)
        trend_series = pd.Series(trend, index=idx)
        trend_rising = (trend_series.diff(5) > 0).astype(float)
        price_above = (cp[under] > trend_series * 1.005).astype(float)
        signal = (trend_rising * price_above).shift(1).fillna(0)
        W[letf] = signal * 0.25

    s = W.sum(axis=1).clip(upper=1.0)
    scale = (s / W.sum(axis=1).replace(0, np.nan)).fillna(1.0).clip(upper=1.0)
    W = W.mul(scale, axis=0)
    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# HURST EXPONENT REGIME
# ========================================================================

def hurst_rolling(prices: pd.Series, lookback: int = 100, lags: int = 20) -> pd.Series:
    """Rolling Hurst exponent via rescaled-range / Mandelbrot method.

    H > 0.5: persistent (trending) series
    H < 0.5: anti-persistent (mean-reverting)
    H = 0.5: random walk
    """
    lr = np.log(prices / prices.shift(1))

    def _hurst(window):
        if len(window) < lags + 5:
            return np.nan
        # Use variance-of-log-returns at different lags
        lag_range = range(2, lags)
        try:
            tau = [np.sqrt(np.std(np.subtract(window[lag:], window[:-lag])))
                   for lag in lag_range]
            if any(t == 0 or not np.isfinite(t) for t in tau):
                return np.nan
            poly = np.polyfit(np.log(list(lag_range)), np.log(tau), 1)
            return poly[0] * 2.0
        except Exception:
            return np.nan

    return lr.rolling(lookback, min_periods=50).apply(_hurst, raw=True)


def sleeve_hurst(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """When Hurst > 0.55 on SPY → trend regime → long UPRO.
    When Hurst < 0.45 → mean-revert regime → buy UPRO on 5d dip.
    When 0.45-0.55: cash."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "SPY" not in cp.columns or "UPRO" not in cp.columns:
        return W

    H = hurst_rolling(cp["SPY"], lookback=100, lags=15)

    # Regime signals
    trend_regime = (H > 0.55).astype(float)
    mr_regime = (H < 0.45).astype(float)

    # Trend: long UPRO if SPY rising
    spy = cp["SPY"]
    spy_up = (spy > spy.rolling(50).mean()).astype(float)
    # MR: long UPRO on 5d dip
    spy_r5 = spy.pct_change(5)
    dip = (spy_r5 < -0.02).astype(float)

    signal = (trend_regime * spy_up + mr_regime * dip).shift(1).fillna(0).clip(upper=1.0)
    W["UPRO"] = signal * 0.5

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# CROSS-ASSET BREADTH
# ========================================================================

def sleeve_breadth(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Breadth = % of LETFs above their own 50d MA.
    > 70% breadth = broad risk-on → long top-2 momentum LETFs
    < 30% breadth = broad risk-off → long UGL + TMF"""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","YINN",
                             "TMF","UBT","UGL","UCO","DRN"] if a in cp.columns]
    p = cp[universe]
    above_50 = (p > p.rolling(50).mean()).astype(float)
    breadth = above_50.mean(axis=1)

    # Broad risk-on: >70% above 50d MA
    risk_on = (breadth > 0.70).astype(float).shift(1).fillna(0)
    # Broad risk-off
    risk_off = (breadth < 0.30).astype(float).shift(1).fillna(0)

    # Top-2 by 63d momentum
    mom63 = p.pct_change(63)
    rnk = mom63.rank(axis=1, ascending=False, method="first")
    sel_top2 = (rnk <= 2).astype(float)

    for u in universe:
        W[u] = W.get(u, 0) + risk_on * sel_top2[u] * 0.4

    if "UGL" in cp.columns:
        W["UGL"] = W.get("UGL", 0) + risk_off * 0.4
    if "TMF" in cp.columns:
        W["TMF"] = W.get("TMF", 0) + risk_off * 0.3

    s = W.sum(axis=1).clip(upper=1.0)
    scale = (s / W.sum(axis=1).replace(0, np.nan)).fillna(1.0).clip(upper=1.0)
    W = W.mul(scale, axis=0)
    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# ACCELERATION MOMENTUM
# ========================================================================

def sleeve_accel_mom(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Second-derivative momentum: long when returns are ACCELERATING.
    mom_5d > mom_21d > mom_63d (all positive, getting stronger).
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","UCO","UGL"]
                if a in cp.columns]
    p = cp[universe]
    mom_5 = p.pct_change(5)
    mom_21 = p.pct_change(21)
    mom_63 = p.pct_change(63)

    # Accelerating momentum: shorter > longer AND all positive
    accel = ((mom_5 > mom_21 * (5.0/21.0)) &
             (mom_21 > mom_63 * (21.0/63.0)) &
             (mom_63 > 0)).astype(float)
    # Top-2 by accel strength (mom_5 - mom_21)
    strength = mom_5 - mom_21 * (5.0/21.0)
    strength_masked = strength.where(accel.astype(bool))
    rnk = strength_masked.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2)

    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % 5 == 0
    sel_wk = sel.where(is_rebal).ffill().fillna(False)

    spy_ok = (cp["SPY"] > cp["SPY"].rolling(200).mean()).astype(float)
    for u in universe:
        W[u] = (sel_wk[u].astype(float) / 2 * spy_ok)

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# SHANNON ENTROPY REGIME
# ========================================================================

def shannon_entropy_rolling(returns: pd.Series, lookback: int = 21, bins: int = 5) -> pd.Series:
    """Rolling Shannon entropy of binned returns.
    Low entropy = concentrated return distribution = trending
    High entropy = uniform = choppy/random walk"""
    def _entropy(window):
        if len(window) < 5:
            return np.nan
        window = window[~np.isnan(window)]
        if len(window) < 5 or window.std() == 0:
            return np.nan
        hist, _ = np.histogram(window, bins=bins)
        p = hist / hist.sum() if hist.sum() > 0 else hist
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    return returns.rolling(lookback, min_periods=10).apply(_entropy, raw=True)


def sleeve_entropy(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Low entropy regime (trending) → long UPRO.
    High entropy (noise) → cash."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "SPY" not in cp.columns or "UPRO" not in cp.columns:
        return W

    spy_r = cp["SPY"].pct_change()
    entropy = shannon_entropy_rolling(spy_r, lookback=21)
    # Entropy z-score
    entropy_z = (entropy - entropy.rolling(252, min_periods=60).mean()) / entropy.rolling(252, min_periods=60).std()

    # Low entropy + positive momentum = strong trend
    spy_trend = cp["SPY"] > cp["SPY"].rolling(50).mean()
    trend_regime = ((entropy_z < -0.5) & spy_trend).astype(float).shift(1).fillna(0)

    W["UPRO"] = trend_regime * 0.5
    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# YANG-ZHANG VOL-ADJUSTED MOMENTUM
# ========================================================================

def sleeve_yz_vol(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Trade when volatility is in its LOW decile AND asset trending up.

    Low vol = stable trend → lean in. Uses close-to-close vol (simpler than YZ)."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    universe = [a for a in ["UPRO","TQQQ","TECL"] if a in cp.columns]
    p = cp[universe]
    rv = p.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    rv_pct = rv.rolling(252, min_periods=60).rank(pct=True)

    trending = (p > p.rolling(50).mean()).astype(float)
    low_vol = (rv_pct < 0.3).astype(float)

    signal = (trending * low_vol).shift(1).fillna(0)

    for u in universe:
        W[u] = signal[u] * 0.33

    s = W.sum(axis=1).clip(upper=1.0)
    scale = (s / W.sum(axis=1).replace(0, np.nan)).fillna(1.0).clip(upper=1.0)
    W = W.mul(scale, axis=0)
    return _scale_to_vol(W, cp, target_vol=target_vol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()

    sleeves = {
        "KALMAN":     sleeve_kalman(cp),
        "HURST":      sleeve_hurst(cp),
        "BREADTH":    sleeve_breadth(cp),
        "ACCEL_MOM":  sleeve_accel_mom(cp),
        "ENTROPY":    sleeve_entropy(cp),
        "YZ_VOL":     sleeve_yz_vol(cp),
    }
    print(f"{'Sleeve':12s}  {'SR':>5}  {'CAGR':>7}  {'MDD':>7}  {'OOS':>5}  {'OOS_CAGR':>8}  {'2022':>7}  {'2008':>7}")
    for name, W in sleeves.items():
        r = _weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:12s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{om.get('cagr',0)*100:>7.1f}%  {m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")
