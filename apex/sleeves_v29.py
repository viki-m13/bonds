"""APEX v29 — more novel methods: Savitzky-Golay, skew-signal, OU mean-reversion.

  SL_SAVGOL          — Savitzky-Golay polynomial smoother + trend
  SL_SKEW_MOM        — Skewness-signed momentum: high skew → right tail → long
  SL_OU_MEAN_REV     — Ornstein-Uhlenbeck mean reversion on residuals
  SL_RSI_DIVERGENCE  — Bullish divergence: price new low, RSI higher low
  SL_RANGE_EXPANSION — When 5d range << 60d range (quiet before storm)
  SL_TRENDLINE_BREAK — Linear regression breakout
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew
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
# SAVITZKY-GOLAY TREND (polynomial smoother)
# ========================================================================

def _savgol_trend_causal(x, window_length=21, polyorder=3):
    """CAUSAL Savitzky-Golay: only use past data (no lookahead bias).

    For each time t, fit polynomial on [t-window+1, t] and evaluate at t.
    Returns (smoothed, 1st derivative) series.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    smoothed = np.full(n, np.nan)
    deriv = np.full(n, np.nan)
    for i in range(window_length - 1, n):
        window = x[i - window_length + 1 : i + 1]
        # Handle NaNs: replace with fwd-fill within window
        mask = ~np.isnan(window)
        if mask.sum() < polyorder + 2:
            continue
        # Fit polynomial on valid points
        t = np.arange(window_length)
        valid_t = t[mask]
        valid_y = window[mask]
        try:
            coefs = np.polyfit(valid_t, valid_y, polyorder)
            poly = np.poly1d(coefs)
            smoothed[i] = poly(window_length - 1)   # evaluate at t
            # Derivative at t
            dpoly = poly.deriv()
            deriv[i] = dpoly(window_length - 1)
        except Exception:
            continue
    return smoothed, deriv


def sleeve_savgol(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Long LETF when SavGol smoother is rising (positive 1st derivative)
    AND price is above smoothed trend line."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    pairs = {"SPY": "UPRO", "QQQ": "TQQQ", "GLD": "UGL", "TLT": "TMF"}
    for under, letf in pairs.items():
        if under not in cp.columns or letf not in cp.columns:
            continue
        p = cp[under].values
        smoothed, deriv = _savgol_trend_causal(p, window_length=21, polyorder=3)
        s_series = pd.Series(smoothed, index=idx)
        d_series = pd.Series(deriv, index=idx)
        # Signal: derivative rising AND price close to smoothed
        signal = ((d_series > 0) & (cp[under] > s_series * 0.99)).astype(float).shift(1).fillna(0)
        W[letf] = signal * 0.25

    s = W.sum(axis=1).clip(upper=1.0)
    scale = (s / W.sum(axis=1).replace(0, np.nan)).fillna(1.0).clip(upper=1.0)
    W = W.mul(scale, axis=0)
    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# SKEWNESS-SIGNED MOMENTUM
# ========================================================================

def sleeve_skew_mom(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """High positive skewness = right-tail gains dominant = persistent strong trend.
    Long top-2 LETFs by (momentum × skewness) score when skew is positive."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","UGL","UCO"]
                if a in cp.columns]
    p = cp[universe]
    r = p.pct_change()
    skew_63 = r.rolling(63).skew()
    mom_63 = p.pct_change(63)
    score = mom_63 * np.sign(skew_63) * (skew_63.abs() + 0.1)

    # Top-2 when skew positive AND momentum positive
    valid = (skew_63 > 0) & (mom_63 > 0)
    score_masked = score.where(valid)
    rnk = score_masked.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2).fillna(False)

    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % 10 == 0
    sel_wk = sel.where(is_rebal).ffill().fillna(False)
    spy_ok = (cp["SPY"] > cp["SPY"].rolling(200).mean()).astype(float)

    for u in universe:
        W[u] = sel_wk[u].astype(float) / 2 * spy_ok

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# ORNSTEIN-UHLENBECK MEAN REVERSION
# ========================================================================

def sleeve_ou_mean_rev(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Ornstein-Uhlenbeck: z-score from long-term mean. Very negative z → buy.
    Parametrize: fitted θ (mean-rev speed) × (price - mu)."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "SPY" not in cp.columns or "UPRO" not in cp.columns:
        return W

    # OU residual: price - long MA, normalized by rolling std
    spy = cp["SPY"]
    log_spy = np.log(spy)
    mu = log_spy.rolling(252, min_periods=60).mean()
    sigma = log_spy.rolling(252, min_periods=60).std()
    z = (log_spy - mu) / sigma

    # Very negative z (>2σ below mean) in uptrend = mean-rev buy
    ma_up = (spy > spy.rolling(500).mean()).astype(float)
    z_low = (z < -1.5).astype(float)

    held = z_low.rolling(10, min_periods=1).sum().clip(upper=1.0).shift(1).fillna(0)
    W["UPRO"] = held * ma_up * 0.5

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# TRENDLINE LINEAR REGRESSION BREAKOUT
# ========================================================================

def sleeve_trendline_break(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Fit linear regression on past 60d. Breakout = price > (regression_last + 2σ_residual)."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    def _regression_breakout(p, lookback=60):
        p = np.asarray(p)
        n = len(p)
        if n < lookback:
            return np.nan
        window = p[-lookback:]
        x = np.arange(lookback)
        slope, intercept = np.polyfit(x, window, 1)
        fitted = intercept + slope * x
        resid = window - fitted
        sigma = np.std(resid)
        upper = fitted[-1] + 2 * sigma
        # Return 1 if current > upper
        return 1.0 if window[-1] > upper and slope > 0 else 0.0

    pairs = {"SPY": "UPRO", "QQQ": "TQQQ"}
    for under, letf in pairs.items():
        if under not in cp.columns or letf not in cp.columns:
            continue
        p = cp[under]
        signal = p.rolling(60).apply(_regression_breakout, raw=True)
        held = signal.rolling(10, min_periods=1).sum().clip(upper=1.0)
        W[letf] = held.shift(1).fillna(0) * 0.4

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# RANGE COMPRESSION (quiet before storm)
# ========================================================================

def sleeve_range_compress(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """When 5d range is much less than 60d range → coiling for breakout.
    Enter on the break of the 10d high/low, direction TBD."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    spy = cp["SPY"]
    r5 = spy.rolling(5).max() - spy.rolling(5).min()
    r60 = spy.rolling(60).max() - spy.rolling(60).min()
    compression = r5 / r60

    # Very compressed + already trending up → momentum breakout
    compressed = (compression < 0.15).astype(float)
    above_ma = (spy > spy.rolling(50).mean()).astype(float)
    # Break: price > 10d high
    at_breakout = (spy >= spy.rolling(10).max() * 0.995).astype(float)

    signal = (compressed * above_ma * at_breakout).shift(1).fillna(0)
    held = signal.rolling(5, min_periods=1).sum().clip(upper=1.0)
    if "UPRO" in cp.columns:
        W["UPRO"] = held * 0.6

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ========================================================================
# SHARP RATIO REGIME
# ========================================================================

def sleeve_sharpe_regime(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """When SPY rolling 60d Sharpe is high AND improving, risk on.
    When low and declining, risk off."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    spy_r = cp["SPY"].pct_change()
    rolling_sr_60 = spy_r.rolling(60).mean() / spy_r.rolling(60).std() * np.sqrt(util.DPY)
    rolling_sr_21 = spy_r.rolling(21).mean() / spy_r.rolling(21).std() * np.sqrt(util.DPY)

    # Good regime: 60d SR > 1 AND 21d SR > 60d SR (improving)
    good = ((rolling_sr_60 > 1.0) & (rolling_sr_21 > rolling_sr_60)).astype(float).shift(1).fillna(0)
    # Bad regime: 60d SR < 0 AND 21d SR < 60d SR (deteriorating)
    bad = ((rolling_sr_60 < 0) & (rolling_sr_21 < rolling_sr_60)).astype(float).shift(1).fillna(0)

    if "UPRO" in cp.columns:
        W["UPRO"] = good * 0.5
    if "UGL" in cp.columns:
        W["UGL"] = bad * 0.3
    if "TMF" in cp.columns:
        W["TMF"] = bad * 0.2

    return _scale_to_vol(W, cp, target_vol=target_vol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()

    sleeves = {
        "SAVGOL":           sleeve_savgol(cp),
        "SKEW_MOM":         sleeve_skew_mom(cp),
        "OU_MEAN_REV":      sleeve_ou_mean_rev(cp),
        "TRENDLINE_BRK":    sleeve_trendline_break(cp),
        "RANGE_COMPRESS":   sleeve_range_compress(cp),
        "SHARPE_REGIME":    sleeve_sharpe_regime(cp),
    }
    print(f"{'Sleeve':18s}  {'SR':>5}  {'CAGR':>7}  {'MDD':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sleeves.items():
        r = _weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:18s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")
