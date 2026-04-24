"""Crypto APEX sleeves — applying APEX methodology to 20-coin universe.

Six uncorrelated sleeves mirroring APEX v33:
  1. MOMENTUM    — 6-month momentum (skip 21d), top-3, inverse-vol weighted
  2. ACCEL       — 2nd-derivative: mom_5 > mom_21 > mom_63 all positive
  3. SKEW_MOM    — momentum × sign(skew) × |skew|, top-3
  4. HMM         — BTC-regime HMM, long BTC+ETH in bull state
  5. HURST       — fractal regime: H > 0.55 trend, H < 0.45 mean-revert
  6. DOMINANCE   — BTC dominance regime (BTC strength vs alts)

All sleeves: coin only included when it has >=252d history AT THAT DATE (prevents
forward-peeking). Dead coins (LUNA1, FTT, USTC, MATIC, UNI) naturally exit
when their data ends — NO forward fill of delisted prices.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from util import DPY, load_prices, load_macro, SURVIVORS, DEAD, ALL_COINS


def _eligibility(cp: pd.DataFrame, min_history: int = 252,
                  catastrophe_dd: float = -0.50) -> pd.DataFrame:
    """Eligibility mask: True when coin has at least `min_history` days of past data
    AND is not in catastrophic drawdown (>= `catastrophe_dd` from 90-day high).

    Critically: once a coin's data ends (delisted / dead), mark ineligible —
    this is how we honor survivorship without look-ahead. And once a coin crashes
    beyond a realistic stop-loss threshold, it's out until rebuild.
    """
    has_data = cp.notna()
    age = has_data.cumsum()
    last_valid = cp.apply(lambda s: s.last_valid_index())
    dead_at = pd.Series(last_valid, index=cp.columns)
    mask = (age >= min_history).astype(float)
    for c in cp.columns:
        lv = dead_at[c]
        if pd.notna(lv):
            mask.loc[mask.index > lv, c] = 0.0

    # Catastrophic drawdown filter: coin at >50% DD from 90d high → ineligible
    hwm90 = cp.rolling(90, min_periods=30).max()
    dd = cp / hwm90 - 1
    alive_risk = (dd > catastrophe_dd).astype(float)
    mask = mask * alive_risk.fillna(0.0)
    return mask


def _inverse_vol_weight(ranks: pd.DataFrame, vol: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Given ranked scores, keep top_n, weight by inverse 60d vol."""
    W = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    for dt in ranks.index:
        row = ranks.loc[dt].dropna().sort_values(ascending=False)
        picked = row.head(top_n).index.tolist()
        if not picked:
            continue
        iv = 1.0 / vol.loc[dt, picked].replace(0, np.nan).fillna(1.0)
        iv = iv / iv.sum()
        W.loc[dt, picked] = iv.values
    return W


def sleeve_momentum(cp: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """6-month momentum (skip 21d) top-2 inverse-vol, strong BTC-regime gate."""
    elig = _eligibility(cp, 180)  # 9 months of history required
    mom_180 = cp.pct_change(180).shift(21)
    mom_180 = mom_180.where(elig.astype(bool))
    # Require momentum > 20% (no noise chasing)
    mom_180 = mom_180.where(mom_180 > 0.20)
    rv = cp.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)

    W = _inverse_vol_weight(mom_180, rv, top_n=2)

    # STRICT BTC gate: above 200MA AND 63d mom > +5%
    btc = cp["BTC"]
    btc_above = (btc > btc.rolling(200).mean()).astype(float)
    btc_mom_ok = (btc.pct_change(63) > 0.05).astype(float)
    btc_gate = (btc_above * btc_mom_ok).shift(1).fillna(0.0)
    W = W.mul(btc_gate, axis=0)
    return W


def sleeve_accel(cp: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """2nd-derivative momentum: mom_5 > mom_21 > mom_63, all > 0, w/ BTC gate."""
    elig = _eligibility(cp, 63)
    m5 = cp.pct_change(5)
    m21 = cp.pct_change(21)
    m63 = cp.pct_change(63)
    accel = ((m5 > m21) & (m21 > m63) & (m5 > 0.02) & (m21 > 0.05)).astype(float).where(elig.astype(bool))
    score = accel * m21.clip(lower=0)
    rv = cp.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)
    W = _inverse_vol_weight(score, rv, top_n=2)

    btc = cp["BTC"]
    btc_gate = (btc > btc.rolling(100).mean()).astype(float).shift(1).fillna(0.0)
    W = W.mul(btc_gate, axis=0)
    return W


def sleeve_skew_mom(cp: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """Skewness-signed momentum w/ BTC gate."""
    elig = _eligibility(cp, 126)
    mom = cp.pct_change(63)
    r = cp.pct_change()
    skew = r.rolling(63, min_periods=30).skew()
    score = (mom * np.sign(skew) * skew.abs().clip(upper=2.0)).where(elig.astype(bool))
    score = score.where(score > 0.05)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(DPY)
    W = _inverse_vol_weight(score, rv, top_n=2)

    btc = cp["BTC"]
    btc_gate = (btc > btc.rolling(100).mean()).astype(float).shift(1).fillna(0.0)
    W = W.mul(btc_gate, axis=0)
    return W


def sleeve_hmm(cp: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """BTC HMM regime: long basket in bull state."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        # Fallback: simple BTC-200MA + 63d-mom regime
        btc = cp["BTC"]
        bull = ((btc > btc.rolling(200).mean()) & (btc.pct_change(63) > 0)).astype(float)
        W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
        for c in ["BTC", "ETH"]:
            if c in cp.columns:
                W[c] = bull / 2
        return W.shift(1).fillna(0.0)

    btc_r_full = cp["BTC"].pct_change().dropna()
    if len(btc_r_full) < 500:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    btc_r = btc_r_full.values.reshape(-1, 1)
    # Train on first 60% (IS)
    split = int(len(btc_r) * 0.6)
    train = btc_r[:split]
    model = GaussianHMM(n_components=3, covariance_type="full",
                        n_iter=100, random_state=42)
    model.fit(train)
    states = model.predict(btc_r)
    means = [model.means_[i][0] for i in range(3)]
    bull_state = int(np.argmax(means))
    regime = pd.Series(states, index=btc_r_full.index).reindex(cp.index).ffill()
    bull = (regime == bull_state).astype(float)

    # Long BTC + ETH equal-weighted when bull
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in ["BTC", "ETH"]:
        if c in cp.columns:
            W[c] = 0.5 * bull
    return W.shift(1).fillna(0.0)


def sleeve_hurst(cp: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """BTC Hurst exponent regime."""
    btc = cp["BTC"].dropna()
    r = btc.pct_change().fillna(0.0)
    window = 100
    H = pd.Series(np.nan, index=btc.index)
    for i in range(window, len(r)):
        series = r.iloc[i-window:i].values
        if np.std(series) == 0:
            continue
        mean = np.mean(series)
        Y = np.cumsum(series - mean)
        R = np.max(Y) - np.min(Y)
        S = np.std(series)
        if S == 0 or R == 0:
            continue
        h = np.log(R / S) / np.log(window)
        H.iloc[i] = h
    H = H.reindex(cp.index).ffill()

    btc_mom = cp["BTC"].pct_change(21)
    trend_regime = (H > 0.55).astype(float)
    revert_regime = (H < 0.45).astype(float)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    # Trend regime: long BTC when mom > 0
    W["BTC"] = trend_regime * (btc_mom > 0).astype(float)
    # Mean-revert regime: long BTC when 5d return negative (buy dip)
    dip = (cp["BTC"].pct_change(5) < -0.05).astype(float)
    W["BTC"] = W["BTC"] + revert_regime * dip * 0.5
    return W.shift(1).fillna(0.0)


def sleeve_dominance(cp: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """BTC dominance regime — when BTC outperforms alts (alt bear), concentrate in BTC;
    when alts outperform BTC (alt season), spread wider."""
    if "BTC" not in cp.columns:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    btc_r21 = cp["BTC"].pct_change(21)
    alt_cols = [c for c in cp.columns if c not in ["BTC"]]
    alt_r21 = cp[alt_cols].pct_change(21).mean(axis=1)
    btc_rel = btc_r21 - alt_r21  # positive: BTC winning, negative: alts winning

    btc_above_200 = (cp["BTC"] > cp["BTC"].rolling(200).mean()).astype(float)

    # Alt-season: concentrate in top 3 alts by 21d mom
    mom21 = cp[alt_cols].pct_change(21)
    elig = _eligibility(cp[alt_cols], 63)
    score = mom21.where(elig.astype(bool)).where(mom21 > 0)
    rv = cp.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)[alt_cols] if False else \
         cp[alt_cols].pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)

    W_alt = _inverse_vol_weight(score, rv, top_n=3)
    # Extend to full columns
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in W_alt.columns:
        W[c] = W_alt[c]

    # BTC-season: go BTC only
    W_btc = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W_btc["BTC"] = 1.0

    alt_season = (btc_rel < 0).astype(float)
    btc_season = 1 - alt_season
    # Blend and gate
    Wout = W.mul(alt_season, axis=0) + W_btc.mul(btc_season, axis=0)
    Wout = Wout.mul(btc_above_200.shift(1).fillna(0.0), axis=0)
    return Wout


def sleeve_breakout(cp: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """90-day Donchian breakout: long when new 90d high made, exit at 30d low."""
    elig = _eligibility(cp, 100)
    roll_max_90 = cp.rolling(90, min_periods=50).max()
    roll_min_30 = cp.rolling(30, min_periods=15).min()
    # Breakout signal: price within 2% of 90d high (scan, not just single day)
    at_high = (cp >= 0.98 * roll_max_90).astype(float)
    # Exit: price breaks 30d low
    stop = (cp <= 1.02 * roll_min_30).astype(float)
    # State machine: hold while breakout continues & not stopped
    sig = at_high.where(elig.astype(bool)).fillna(0.0)
    # Simplified: long while at_high=1 AND not below 30d low recently
    signal = sig - stop.shift(-1).fillna(0.0)
    signal = signal.clip(lower=0, upper=1)
    # Rolling — if broke out in last 10 days AND still above 30d low
    recent_breakout = at_high.rolling(10, min_periods=1).max().fillna(0.0)
    above_stop = (cp > roll_min_30).astype(float)
    score = (recent_breakout * above_stop).where(elig.astype(bool)).fillna(0.0)
    mom21 = cp.pct_change(21)
    score = score * mom21.clip(lower=0)
    rv = cp.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)
    W = _inverse_vol_weight(score, rv, top_n=2)

    btc = cp["BTC"]
    btc_gate = (btc > btc.rolling(200).mean()).astype(float).shift(1).fillna(0.0)
    W = W.mul(btc_gate, axis=0)
    return W


def sleeve_rsi_dip(cp: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """Mean-reversion buy-the-dip within STRONG uptrend."""
    elig = _eligibility(cp, 100)
    r = cp.pct_change()
    # 14-day RSI
    gain = r.clip(lower=0).rolling(14, min_periods=7).mean()
    loss = (-r).clip(lower=0).rolling(14, min_periods=7).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # Strong uptrend: above 200MA AND 63d mom > 30%
    strong_up = ((cp > cp.rolling(200, min_periods=100).mean()) &
                 (cp.pct_change(63) > 0.30)).astype(float)
    # Dip: RSI < 35
    dip = (rsi < 35).astype(float)
    score = (strong_up * dip).where(elig.astype(bool)).fillna(0.0)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(DPY)
    W = _inverse_vol_weight(score, rv, top_n=2)

    btc = cp["BTC"]
    btc_gate = (btc > btc.rolling(200).mean()).astype(float).shift(1).fillna(0.0)
    W = W.mul(btc_gate, axis=0)
    return W


BUILDERS = {
    "MOMENTUM":  sleeve_momentum,
    "ACCEL":     sleeve_accel,
    "SKEW_MOM":  sleeve_skew_mom,
    "HMM":       sleeve_hmm,
    "HURST":     sleeve_hurst,
    "DOMINANCE": sleeve_dominance,
    "BREAKOUT":  sleeve_breakout,
    "RSI_DIP":   sleeve_rsi_dip,
}


def build_all(cp: pd.DataFrame, macro: dict) -> dict:
    out = {}
    for name, fn in BUILDERS.items():
        W = fn(cp, macro)
        out[name] = W.fillna(0.0)
    return out
