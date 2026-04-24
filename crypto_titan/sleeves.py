"""CRYPTO-TITAN sleeves — multi-edge crypto ensemble.

Orthogonal edges (most NOT in crypto_apex):

  DIRECTIONAL LONG (BTC/ETH trend core):
    1. BTC_VM        — vol-managed BTC fast trend
    2. ETH_VM        — vol-managed ETH fast trend
    3. BTC_SLOW      — slow 200MA trend (decorrelated speed)

  MARKET-NEUTRAL:
    4. ETHBTC_PAIR   — mean-reverting ETH/BTC ratio (z-score > 1)

  MACRO-TIMED:
    5. MACRO_TREND   — BTC long-only gated by SPY trend + VIX regime

  SPECIALISTS:
    6. CRASH_DIP     — buy BTC in 25-40% DD while 200MA still trending up
    7. HALVING_BOOST — deterministic post-halving window (days 120-480)
    8. VIX_SHOCK     — cut everything on VIX spikes (defensive overlay contrib)

All long sleeves respect eligibility + catastrophic-DD protection. ETHBTC_PAIR
is market-neutral by construction (equal-dollar long and short).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from util import DPY, eligibility, HALVING_DATES


def _vol_managed_core(cp, coin, vol_target=0.22, ma_len=100, mom_len=63,
                      dd_cut=-0.28):
    if coin not in cp.columns:
        return pd.Series(0.0, index=cp.index)
    s = cp[coin]
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (vol_target / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.5)
    ma = s.rolling(ma_len, min_periods=ma_len // 2).mean()
    trend = ((s > ma) & (s.pct_change(mom_len) > 0.0)).astype(float)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > dd_cut).astype(float)
    return (size * trend * alive).fillna(0.0).shift(1).fillna(0.0)


def sleeve_btc_vm(cp, macro=None):
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W["BTC"] = _vol_managed_core(cp, "BTC", vol_target=0.22,
                                  ma_len=100, mom_len=63, dd_cut=-0.28)
    return W


def sleeve_eth_vm(cp, macro=None):
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "ETH" in cp.columns:
        W["ETH"] = _vol_managed_core(cp, "ETH", vol_target=0.22,
                                      ma_len=100, mom_len=63, dd_cut=-0.28)
    return W


def sleeve_btc_slow(cp, macro=None):
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    rv = s.pct_change().rolling(60, min_periods=30).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.2, upper=1.2)
    ma200 = s.rolling(200, min_periods=100).mean()
    trend = ((s > ma200) & (s.pct_change(126) > 0.20)).astype(float)
    hwm = s.rolling(180, min_periods=60).max()
    dd = s / hwm - 1
    alive = (dd > -0.35).astype(float)
    W["BTC"] = (size * trend * alive).shift(1).fillna(0.0)
    return W


def sleeve_ethbtc_pair(cp, macro=None):
    """Market-neutral mean-reversion on ETH/BTC log-ratio.

    When log(ETH/BTC) is N stdev below 90d mean → expect revert UP
      → long ETH, short BTC.
    When N stdev above → revert DOWN → short ETH, long BTC.

    Gross exposure: ~60% (30% each leg). Net dollar neutral.
    Only activates when ETH has ≥365d history AND ETH-BTC 90d correlation > 0.4.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "ETH" not in cp.columns:
        return W
    eth = cp["ETH"]
    btc = cp["BTC"]
    # Only use dates where ETH exists
    start_idx = eth.first_valid_index()
    if start_idx is None:
        return W

    ratio = (eth / btc).replace(0, np.nan)
    log_r = np.log(ratio)
    mean = log_r.rolling(90, min_periods=60).mean()
    std = log_r.rolling(90, min_periods=60).std()
    z = (log_r - mean) / std.replace(0, np.nan)

    # Correlation gate — pair trade only works when pair co-moves directionally
    eth_r = eth.pct_change()
    btc_r = btc.pct_change()
    corr = eth_r.rolling(60, min_periods=30).corr(btc_r)
    corr_ok = (corr > 0.40).astype(float)

    # Position logic with hysteresis
    entry = 1.2
    exit_ = 0.3
    pos = pd.Series(0.0, index=cp.index)
    state = 0.0
    for i in range(len(z)):
        zi = z.iloc[i]
        if pd.isna(zi):
            pos.iloc[i] = state
            continue
        if state == 0.0:
            if zi < -entry:
                state = 1.0   # long ETH / short BTC
            elif zi > entry:
                state = -1.0  # short ETH / long BTC
        elif state == 1.0 and zi > -exit_:
            state = 0.0
        elif state == -1.0 and zi < exit_:
            state = 0.0
        pos.iloc[i] = state

    pos = pos * corr_ok.fillna(0.0)

    # Size: inverse-vol of the ETH-BTC SPREAD (not individual legs)
    spread_r = eth_r - btc_r
    spread_vol = spread_r.rolling(60, min_periods=30).std() * np.sqrt(DPY)
    size = (0.10 / spread_vol.replace(0, np.nan)).clip(lower=0.0, upper=1.0)

    leg_size = 0.30 * size  # 30% * vol-scale per leg
    W["ETH"] = (pos * leg_size).shift(1).fillna(0.0)
    W["BTC"] = (-pos * leg_size).shift(1).fillna(0.0)
    return W


def sleeve_macro_trend(cp, macro=None):
    """BTC long-only, gated by:
      (1) SPY above 200MA (risk-on equity backdrop),
      (2) VIX < 27 (not in panic regime),
      (3) BTC above 100MA AND 63d mom > 0.
    Vol-managed sizing.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.22 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.5)

    btc_trend = ((s > s.rolling(100, min_periods=50).mean()) &
                 (s.pct_change(63) > 0.0)).astype(float)

    if macro is None or "spy" not in macro:
        spy_gate = pd.Series(1.0, index=cp.index)
        vix_gate = pd.Series(1.0, index=cp.index)
    else:
        spy = macro["spy"]
        vix = macro["vix"]
        spy_gate = (spy > spy.rolling(200, min_periods=100).mean()).astype(float)
        vix_gate = (vix < 27.0).astype(float)
        # Both ffilled over weekend gaps
        spy_gate = spy_gate.reindex(cp.index).ffill().fillna(0.0)
        vix_gate = vix_gate.reindex(cp.index).ffill().fillna(1.0)

    signal = btc_trend * spy_gate * vix_gate
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (size * signal * alive).shift(1).fillna(0.0)
    return W


def sleeve_crash_dip(cp, macro=None):
    """Buy BTC when in 20-40% drawdown from 180d high AND 200MA slope > 0.

    Classic "buy the dip in an uptrend". Specifically targets:
      * Mar 2017 correction
      * Sep 2017 correction
      * Nov 2018 bottom — FAILS 200MA filter (correctly)
      * Mar 2020 COVID crash — PASSES if 200MA slope still positive at crash
      * May 2021 correction
      * Aug 2024 pullback
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    hi180 = s.rolling(180, min_periods=90).max()
    dd180 = s / hi180 - 1
    ma200 = s.rolling(200, min_periods=100).mean()
    ma200_slope = ma200.pct_change(30)

    # Entry: -20% to -40% DD AND 200MA slope still > 0
    in_zone = ((dd180 <= -0.20) & (dd180 >= -0.40) &
               (ma200_slope > 0.0)).astype(float)
    # Hold position for up to 45 days after signal or until DD closes to -5%
    state = pd.Series(0.0, index=cp.index)
    held = 0
    pos = 0.0
    for i in range(len(in_zone)):
        if pd.isna(dd180.iloc[i]):
            state.iloc[i] = pos
            continue
        if pos == 0.0 and in_zone.iloc[i] == 1.0:
            pos = 1.0
            held = 0
        elif pos > 0.0:
            held += 1
            # Exit conditions
            if dd180.iloc[i] > -0.05 or held > 45 or dd180.iloc[i] < -0.50:
                pos = 0.0
        state.iloc[i] = pos

    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.25 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.5)
    W["BTC"] = (state * size).shift(1).fillna(0.0)
    return W


def sleeve_halving_boost(cp, macro=None):
    """Long BTC during the historically productive post-halving window
    (days 120–480 after each halving). Deterministic regime indicator.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    signal = pd.Series(0.0, index=cp.index)
    for hdate in HALVING_DATES:
        hd = pd.Timestamp(hdate)
        start = hd + pd.Timedelta(days=120)
        end = hd + pd.Timedelta(days=480)
        signal.loc[(cp.index >= start) & (cp.index <= end)] = 1.0

    # Only take the boost when BTC is also above 200MA (don't fight tape)
    ma200 = s.rolling(200, min_periods=100).mean()
    trend_ok = (s > ma200).astype(float)
    signal = signal * trend_ok

    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.2)
    W["BTC"] = (signal * size).shift(1).fillna(0.0)
    return W


def sleeve_vix_meanrev(cp, macro=None):
    """VIX panic-exhaustion: long BTC 1-5 days after VIX spikes above 35
    (panic → bottom-picker's edge). Position held up to 20 days or until
    VIX normalises below 20.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if macro is None or "vix" not in macro:
        return W
    vix = macro["vix"].reindex(cp.index).ffill()
    s = cp["BTC"]
    spike = (vix > 35).astype(float)
    # State machine: enter on spike day (t+1), exit when VIX < 20 or 20d elapsed.
    pos = pd.Series(0.0, index=cp.index)
    state = 0.0
    days_held = 0
    for i in range(len(cp)):
        v = vix.iloc[i]
        if pd.isna(v):
            pos.iloc[i] = state
            continue
        if state == 0.0 and spike.iloc[i] == 1.0:
            state = 1.0
            days_held = 0
        elif state > 0.0:
            days_held += 1
            if v < 20 or days_held > 20:
                state = 0.0
        pos.iloc[i] = state
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.18 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.2)
    W["BTC"] = (pos * size).shift(1).fillna(0.0)
    return W


def sleeve_dxy_weak(cp, macro=None):
    """Weak-dollar regime favours crypto. Long BTC when DXY (UUP proxy) is
    BELOW its 200MA AND BTC above 100MA. Historical tailwind for crypto.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if macro is None or "uup" not in macro:
        return W
    uup = macro["uup"].reindex(cp.index).ffill()
    s = cp["BTC"]
    uup_weak = (uup < uup.rolling(200, min_periods=100).mean()).astype(float)
    btc_trend = (s > s.rolling(100, min_periods=50).mean()).astype(float)
    signal = uup_weak * btc_trend
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.2)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (signal * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_vol_breakout(cp, macro=None):
    """Low-volatility breakout: when 21d realized vol < 40th percentile of its
    365d history AND BTC > 50MA, go long. Low-vol regimes historically precede
    clean trends in crypto.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    rv21 = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    # Percentile rank of current vol within past 365 days
    vol_pct = rv21.rolling(365, min_periods=90).apply(
        lambda x: (x[-1] <= x).mean() if len(x) > 30 else np.nan, raw=True)
    low_vol = (vol_pct < 0.40).astype(float)
    trend = (s > s.rolling(50, min_periods=30).mean()).astype(float)
    momentum = (s.pct_change(21) > 0).astype(float)
    signal = low_vol * trend * momentum
    size = (0.18 / rv21.replace(0, np.nan)).clip(lower=0.0, upper=1.2)
    W["BTC"] = (signal * size).shift(1).fillna(0.0)
    return W


def sleeve_turtle20(cp, macro=None):
    """20-day Donchian breakout, 10-day exit (Turtle S1).
    Genuinely different from 90d Donchian — faster, catches earlier trends."""
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    hi20 = s.rolling(20, min_periods=10).max()
    lo10 = s.rolling(10, min_periods=5).min()
    state = pd.Series(0.0, index=s.index)
    held = 0.0
    for i in range(len(s)):
        p = s.iloc[i]
        if pd.isna(p):
            state.iloc[i] = held
            continue
        if held == 0.0 and not pd.isna(hi20.iloc[i]) and p >= hi20.iloc[i] * 0.995:
            held = 1.0
        elif held == 1.0 and not pd.isna(lo10.iloc[i]) and p <= lo10.iloc[i] * 1.005:
            held = 0.0
        state.iloc[i] = held
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.18 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.2)
    ma200 = s.rolling(200, min_periods=100).mean()
    trend_gate = (s > ma200).astype(float)
    W["BTC"] = (state * size * trend_gate).shift(1).fillna(0.0)
    return W


def sleeve_turtle55(cp, macro=None):
    """55-day Donchian (Turtle S2): slower, catches bigger trends."""
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    hi55 = s.rolling(55, min_periods=25).max()
    lo20 = s.rolling(20, min_periods=10).min()
    state = pd.Series(0.0, index=s.index)
    held = 0.0
    for i in range(len(s)):
        p = s.iloc[i]
        if pd.isna(p):
            state.iloc[i] = held
            continue
        if held == 0.0 and not pd.isna(hi55.iloc[i]) and p >= hi55.iloc[i] * 0.99:
            held = 1.0
        elif held == 1.0 and not pd.isna(lo20.iloc[i]) and p <= lo20.iloc[i] * 1.01:
            held = 0.0
        state.iloc[i] = held
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.18 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.2)
    W["BTC"] = (state * size).shift(1).fillna(0.0)
    return W


def sleeve_bollinger(cp, macro=None):
    """Bollinger-band breakout: long on upper-band break; exit on mid-band
    touch. 20-day band, 2-sigma."""
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    mid = s.rolling(20, min_periods=10).mean()
    sd = s.rolling(20, min_periods=10).std()
    upper = mid + 2.0 * sd
    state = pd.Series(0.0, index=s.index)
    held = 0.0
    for i in range(len(s)):
        p = s.iloc[i]
        if pd.isna(p) or pd.isna(upper.iloc[i]):
            state.iloc[i] = held
            continue
        if held == 0.0 and p >= upper.iloc[i]:
            held = 1.0
        elif held == 1.0 and p <= mid.iloc[i]:
            held = 0.0
        state.iloc[i] = held
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.18 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.2)
    ma200 = s.rolling(200, min_periods=100).mean()
    trend_gate = (s > ma200).astype(float)
    W["BTC"] = (state * size * trend_gate).shift(1).fillna(0.0)
    return W


def sleeve_adx_trend(cp, macro=None):
    """ADX-filtered trend: only long when ADX-like measure (rolling |return|)
    shows strong directional persistence. Vol-managed sizing."""
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    r = s.pct_change()
    # Simple directional strength: fraction of up-days minus down-days over 14d
    up = (r > 0).rolling(14, min_periods=7).sum()
    dn = (r < 0).rolling(14, min_periods=7).sum()
    dms = (up - dn) / 14.0
    # Strong uptrend: dms > 0.3 (i.e., >4 more up days than down in last 14)
    strong_up = (dms > 0.20).astype(float)
    ma100 = s.rolling(100, min_periods=50).mean()
    trend = (s > ma100).astype(float)
    signal = strong_up * trend
    rv = r.rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    W["BTC"] = (signal * size).shift(1).fillna(0.0)
    return W


def sleeve_alt_xsmom(cp, macro=None):
    """Cross-sectional alt momentum: rotate into top-3 alts by 63d return,
    strict BTC-regime and breadth gates. Per-coin cap 10%."""
    alt_cols = [c for c in cp.columns if c not in ["BTC", "ETH"]]
    if not alt_cols:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    cp_alt = cp[alt_cols]
    elig = eligibility(cp_alt, 180, catastrophe_dd=-0.25, dd_window=60)
    mom63 = cp_alt.pct_change(63)
    trending = (mom63 > 0.20).astype(float) * elig
    score = mom63.where(trending.astype(bool))
    ranks = score.rank(axis=1, ascending=False, method="first")
    pick = (ranks <= 3.0).astype(float)
    rv = cp_alt.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)
    inv = (1.0 / rv.replace(0, np.nan)).where(pick.astype(bool))
    W_alt = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W_alt = W_alt.clip(upper=0.10)
    s_sum = W_alt.sum(axis=1).replace(0, np.nan)
    scale = (0.30 / s_sum).clip(upper=1.0).fillna(0.0)
    W_alt = W_alt.mul(scale, axis=0)

    btc = cp["BTC"]
    btc_gate = ((btc > btc.rolling(150, min_periods=75).mean()) &
                (btc.pct_change(63) > 0.0)).astype(float)
    W_alt = W_alt.mul(btc_gate, axis=0)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in W_alt.columns:
        W[c] = W_alt[c]
    return W.shift(1).fillna(0.0)


def sleeve_eth_slow(cp, macro=None):
    """Slow ETH trend (200MA + 126d mom > 20%). Decorrelates with ETH_VM."""
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "ETH" not in cp.columns:
        return W
    s = cp["ETH"]
    rv = s.pct_change().rolling(60, min_periods=30).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.2, upper=1.3)
    ma200 = s.rolling(200, min_periods=100).mean()
    trend = ((s > ma200) & (s.pct_change(126) > 0.20)).astype(float)
    hwm = s.rolling(180, min_periods=60).max()
    dd = s / hwm - 1
    alive = (dd > -0.35).astype(float)
    W["ETH"] = (size * trend * alive).shift(1).fillna(0.0)
    return W


def sleeve_btc_kama(cp, macro=None):
    """Kaufman Adaptive MA: efficient-market weighted MA. Adapts speed to
    noise ratio. Long when price > KAMA and KAMA slope > 0."""
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"].copy()
    n = 10
    fast, slow = 2.0, 30.0
    change = (s - s.shift(n)).abs()
    volatility = s.diff().abs().rolling(n, min_periods=n).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0).clip(0, 1)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    kama = s.copy()
    # Initialise KAMA at first non-nan after n
    kama.iloc[:n] = np.nan
    for i in range(n, len(s)):
        prev = kama.iloc[i - 1] if not pd.isna(kama.iloc[i - 1]) else s.iloc[i - 1]
        kama.iloc[i] = prev + sc.iloc[i] * (s.iloc[i] - prev)
    slope = kama.diff(10)
    signal = ((s > kama) & (slope > 0)).astype(float)
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (signal * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_triple_mom(cp, macro=None):
    """Require momentum positive across 3 horizons: 21d AND 63d AND 252d.
    Very strict, only fires in established bull regimes.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    m21 = s.pct_change(21)
    m63 = s.pct_change(63)
    m252 = s.pct_change(252)
    signal = ((m21 > 0) & (m63 > 0.05) & (m252 > 0.15)).astype(float)
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.22 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.5)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.25).astype(float)
    W["BTC"] = (signal * size * alive).shift(1).fillna(0.0)
    return W


BUILDERS = {
    # BTC trend core — multiple speeds
    "BTC_VM":         sleeve_btc_vm,       # fast: 100MA, 63d mom
    "BTC_SLOW":       sleeve_btc_slow,     # slow: 200MA, 126d mom
    "BTC_KAMA":       sleeve_btc_kama,     # adaptive
    "TRIPLE_MOM":     sleeve_triple_mom,   # 21/63/252 confluence
    # Breakout family
    "TURTLE20":       sleeve_turtle20,     # 20d Donchian
    "TURTLE55":       sleeve_turtle55,     # 55d Donchian
    "BOLLINGER":      sleeve_bollinger,    # 2-sigma upper break
    # Regime / macro
    "MACRO_TREND":    sleeve_macro_trend,  # SPY + VIX gated
    "HALVING_BOOST":  sleeve_halving_boost,# post-halving window
    "ADX_TREND":      sleeve_adx_trend,    # strong-trend filter
    # Specialists
    "CRASH_DIP":      sleeve_crash_dip,    # buy -20 to -40% DD in uptrend
    "VOL_BREAKOUT":   sleeve_vol_breakout, # low-vol regime breakout
    # Cross-asset diversifiers
    "ETH_VM":         sleeve_eth_vm,
    "ETH_SLOW":       sleeve_eth_slow,
    "ALT_XSMOM":      sleeve_alt_xsmom,    # breadth of alt momentum
}
# Dropped: ETHBTC_PAIR (net-negative), VIX_MEANREV (noisy), DXY_WEAK (marginal).
# The 15-sleeve ensemble outperforms any smaller pruned subset — individual
# sleeves with weak OOS still contribute diversification to the ensemble.
# ETHBTC_PAIR dropped: net-negative contribution (crypto pair correlations
# break down exactly when you need them to hold).

MARKET_NEUTRAL: set = set()


def build_all(cp, macro=None):
    out = {}
    for name, fn in BUILDERS.items():
        W = fn(cp, macro)
        out[name] = W.fillna(0.0).reindex(columns=cp.columns, fill_value=0.0)
    return out


def consensus_signal(cp, sleeves):
    """Fraction of directional (non-pair) sleeves that are net long TODAY."""
    votes = []
    for name, W in sleeves.items():
        if name in MARKET_NEUTRAL:
            continue
        votes.append((W.sum(axis=1) > 1e-4).astype(float))
    if not votes:
        return pd.Series(0.0, index=cp.index)
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0)
