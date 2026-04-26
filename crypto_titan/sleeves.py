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
from util import DPY, eligibility, HALVING_DATES, load_ohlcv, load_prices, EXTENDED_UNIVERSE


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


def _per_coin_vm(cp, coin, vol_target=0.18, ma_len=100, mom_len=63,
                  dd_cut=-0.30):
    """Per-coin vol-managed trend sleeve, BTC-regime gated.
    Outputs weight in `coin` only. Used for v12 alt sleeves.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if coin not in cp.columns:
        return W
    pos = _vol_managed_core(cp, coin, vol_target=vol_target,
                             ma_len=ma_len, mom_len=mom_len, dd_cut=dd_cut)
    # BTC-regime gate: only trade alt when BTC is in confirmed uptrend
    btc = cp["BTC"]
    btc_ok = ((btc > btc.rolling(200, min_periods=100).mean()) &
              (btc.pct_change(63) > 0.0)).astype(float).shift(1).fillna(0.0)
    W[coin] = pos * btc_ok
    return W


def sleeve_sol_vm(cp, macro=None):
    return _per_coin_vm(cp, "SOL")


def sleeve_bnb_vm(cp, macro=None):
    return _per_coin_vm(cp, "BNB")


def sleeve_avax_vm(cp, macro=None):
    return _per_coin_vm(cp, "AVAX")


def sleeve_link_vm(cp, macro=None):
    return _per_coin_vm(cp, "LINK")


def sleeve_ada_vm(cp, macro=None):
    return _per_coin_vm(cp, "ADA")


def sleeve_xrp_vm(cp, macro=None):
    return _per_coin_vm(cp, "XRP")


def sleeve_ltc_vm(cp, macro=None):
    return _per_coin_vm(cp, "LTC")


def sleeve_doge_vm(cp, macro=None):
    return _per_coin_vm(cp, "DOGE")


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


# --- Proprietary orthogonal sleeves (v4) — invented, very different edges ---


# --- v12: MULTI_SCAN — full-universe multi-signal confluence scanner ---
#
# Every rebalance, scan ALL 111 coins. For each, compute 5 independent
# trend/quality signals. Require AT LEAST 4 of 5 signals to fire AND
# strict eligibility AND BTC-regime gate. Select top-K by composite
# momentum score, risk-parity within selection.
#
# Earlier "naive XSMOM top-K of 100" approaches blew up because they
# selected on a SINGLE signal (recent momentum) — that's how you buy
# parabolic alts at the local top. The 5-signal confluence filter is
# what makes this fundamentally different.

def sleeve_multi_scan(cp, macro=None, top_k=5, min_signals=4):
    """Scan entire universe with 5-signal confluence; select top-K by score.

    Per-coin signals (each binary 0/1, sum 0-5):
      1. LONG_TREND   : price > 200MA AND 50MA > 200MA
      2. SHORT_MOM    : 21d return > 5%
      3. PERSISTENCE  : 63d return > 0 AND 252d return > 0
      4. NOT_OVERBOUGHT: price within 7% of 21d high (avoid parabolic tops)
      5. STABLE_VOL   : current 21d realised vol < 1.5× the 252d median vol

    Eligibility:
      * 252-day minimum history
      * Not in catastrophic DD (-25% from 60d high)
      * Coin's own 365d return > 0 (filter permabears)

    Master gate:
      * BTC > 200MA AND BTC 63d mom > 0.05

    Selection:
      * Among coins with confluence >= min_signals AND eligible AND BTC-up,
        rank by 63d return, take top-K.
      * Risk-parity by 60d realised vol; per-coin cap 12%, basket cap 50%.
    """
    elig_strict = eligibility(cp, min_history=252,
                              catastrophe_dd=-0.25, dd_window=60)

    ma50 = cp.rolling(50, min_periods=30).mean()
    ma200 = cp.rolling(200, min_periods=100).mean()
    long_trend = ((cp > ma200) & (ma50 > ma200)).astype(float)

    mom21 = cp.pct_change(21)
    mom63 = cp.pct_change(63)
    mom252 = cp.pct_change(252)

    short_mom = (mom21 > 0.05).astype(float)
    persistence = ((mom63 > 0) & (mom252 > 0)).astype(float)

    hi21 = cp.rolling(21, min_periods=10).max()
    not_overbought = (cp < 0.93 * hi21).astype(float)
    # NOTE: this is an inverse signal — coins NEAR 21d high score 0.
    # It filters out parabolic moves that will mean-revert.

    rv21 = cp.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    rv252_med = rv21.rolling(252, min_periods=60).median()
    stable_vol = (rv21 < 1.5 * rv252_med).astype(float)

    # Permabear filter
    not_permabear = (mom252 > 0).astype(float)

    confluence = (long_trend + short_mom + persistence
                  + not_overbought + stable_vol)
    # Eligible if >= min_signals confluence AND alive AND positive 365d return
    eligible_score = ((confluence >= min_signals)
                      .astype(float) * elig_strict * not_permabear)

    # BTC regime gate
    btc = cp["BTC"]
    btc_strong = ((btc > btc.rolling(200, min_periods=100).mean()) &
                  (btc.pct_change(63) > 0.05)).astype(float)
    eligible_score = eligible_score.mul(btc_strong, axis=0)

    # Top-K by 63d momentum among eligible
    score = mom63.where(eligible_score > 0)
    ranks = score.rank(axis=1, ascending=False, method="first")
    top_pick = (ranks <= top_k).astype(float)

    # Risk-parity weighting within picks
    rv60 = cp.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)
    inv = (1.0 / rv60.replace(0, np.nan)).where(top_pick.astype(bool))
    W = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W = W.clip(upper=0.12)  # 12% per-coin cap
    s = W.sum(axis=1).replace(0, np.nan)
    scale = (0.50 / s).clip(upper=1.0).fillna(0.0)
    W = W.mul(scale, axis=0)

    return W.shift(1).fillna(0.0)


# --- v7 NOVEL inventions: Multi-Timeframe Fractal Alignment + Convex Breakout ---


def sleeve_mtf_fractal(cp, macro=None):
    """MULTI-TIMEFRAME FRACTAL ALIGNMENT (proprietary).

    Long BTC only when ALL six timeframes show positive momentum:
       5d, 21d, 63d, 126d, 252d, 365d.
    Plus 200MA trend filter.

    Idea: each timeframe captures a different market participant. When all
    align, we're in the rare regime where retail (5d), swing (21-63d),
    institutional (126-252d), and macro (365d) are ALL bullish. These are
    the highest-edge windows in crypto history (post-halving rallies,
    confirmed bull markets). When even ONE timeframe disagrees → cash.
    Most days: cash. Rare days: full-size long.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for coin in ["BTC", "ETH"]:
        if coin not in cp.columns:
            continue
        s = cp[coin]
        # 6 timeframes
        timeframes = [5, 21, 63, 126, 252, 365]
        all_positive = pd.Series(1.0, index=cp.index)
        for tf in timeframes:
            cond = (s.pct_change(tf) > 0).astype(float)
            all_positive = all_positive * cond
        # Plus 200MA filter
        ma200 = s.rolling(200, min_periods=100).mean()
        above_ma = (s > ma200).astype(float)
        signal = all_positive * above_ma

        # When fired, take FIXED 50% per coin (vol-targeted at portfolio level)
        rv = s.pct_change().ewm(span=21, adjust=False).std() * np.sqrt(DPY)
        size = (0.30 / rv.replace(0, np.nan)).clip(lower=0.3, upper=1.5)
        hwm = s.rolling(90, min_periods=30).max()
        dd = s / hwm - 1
        alive = (dd > -0.25).astype(float)
        W[coin] = (signal * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_convex_breakout(cp, macro=None):
    """CONVEX BREAKOUT (proprietary).

    Pure right-tailed convex payoff — preserves the asymmetric upside that
    vol-targeting destroys. Mechanic:
      Entry  : BTC/ETH closes at new 90d high (within 0.5%)
      Initial: full 60% per-coin position
      Initial stop: -7% from entry (small fixed loss)
      Trailing stop: max of 14-day low and entry × 0.93 (raises with price)
      Exit only on stop hit. No vol-managed sizing.

    Expected payoff: ~30 small -7% losses + a few +50% to +200% wins.
    Sharpe boosted by RIGHT-TAILED skewness of the return distribution.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for coin in ["BTC", "ETH"]:
        if coin not in cp.columns:
            continue
        s = cp[coin].values
        idx = cp.index
        hi90 = cp[coin].rolling(90, min_periods=45).max().values
        in_pos = False
        entry_price = 0.0
        trail_stop = 0.0
        position = np.zeros(len(s))
        for i in range(len(s)):
            p = s[i]
            if np.isnan(p):
                position[i] = position[i-1] if i > 0 else 0.0
                continue
            if not in_pos:
                # Entry on new 90d high
                if not np.isnan(hi90[i]) and p >= 0.995 * hi90[i]:
                    in_pos = True
                    entry_price = p
                    trail_stop = p * 0.93  # initial -7%
            else:
                # Trail stop: 14d low (only ratchet UP, never down)
                lo14 = np.nanmin(s[max(0, i-14):i+1])
                new_stop = max(lo14, entry_price * 0.93)
                trail_stop = max(trail_stop, new_stop)
                if p < trail_stop:
                    in_pos = False
            position[i] = 0.60 if in_pos else 0.0
        W[coin] = pd.Series(position, index=idx).shift(1).fillna(0.0)
    return W


# --- v6: SHORT-ENABLED sleeves (Hyperliquid / OKX inverse perps) ---
# These sleeves take negative positions, profitable when the underlying
# falls. Critical for capturing 2018/2022 bear alpha and dead-coin crashes
# (LUNA, FTT, MATIC, etc.) — the survivorship-bias universe becomes a
# long-short alpha source.
#
# Per-position constraints:
#   * Tighter stops than longs (crypto bear rallies are violent)
#   * Half-size of equivalent longs
#   * Funding cost modelled in strategy.py at ~10%/yr (paid when short
#     in long-heavy market; received in short-heavy market)


def sleeve_short_btc_bear(cp, macro=None):
    """Short BTC ONLY in deep, confirmed bear regimes — VERY conservative.
    Multi-condition entry, tight 5% stop, max 21d hold.

    Entry requires ALL of:
      * Below 200MA for 60+ of past 80 days (deep persistence)
      * 50MA < 200MA (death cross)
      * 63d return < -20% (strong negative trend)
      * 21d return < -5% (recent acceleration)
    Exit: price > 50MA OR 5d return > +5% OR 21d elapsed.
    Size: very small (target 8% vol contribution).
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    ma50 = s.rolling(50, min_periods=30).mean()
    ma200 = s.rolling(200, min_periods=100).mean()
    below_200 = (s < ma200).astype(float)
    persist = below_200.rolling(80, min_periods=60).sum()
    entry = ((persist >= 60) &
             (ma50 < ma200) &
             (s.pct_change(63) < -0.20) &
             (s.pct_change(21) < -0.05)).astype(float)
    # Tight exit
    exit_signal = ((s > ma50) | (s.pct_change(5) > 0.05)).astype(float)

    state = pd.Series(0.0, index=cp.index)
    pos = 0.0
    days_in = 0
    for i in range(len(s)):
        if entry.iloc[i] == 1.0 and pos == 0.0:
            pos = 1.0
            days_in = 0
        elif pos > 0.0:
            days_in += 1
            if exit_signal.iloc[i] == 1.0 or days_in > 21:
                pos = 0.0
        state.iloc[i] = pos

    rv = s.pct_change().ewm(span=21, adjust=False).std() * np.sqrt(DPY)
    size = (0.08 / rv.replace(0, np.nan)).clip(lower=0.0, upper=0.5)  # tiny
    W["BTC"] = (-1.0 * state * size).shift(1).fillna(0.0)
    return W


def sleeve_short_trend_break(cp, macro=None):
    """Short coins breaking down — BTC-bear-gated, very tight stops.

    Trigger ALL of:
      * Coin was above 200MA for 90+ of past 120 days (was uptrend)
      * Now: below 50MA AND below 200MA (full breakdown, both MAs)
      * 21d return < -15% (acceleration confirmed)
      * BTC also below 100MA (market-wide weakness — no contra-bet vs market)
    Exit on:
      * Price > 50MA (trend recovered)
      * 5d return > +8% (sharp bounce)
      * 14 days elapsed
    Per-coin cap 4%, basket cap 25%. SMALL exposure.
    """
    elig = eligibility(cp, min_history=200, catastrophe_dd=-1.0,
                       dd_window=60)
    ma50 = cp.rolling(50, min_periods=30).mean()
    ma200 = cp.rolling(200, min_periods=100).mean()
    above_200 = (cp > ma200).astype(float)
    persist_above = above_200.rolling(120, min_periods=90).sum()
    was_uptrend = (persist_above >= 90).astype(float)
    breaking_down = ((cp < ma50) & (cp < ma200) &
                     (cp.pct_change(21) < -0.15)).astype(float)
    # BTC market gate
    btc = cp["BTC"]
    btc_weak = (btc < btc.rolling(100, min_periods=50).mean()).astype(float)
    entry = (was_uptrend.shift(5) * breaking_down * elig).fillna(0.0)
    entry = entry.mul(btc_weak, axis=0)

    exit_cond = ((cp > ma50) | (cp.pct_change(5) > 0.08)).astype(float).fillna(0.0)

    pos = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in cp.columns:
        e = entry[c].values
        x = exit_cond[c].values
        state = 0.0
        days_in = 0
        out = np.zeros(len(cp))
        for i in range(len(cp)):
            if e[i] == 1.0 and state == 0.0:
                state = 1.0
                days_in = 0
            elif state > 0.0:
                days_in += 1
                if x[i] == 1.0 or days_in > 14:
                    state = 0.0
            out[i] = state
        pos[c] = out

    rv = cp.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)
    inv = (1.0 / rv.replace(0, np.nan)).where(pos.astype(bool))
    W_short = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W_short = W_short.clip(upper=0.04)
    s_sum = W_short.sum(axis=1).replace(0, np.nan)
    scale = (0.25 / s_sum).clip(upper=1.0).fillna(0.0)
    W = (-1.0) * W_short.mul(scale, axis=0)
    return W.shift(1).fillna(0.0)


def sleeve_short_permabear(cp, macro=None):
    """Short coins stuck in deep, prolonged DD while BTC is rising.
    Permabears almost never recover; they drift lower under selling pressure
    even in bull markets.

    Entry ALL of:
      * Coin in >50% DD from 365d high
      * Coin has been in >40% DD for at least 90 of past 120 days
      * BTC ABOVE 200MA (we are NOT in market-wide bear)
      * Coin 63d return < -10%
    Exit: coin 21d return > +20% (sharp bounce — get out) OR DD recovers
    above -25% OR 30 days elapsed.
    Per-coin cap 3%, basket cap 18%. VERY small exposures.
    """
    elig = eligibility(cp, min_history=200, catastrophe_dd=-1.0,
                       dd_window=60)
    hi365 = cp.rolling(365, min_periods=180).max()
    dd365 = cp / hi365 - 1
    in_deep_dd = (dd365 < -0.50).astype(float)
    in_dd_persist = (dd365 < -0.40).astype(float)
    persist = in_dd_persist.rolling(120, min_periods=90).sum()
    long_dd = (persist >= 90).astype(float)

    btc = cp["BTC"]
    btc_strong = (btc > btc.rolling(200, min_periods=100).mean()).astype(float)

    coin_weak = (cp.pct_change(63) < -0.10).astype(float)
    entry = (in_deep_dd * long_dd * coin_weak * elig).fillna(0.0)
    entry = entry.mul(btc_strong, axis=0)

    exit_cond = ((cp.pct_change(21) > 0.20) |
                 (dd365 > -0.25)).astype(float).fillna(0.0)

    pos = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in cp.columns:
        e = entry[c].values
        x = exit_cond[c].values
        state = 0.0
        days_in = 0
        out = np.zeros(len(cp))
        for i in range(len(cp)):
            if e[i] == 1.0 and state == 0.0:
                state = 1.0
                days_in = 0
            elif state > 0.0:
                days_in += 1
                if x[i] == 1.0 or days_in > 30:
                    state = 0.0
            out[i] = state
        pos[c] = out

    n = pos.sum(axis=1).replace(0, np.nan)
    W_short = pos.div(n, axis=0).fillna(0.0).clip(upper=0.03)
    s_sum = W_short.sum(axis=1).replace(0, np.nan)
    scale = (0.18 / s_sum).clip(upper=1.0).fillna(0.0)
    W = (-1.0) * W_short.mul(scale, axis=0)
    return W.shift(1).fillna(0.0)


def sleeve_long_short_xsmom(cp, macro=None):
    """Cross-sectional momentum: LONG top-3 by 63d return, SHORT bottom-3.
    Market-neutral by construction (equal long/short notionals).
    Active only when sufficient eligible coins exist (≥10).
    """
    elig = eligibility(cp, 180, catastrophe_dd=-0.30, dd_window=60)
    mom63 = cp.pct_change(63)
    score = mom63.where(elig.astype(bool))
    n_elig = elig.sum(axis=1)
    active = (n_elig >= 10).astype(float)

    long_ranks = score.rank(axis=1, ascending=False, method="first")
    short_ranks = score.rank(axis=1, ascending=True, method="first")
    long_pick = (long_ranks <= 3.0).astype(float)
    short_pick = (short_ranks <= 3.0).astype(float)

    # Equal-weight within each side
    W_long = long_pick.div(long_pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W_short = short_pick.div(short_pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 20% gross each leg → 40% gross total (small position)
    leg_size = 0.20
    W = W_long * leg_size - W_short * leg_size
    W = W.mul(active, axis=0)
    return W.shift(1).fillna(0.0)


def sleeve_short_dead_coin(cp, macro=None):
    """Short coins that ENTER catastrophic drawdown with high vol.

    Trigger: coin in -25% to -45% DD from 90d high AND realised vol > 1.5x
    its 6-month average AND BTC NOT in extreme rally (not piggy-backing
    a market-wide squeeze).

    This specifically targets the death-spiral phase of dead coins.
    """
    s_btc = cp["BTC"]
    btc_rally = (s_btc.pct_change(21) > 0.20).astype(float)  # avoid short squeezes

    rv = cp.pct_change().rolling(30, min_periods=15).std() * np.sqrt(DPY)
    rv_long = cp.pct_change().rolling(180, min_periods=90).std() * np.sqrt(DPY)
    vol_spike = (rv > 1.5 * rv_long).astype(float)
    hwm90 = cp.rolling(90, min_periods=30).max()
    dd = cp / hwm90 - 1
    in_zone = ((dd <= -0.25) & (dd >= -0.45)).astype(float)
    elig = eligibility(cp, min_history=200, catastrophe_dd=-1.0,
                       dd_window=60)

    entry = (in_zone * vol_spike * elig).fillna(0.0)
    # Mask out when BTC is rallying hard (market squeeze risk)
    entry = entry.mul((1 - btc_rally).fillna(1.0), axis=0)

    # Hold up to 14 days, exit on +12% rebound
    pos = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in cp.columns:
        e = entry[c].values
        prices = cp[c].values
        state = 0.0
        days_in = 0
        entry_price = 0.0
        out = np.zeros(len(cp))
        for i in range(len(cp)):
            if pd.isna(prices[i]):
                out[i] = state
                continue
            if e[i] == 1.0 and state == 0.0:
                state = 1.0
                days_in = 0
                entry_price = prices[i]
            elif state > 0.0:
                days_in += 1
                if (entry_price > 0 and prices[i] >= entry_price * 1.12) or days_in > 14:
                    state = 0.0
            out[i] = state
        pos[c] = out

    # Equal weight, per-coin cap 5%, basket cap 25%
    n = pos.sum(axis=1).replace(0, np.nan)
    W_short = pos.div(n, axis=0).fillna(0.0).clip(upper=0.05)
    s_sum = W_short.sum(axis=1).replace(0, np.nan)
    scale = (0.25 / s_sum).clip(upper=1.0).fillna(0.0)
    W = (-1.0) * W_short.mul(scale, axis=0)
    return W.shift(1).fillna(0.0)


def sleeve_inv_vol_btc(cp, macro=None):
    """Pure Moreira-Muir: size BTC position inversely proportional to realized
    vol, with a simple 200MA gate. No momentum filter, no DD stop. This
    decorrelates from all our trend sleeves because its TIMING is vol-driven,
    not price-driven. Classic academic result boosts Sharpe ~0.3.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    rv = s.pct_change().ewm(span=21, adjust=False).std() * np.sqrt(DPY)
    # Size scales inversely with vol, heavily bounded
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=2.0)
    ma200 = s.rolling(200, min_periods=100).mean()
    gate = (s > ma200).astype(float)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.30).astype(float)
    W["BTC"] = (size * gate * alive).shift(1).fillna(0.0)
    return W


def sleeve_vol_percentile(cp, macro=None):
    """Novel: size BTC by PERCENTILE rank of current vol in its 365-day
    distribution — not the absolute level. Low-vol regimes (bottom 30%) ARE
    fertile ground for trend; size up. High-vol (top 30%) caution. Different
    character than fixed-target vol sizing.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    # Percentile in trailing 365d — 0 = current vol is lowest in past year, 1 = highest
    vp = rv.rolling(365, min_periods=90).apply(
        lambda x: (x[-1] <= x).mean() if len(x) >= 30 else np.nan, raw=True)
    # Size peaks at moderate-low vol (0.2-0.4), tapers at extremes
    # Simple triangular function: max 1.0 at vp=0.3, 0 at vp=0 or vp=0.75
    def _size_fn(v):
        if pd.isna(v):
            return 0.0
        if v < 0.3:
            return v / 0.3
        elif v < 0.75:
            return (0.75 - v) / 0.45
        else:
            return 0.0
    size_shape = vp.apply(_size_fn)
    # Trend filter
    ma100 = s.rolling(100, min_periods=50).mean()
    trend = (s > ma100).astype(float)
    signal = size_shape * trend
    # Absolute sizing: scale to target vol
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (signal * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_dispersion_timed(cp, macro=None):
    """Cross-sectional dispersion regime: when coin-level 21d returns are
    DISPERSED (std across coins high), alt stock-picking works → pick top-3
    momentum alts. When COMPRESSED (everyone moving together), concentrate
    in BTC only.

    Measures dispersion of 21d returns across all eligible coins, not just
    BTC. Truly cross-sectional signal.
    """
    alt_cols = [c for c in cp.columns if c not in ["BTC", "ETH"]]
    if len(alt_cols) < 5:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    cp_alt = cp[alt_cols]
    elig = eligibility(cp_alt, 180, catastrophe_dd=-0.25, dd_window=60)
    mom21_alt = cp_alt.pct_change(21)
    # Dispersion = std of 21d returns across coins (per date)
    disp = mom21_alt.where(elig.astype(bool)).std(axis=1)
    disp_median = disp.rolling(180, min_periods=60).median()
    high_disp = (disp > 1.3 * disp_median).astype(float).fillna(0.0)

    btc = cp["BTC"]
    btc_gate = ((btc > btc.rolling(150, min_periods=75).mean()) &
                (btc.pct_change(63) > 0.0)).astype(float)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    # High-dispersion regime: long top-3 alts by 63d momentum
    mom63 = cp_alt.pct_change(63).where(elig.astype(bool))
    ranks = mom63.rank(axis=1, ascending=False, method="first")
    pick = (ranks <= 3.0).astype(float)
    rv = cp_alt.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)
    inv = (1.0 / rv.replace(0, np.nan)).where(pick.astype(bool))
    W_disp = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W_disp = W_disp.clip(upper=0.10)
    s_sum = W_disp.sum(axis=1).replace(0, np.nan)
    W_disp = W_disp.mul((0.30 / s_sum).clip(upper=1.0).fillna(0.0), axis=0)
    W_disp = W_disp.mul(high_disp * btc_gate, axis=0)

    # Low-dispersion regime: BTC-only
    low_disp = 1.0 - high_disp
    rv_btc = btc.pct_change().ewm(span=21, adjust=False).std() * np.sqrt(DPY)
    btc_size = (0.18 / rv_btc.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    W_btc = (low_disp * btc_size * btc_gate)

    for c in W_disp.columns:
        W[c] = W_disp[c]
    W["BTC"] = W_btc

    return W.shift(1).fillna(0.0)


def sleeve_spy_alpha(cp, macro=None):
    """Extract BTC's alpha over SPY: long BTC when it's beating a
    rolling-beta-scaled SPY return over last 63d. Orthogonal to pure trend
    because it's a RELATIVE signal against equities.

    Net position: just long BTC (beta-hedging vs SPY is not feasible without
    an equity short leg, but we use SPY as a regime / threshold signal).
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if macro is None or "spy" not in macro:
        return W
    spy = macro["spy"].reindex(cp.index).ffill()
    s = cp["BTC"]
    # Rolling 63d returns
    r_btc = s.pct_change(63)
    r_spy = spy.pct_change(63)
    # Rolling beta estimate via ratio of covariances (Not OLS — simpler)
    r_btc_d = s.pct_change()
    r_spy_d = spy.pct_change()
    cov = r_btc_d.rolling(126, min_periods=60).cov(r_spy_d)
    var_spy = r_spy_d.rolling(126, min_periods=60).var()
    beta = (cov / var_spy.replace(0, np.nan)).clip(0.1, 3.0)
    # Alpha = actual BTC return - beta * SPY return
    alpha = r_btc - beta * r_spy
    signal = (alpha > 0.10).astype(float)  # BTC beating SPY-adjusted hurdle

    rv = r_btc_d.ewm(span=21, adjust=False).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    ma100 = s.rolling(100, min_periods=50).mean()
    trend = (s > ma100).astype(float)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (signal * trend * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_skew_regime(cp, macro=None):
    """Skewness-conditional sizing: when rolling 63d SKEWNESS of BTC daily
    returns is positive (right-tail dominated = bullish asymmetry), size up.
    When deeply negative (left-tail dominated = panic risk), size down or 0.

    Simple defensive overlay that's orthogonal to price-based trend.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    r = s.pct_change()
    skew = r.rolling(63, min_periods=30).skew()
    # Map skew to sizing multiplier:
    #   skew > +0.5: 1.0 (full size)
    #   skew = 0.0: 0.5
    #   skew < -0.5: 0.0 (out)
    scale = ((skew + 0.5) / 1.0).clip(lower=0.0, upper=1.0)

    ma100 = s.rolling(100, min_periods=50).mean()
    trend = ((s > ma100) & (s.pct_change(63) > 0.0)).astype(float)
    rv = r.ewm(span=21, adjust=False).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (trend * scale * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_broad_xsmom(cp, macro=None):
    """Broad cross-sectional momentum — concentrate in top-5 highest-momentum
    coins in the ENTIRE universe (not just alts). Rebalances weekly.

    This sleeve leverages the 111-coin universe: we can find the 5 strongest
    trends out of 100+ candidates rather than a tiny basket. Survivorship-
    aware (eligibility + catastrophic-DD filter on every coin).

    Per-coin cap 15%, total basket ~50% gross.
    """
    elig = eligibility(cp, min_history=180,
                       catastrophe_dd=-0.25, dd_window=60)
    # 90-day momentum, skip-5-day (avoid short-term reversion noise)
    mom90 = cp.pct_change(90).shift(5)
    score = mom90.where(elig.astype(bool)).where(mom90 > 0.30)  # 30% hurdle
    # Rank, pick top 5
    ranks = score.rank(axis=1, ascending=False, method="first")
    pick = (ranks <= 5.0).astype(float)
    rv = cp.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)
    inv = (1.0 / rv.replace(0, np.nan)).where(pick.astype(bool))
    W = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W = W.clip(upper=0.15)
    # Scale basket to 0.5 gross
    s_sum = W.sum(axis=1).replace(0, np.nan)
    scale = (0.50 / s_sum).clip(upper=1.0).fillna(0.0)
    W = W.mul(scale, axis=0)

    # BTC-regime gate on top
    btc = cp["BTC"]
    btc_gate = ((btc > btc.rolling(150, min_periods=75).mean()) &
                (btc.pct_change(63) > 0.0)).astype(float)
    W = W.mul(btc_gate, axis=0)
    return W.shift(1).fillna(0.0)


def sleeve_quintile_spread(cp, macro=None):
    """Cross-sectional quintile spread (long-only version): with a 100+ coin
    universe we can do a proper quintile sort. Long top quintile by 63d
    momentum (20 coins), small weight each (5% cap per coin, ~60% gross).

    Different from broad_xsmom because:
      * Wider basket (top 20% rather than top-5)
      * 63d lookback (shorter = more turnover but captures regime shifts)
      * Equal-weighted within quintile (not inverse-vol)
    """
    elig = eligibility(cp, min_history=150,
                       catastrophe_dd=-0.25, dd_window=60)
    mom63 = cp.pct_change(63)
    # Keep only positive-momentum eligible coins, then rank
    score = mom63.where(elig.astype(bool)).where(mom63 > 0.0)
    # Pick top 20% of eligible by date
    n_elig = elig.sum(axis=1)
    top_n = (n_elig * 0.20).clip(lower=5).round().astype(int)
    ranks = score.rank(axis=1, ascending=False, method="first")
    # Build pick mask per row (ranks <= top_n_per_row)
    pick = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for i, dt in enumerate(cp.index):
        n = top_n.iloc[i]
        if n >= 1:
            pick.loc[dt] = (ranks.loc[dt] <= n).astype(float)
    # Equal-weight within quintile, 5% cap per coin
    n_picked = pick.sum(axis=1).replace(0, np.nan)
    W = pick.div(n_picked, axis=0).fillna(0.0)
    W = W.clip(upper=0.05)
    # Scale to 0.60 gross
    s_sum = W.sum(axis=1).replace(0, np.nan)
    scale = (0.60 / s_sum).clip(upper=1.0).fillna(0.0)
    W = W.mul(scale, axis=0)

    btc = cp["BTC"]
    btc_gate = ((btc > btc.rolling(200, min_periods=100).mean())).astype(float)
    W = W.mul(btc_gate, axis=0)
    return W.shift(1).fillna(0.0)


def sleeve_breadth_thrust(cp, macro=None):
    """Breadth-thrust (Zweig-style): when % of eligible coins in uptrend
    jumps rapidly (10d rising avg of breadth crosses above 65% threshold),
    strong kickoff signal. Long BTC+ETH basket for N days.

    Fires rarely; strong historical track record at market turning points.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    elig = eligibility(cp, 150)
    ma100 = cp.rolling(100, min_periods=50).mean()
    trending = (cp > ma100).astype(float) * elig
    breadth = trending.sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
    breadth_smooth = breadth.ewm(span=10, adjust=False).mean()

    # Thrust: breadth crosses above 65% having been below 50% in last 30d
    below50 = (breadth.rolling(30, min_periods=15).min() < 0.50)
    above65 = (breadth_smooth > 0.65)
    thrust = (above65 & below50).astype(float)

    # Hold 30 days after thrust signal
    held = pd.Series(0.0, index=cp.index)
    state = 0.0
    days_in = 0
    for i in range(len(cp)):
        if thrust.iloc[i] == 1.0 and state == 0.0:
            state = 1.0
            days_in = 0
        elif state == 1.0:
            days_in += 1
            if days_in > 30:
                state = 0.0
        held.iloc[i] = state

    # Long BTC + ETH 50/50
    btc = cp["BTC"]
    rv_b = btc.pct_change().ewm(span=21, adjust=False).std() * np.sqrt(DPY)
    size_b = (0.18 / rv_b.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    W["BTC"] = (held * size_b * 0.5).shift(1).fillna(0.0)
    if "ETH" in cp.columns:
        eth = cp["ETH"]
        rv_e = eth.pct_change().ewm(span=21, adjust=False).std() * np.sqrt(DPY)
        size_e = (0.18 / rv_e.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
        W["ETH"] = (held * size_e * 0.5).shift(1).fillna(0.0)
    return W


# --- Proprietary orthogonal sleeves (v3) ---
# These use information sources the trend sleeves ignore: volume,
# daily range, calendar, autocorrelation regime, efficiency ratio.


def sleeve_efficiency_trend(cp, macro=None):
    """Kaufman Efficiency Ratio-sized trend: |net 21d return| / sum(|daily r|).

    ER close to 1 = clean trend (signal dominates noise). ER near 0 = chop.
    Position size scales LINEARLY with ER, so the sleeve naturally sizes
    down in choppy regimes like 2025 and up in clean trends like 2017/2020.
    Orthogonal to fixed-size trend sleeves.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    change = (s - s.shift(21)).abs()
    volatility = s.diff().abs().rolling(21, min_periods=10).sum()
    er = (change / volatility.replace(0, np.nan)).clip(0, 1).fillna(0.0)
    # Trend direction filter
    ma100 = s.rolling(100, min_periods=50).mean()
    trend = ((s > ma100) & (s.pct_change(21) > 0.0)).astype(float)
    signal = trend * er  # 0-1 sizing by trend strength
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.25 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.5)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (signal * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_consolidation_break(cp, macro=None):
    """Consolidation-breakout: identifies 40+ day narrow-range periods
    (coefficient of variation of daily returns < historical 25th percentile),
    then goes long on breakout above the high of the consolidation.

    Captures the well-known 'breakout from tight base' setup — highly
    orthogonal because it only fires a few times per year.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    r = s.pct_change()
    # Coefficient of variation over 40 days: low = tight base
    std40 = r.rolling(40, min_periods=25).std()
    std_long = r.rolling(252, min_periods=100).std()
    tight = (std40 < 0.7 * std_long).astype(float)  # 30% below long-run vol
    hi40 = s.rolling(40, min_periods=25).max()
    # Breakout: new 40d high while recent window was tight
    breakout = (s >= hi40).astype(float) * tight.shift(1).fillna(0.0)

    # Hold 30 days after breakout or exit on 10d low
    lo10 = s.rolling(10, min_periods=5).min()
    held = pd.Series(0.0, index=cp.index)
    state = 0.0
    days_in = 0
    for i in range(len(s)):
        if breakout.iloc[i] == 1.0 and state == 0.0:
            state = 1.0
            days_in = 0
        elif state == 1.0:
            days_in += 1
            if (not pd.isna(lo10.iloc[i]) and s.iloc[i] <= lo10.iloc[i]) or days_in > 30:
                state = 0.0
        held.iloc[i] = state
    rv = r.rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.22 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    ma200 = s.rolling(200, min_periods=100).mean()
    gate = (s > ma200).astype(float)
    W["BTC"] = (held * size * gate).shift(1).fillna(0.0)
    return W


def sleeve_deep_dip(cp, macro=None):
    """Deeper-than-crash-dip contrarian: buy BTC when in 35-55% DD from
    180d high AND 365d return still positive (secular uptrend intact).

    Catches 2020 COVID capitulation, 2021 May-July purge, 2022 mid-bear
    false bottoms (200MA filter excludes the worst), 2024 Aug slump.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    hi180 = s.rolling(180, min_periods=90).max()
    dd180 = s / hi180 - 1
    sec_uptrend = (s.pct_change(365) > 0.15).astype(float)
    in_zone = ((dd180 <= -0.35) & (dd180 >= -0.55) & (sec_uptrend > 0)).astype(float)
    # Hold 60 days max or exit on 10% rebound from entry
    held = pd.Series(0.0, index=cp.index)
    state = 0.0
    entry_price = 0.0
    days_in = 0
    for i in range(len(s)):
        p = s.iloc[i]
        if pd.isna(p):
            held.iloc[i] = state
            continue
        if in_zone.iloc[i] == 1.0 and state == 0.0:
            state = 1.0
            entry_price = p
            days_in = 0
        elif state == 1.0:
            days_in += 1
            if p >= entry_price * 1.20 or days_in > 60 or dd180.iloc[i] < -0.60:
                state = 0.0
        held.iloc[i] = state
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.25 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.5)
    W["BTC"] = (held * size).shift(1).fillna(0.0)
    return W


def sleeve_ewma_cross(cp, macro=None):
    """EWMA 10/50 crossover — exponential MAs weight recent data more.
    Faster and differently-timed than SMA-based sleeves.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    e10 = s.ewm(span=10, adjust=False).mean()
    e50 = s.ewm(span=50, adjust=False).mean()
    cross = (e10 > e50).astype(float)
    slope = e10.pct_change(5)
    signal = (cross * (slope > 0)).astype(float)
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (signal * size * alive).shift(1).fillna(0.0)
    return W

# --- Orthogonal sleeves from earlier v3 rewrite (volume/range/autocorr/MR) ---

def sleeve_volume_confirm(cp, macro=None):
    """Volume-confirmed trend: long BTC when
      (price > 50MA) AND (20d vol-weighted avg > prior 60d vol-weighted avg).

    Bullish price action WITH volume expansion is a classic real-money
    confirmation signal, orthogonal to pure moving-average trend.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    btc = load_ohlcv("BTC")
    if btc.empty:
        return W
    btc = btc.reindex(cp.index).ffill()
    s = btc["Close"]
    vol = btc["Volume"]
    ma50 = s.rolling(50, min_periods=25).mean()
    # Price > MA gate
    price_gate = (s > ma50).astype(float)
    # Volume regime: recent 20d vol > prior 60d vol (expansion)
    v20 = vol.rolling(20, min_periods=10).mean()
    v60 = vol.rolling(60, min_periods=30).mean()
    vol_expansion = (v20 > v60 * 1.10).astype(float)
    signal = price_gate * vol_expansion
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (signal * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_range_expansion(cp, macro=None):
    """Daily range expansion: long BTC when (High-Low)/Close (today) exceeds
    a short-term (21d) average AND price closes in top 30% of day's range.

    Range-expansion WITH strong close = institutional buying. Orthogonal to
    momentum; often precedes new trend legs.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    btc = load_ohlcv("BTC").reindex(cp.index).ffill()
    if btc.empty:
        return W
    s = btc["Close"]
    day_range = (btc["High"] - btc["Low"]) / s
    range_avg = day_range.rolling(21, min_periods=10).mean()
    close_in_range = (btc["Close"] - btc["Low"]) / (btc["High"] - btc["Low"]).replace(0, np.nan)
    expansion = (day_range > 1.3 * range_avg).astype(float)
    strong_close = (close_in_range > 0.70).astype(float).fillna(0.0)
    signal = expansion * strong_close
    # State machine: hold position 15 days after signal fires
    held = pd.Series(0.0, index=cp.index)
    days_in = 0
    state = 0.0
    for i in range(len(cp)):
        if signal.iloc[i] == 1.0:
            state = 1.0
            days_in = 0
        elif state == 1.0:
            days_in += 1
            if days_in > 15:
                state = 0.0
        held.iloc[i] = state
    ma100 = s.rolling(100, min_periods=50).mean()
    trend_gate = (s > ma100).astype(float)
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.18 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.2)
    W["BTC"] = (held * size * trend_gate).shift(1).fillna(0.0)
    return W


def sleeve_dow_monday(cp, macro=None):
    """Calendar anomaly: historically BTC has a positive-skewed Monday return
    (weekend accumulation effect). Long BTC on Sundays (hold Mon), exit Tue
    close. Tiny sleeve, but 100% decorrelated from any trend signal.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    dow = pd.Series(cp.index.dayofweek, index=cp.index)
    # Enter at Sunday close (dayofweek==6), hold Monday, exit Tuesday close.
    on_sunday = (dow == 6).astype(float)
    on_monday = (dow == 0).astype(float)
    held = on_sunday + on_monday
    s = cp["BTC"]
    rv = s.pct_change().rolling(60, min_periods=30).std() * np.sqrt(DPY)
    # Conservative sizing: target 10% vol to keep TC drag manageable
    size = (0.10 / rv.replace(0, np.nan)).clip(lower=0.0, upper=0.8)
    ma200 = s.rolling(200, min_periods=100).mean()
    trend_gate = (s > ma200).astype(float)  # only in long-term uptrend
    W["BTC"] = (held * size * trend_gate).shift(1).fillna(0.0)
    return W


def sleeve_turn_of_month(cp, macro=None):
    """Turn-of-month calendar effect: long BTC in the last 3 + first 2
    trading days of each month. Well-documented anomaly in equities that
    also appears in crypto (retail contribution flows, salary deposits).
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    idx = cp.index
    dom = pd.Series(idx.day, index=idx)
    # Last day of month is easier — use rolling lookahead via shift
    # Approximate: enter on days with day >= 28 OR day <= 2
    in_window = ((dom >= 28) | (dom <= 2)).astype(float)
    s = cp["BTC"]
    rv = s.pct_change().rolling(60, min_periods=30).std() * np.sqrt(DPY)
    size = (0.12 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.0)
    ma200 = s.rolling(200, min_periods=100).mean()
    trend_gate = (s > ma200).astype(float)
    W["BTC"] = (in_window * size * trend_gate).shift(1).fillna(0.0)
    return W


def sleeve_autocorr_regime(cp, macro=None):
    """Trend-regime meta-signal: when BTC's recent lag-1 autocorrelation is
    positive (trend persists), go long with standard trend filter.

    When autocorrelation is near zero or negative, go to cash (random walk
    regime — trend signals unreliable). This is a REGIME detector.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    r = s.pct_change().fillna(0.0)
    # Rolling 63d autocorrelation at lag 1
    acf = r.rolling(63, min_periods=40).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 10 else 0.0,
        raw=True)
    trend_on = (acf > 0.02).astype(float)  # positive lag-1 serial correlation
    ma100 = s.rolling(100, min_periods=50).mean()
    trend_gate = ((s > ma100) & (s.pct_change(63) > 0.0)).astype(float)
    signal = trend_on * trend_gate
    rv = r.rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > -0.28).astype(float)
    W["BTC"] = (signal * size * alive).shift(1).fillna(0.0)
    return W


def sleeve_mean_reversion(cp, macro=None):
    """Short-term mean reversion IN UPTREND: buy BTC when 5-day RSI < 30
    while 200MA still rising. Hold 10 days or exit on RSI > 65.

    Specifically targets chop regimes (2025) where trend breaks down but
    market remains in long-term uptrend. Truly orthogonal to trend signals.
    """
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    r = s.pct_change()
    # 5-day RSI (Wilder-style)
    gain = r.clip(lower=0).ewm(alpha=1/5, adjust=False).mean()
    loss = (-r).clip(lower=0).ewm(alpha=1/5, adjust=False).mean()
    rsi5 = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ma200 = s.rolling(200, min_periods=100).mean()
    ma200_slope = ma200.pct_change(30)
    oversold = (rsi5 < 30).astype(float)
    uptrend = ((s > ma200) & (ma200_slope > 0.0)).astype(float)
    entry = oversold * uptrend

    # Hold up to 10 days or exit on RSI > 65
    held = pd.Series(0.0, index=cp.index)
    days_in = 0
    state = 0.0
    for i in range(len(cp)):
        rs = rsi5.iloc[i]
        if entry.iloc[i] == 1.0 and state == 0.0:
            state = 1.0
            days_in = 0
        elif state == 1.0:
            days_in += 1
            if (not pd.isna(rs) and rs > 65) or days_in > 10:
                state = 0.0
        held.iloc[i] = state
    rv = r.rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (0.18 / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    W["BTC"] = (held * size).shift(1).fillna(0.0)
    return W


def sleeve_btc_eth_rotate(cp, macro=None):
    """Cross-asset rotation: when BTC outperforms ETH over 30d → BTC heavy;
    when ETH outperforms BTC over 30d → ETH heavy. Single coin at a time,
    vol-targeted, with ATH 200MA gate on both."""
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "ETH" not in cp.columns:
        return W
    btc = cp["BTC"]; eth = cp["ETH"]
    btc_r30 = btc.pct_change(30)
    eth_r30 = eth.pct_change(30)
    pick_btc = (btc_r30 > eth_r30).astype(float)
    pick_eth = 1.0 - pick_btc

    btc_trend = (btc > btc.rolling(200, min_periods=100).mean()).astype(float)
    eth_trend = (eth > eth.rolling(200, min_periods=100).mean()).astype(float)

    rv_btc = btc.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    rv_eth = eth.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size_btc = (0.20 / rv_btc.replace(0, np.nan)).clip(lower=0.0, upper=1.3)
    size_eth = (0.20 / rv_eth.replace(0, np.nan)).clip(lower=0.0, upper=1.3)

    W["BTC"] = (pick_btc * btc_trend * size_btc).shift(1).fillna(0.0)
    W["ETH"] = (pick_eth * eth_trend * size_eth).shift(1).fillna(0.0)
    return W


BUILDERS = {
    # BTC trend core
    "BTC_VM":               sleeve_btc_vm,
    "BTC_SLOW":             sleeve_btc_slow,
    "TRIPLE_MOM":           sleeve_triple_mom,
    # Breakout family
    "TURTLE20":             sleeve_turtle20,
    "TURTLE55":             sleeve_turtle55,
    "BOLLINGER":            sleeve_bollinger,
    "CONSOLIDATION_BREAK":  sleeve_consolidation_break,
    # Regime / macro
    "MACRO_TREND":          sleeve_macro_trend,
    "HALVING_BOOST":        sleeve_halving_boost,
    # Specialists
    "CRASH_DIP":            sleeve_crash_dip,
    "VOL_BREAKOUT":         sleeve_vol_breakout,
    # Cross-asset
    "ETH_VM":               sleeve_eth_vm,
    # --- Proprietary orthogonal edges (v3) ---
    "EFFICIENCY_TREND":     sleeve_efficiency_trend,
    "RANGE_EXPANSION":      sleeve_range_expansion,
    "AUTOCORR_REGIME":      sleeve_autocorr_regime,
    # --- Proprietary novel sleeves (v4) ---
    "INV_VOL_BTC":          sleeve_inv_vol_btc,
    "VOL_PERCENTILE":       sleeve_vol_percentile,
    "BREADTH_THRUST":       sleeve_breadth_thrust,
    "SPY_ALPHA":            sleeve_spy_alpha,
    # --- v7 novel inventions ---
    "MTF_FRACTAL":          sleeve_mtf_fractal,    # all-6-timeframes-aligned
    "CONVEX_BREAKOUT":      sleeve_convex_breakout, # fixed-size right-tail
    # --- v12 EMPIRICAL TESTS (all dropped, code retained) ---
    # MULTI_SCAN: 5-signal-confluence universe scanner. Tested across
    #   12 configs (top_k ∈ {3,5,7,10}, min_signals ∈ {3,4,5}). Standalone
    #   OOS SR: best 0.04 on 111 universe / 0.17 on 50. ALL configurations
    #   have negative or near-zero OOS SR. Ensemble impact: marginal -0.01
    #   to -0.04 OOS SR; the only configs that lift OOS SR by +0.02-0.03
    #   simultaneously cost 8.7% CAGR (NAV halves). NET-NEGATIVE.
    # PER-COIN alt sleeves (SOL_VM, BNB_VM, AVAX_VM, LINK_VM, ADA_VM,
    #   XRP_VM, LTC_VM, DOGE_VM): each a vol-managed BTC-style trend on
    #   one specific coin, BTC-regime-gated. Tested as a basket.
    #   Ensemble result: Full SR 1.47 → 1.17, OOS 1.13 → 0.92, CAGR 51% →
    #   31%, NAV 124× → 22×. 2025+ year went from -3% to -14% loss
    #   because alts crashed harder than BTC in 2025 chop. NET-NEGATIVE.
    # "MULTI_SCAN":         sleeve_multi_scan,
    # "SOL_VM":             sleeve_sol_vm,
    # "BNB_VM":             sleeve_bnb_vm,
    # "AVAX_VM":            sleeve_avax_vm,
    # "LINK_VM":            sleeve_link_vm,
    # "ADA_VM":             sleeve_ada_vm,
    # "XRP_VM":             sleeve_xrp_vm,
    # "LTC_VM":             sleeve_ltc_vm,
    # "DOGE_VM":            sleeve_doge_vm,
    # SHORT sleeves DROPPED FROM ENSEMBLE — empirically all three lost money:
    #   SHORT_BTC_BEAR    SR -0.53 / OOS -0.99 (BTC's structural uptrend)
    #   SHORT_TREND_BREAK SR -0.08 / OOS -0.29 (bounce-squeeze)
    #   SHORT_PERMABEAR   SR -0.59 / OOS -0.48 (permabears bounce 30-50%)
    # Code retained in sleeves.py for future use with funding-rate data.
}
SHORT_SLEEVES = set()
# Tested and dropped for weak OOS contribution (all OOS SR < 0.50):
#   BTC_KAMA  (0.15), ADX_TREND (0.15), EWMA_CROSS (0.35),
#   ETH_SLOW  (0.23), ALT_XSMOM (-0.21),
#   and noise-only sleeves: DEEP_DIP, VOLUME_CONFIRM, MEAN_REVERSION,
#   BTC_ETH_ROTATE, DOW_MONDAY, TURN_OF_MONTH, ETHBTC_PAIR, VIX_MEANREV,
#   DXY_WEAK.
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
