"""HYDRA sleeves v3 — orthogonal alpha sources across 8 categories.

v2 had 15 sleeves but 8 were variations of "long equity when trending",
producing 0.35 mean pairwise correlation. v3 restructures into categories
with at most 2 sleeves per category:

  Equity trend     (2): vol-contingent SPY, sector top-3 momentum
  Fixed income     (3): bond regime, credit trend, yield-curve carry
  Commodity        (2): DBC trend, gold-silver ratio regime
  Currency (FX)    (2): FX carry (DBV), dollar trend
  Volatility       (2): VIX contango short, VIX spike fade
  Crypto           (1): BTC trend (post-2015)
  Cross-asset      (1): absolute momentum on 6 assets
  Alternative      (2): turn-of-month, defensive rotation (low-breadth)

15 sleeves. Target: mean |corr| < 0.15.
"""
import numpy as np
import pandas as pd

from hydra_core import (load_etf, load_fred, vol_target, apply_tc,
                        VOL_TARGET, VOL_LOOKBACK, TC_BPS)


def monthly_mask(dates):
    s = pd.Series(dates, index=dates)
    return s.dt.to_period("M") != s.dt.to_period("M").shift(1)


def rebal(w, mask):
    return w.where(mask.reindex(w.index).fillna(False)).ffill().fillna(0)


# ==================== EQUITY TREND (2) ====================

def s1_vol_contingent_spy(dates):
    """Long SPY when SPY>200dma & VIX<25; else SHY."""
    spy = load_etf("SPY").reindex(dates).ffill()
    shy = load_etf("SHY").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    ma = spy.rolling(200).mean()
    sig = ((spy > ma) & (vix < 25)).shift(1).fillna(False).astype(float)
    r_s = spy.pct_change().fillna(0); r_b = shy.pct_change().fillna(0)
    w = pd.DataFrame({"SPY": sig, "SHY": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["SPY"] * r_s + w["SHY"] * r_b
    net = apply_tc(w, gross, dates)
    ret, _ = vol_target(net)
    return ret.rename("s1_eq_regime")


def s2_sector_top3(dates):
    """Top-3 of 9 sectors by 6m momentum, long-only, monthly rebal.
    Requires positive momentum, else cash."""
    tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB"]
    px = pd.DataFrame({t: load_etf(t) for t in tickers}).reindex(dates).ffill()
    ret = px.pct_change().fillna(0)
    mom = (px / px.shift(126) - 1).shift(1)
    rank = mom.rank(axis=1, ascending=False)
    w_long = ((rank <= 3) & (mom > 0)).astype(float) * (1.0 / 3.0)
    w_long = rebal(w_long, monthly_mask(dates))
    bil = load_etf("BIL").reindex(dates).ffill()
    bil_r = bil.pct_change().fillna(0)
    cash = 1 - w_long.sum(axis=1)
    gross = (w_long * ret).sum(axis=1) + cash * bil_r
    net = apply_tc(w_long, gross, ret.index)
    r, _ = vol_target(net)
    return r.rename("s2_sector_top3")


# ==================== FIXED INCOME (3) ====================

def s3_bond_duration(dates):
    """Long TLT when 10y 6m yield trend < 0; else SHY."""
    yld = load_fred("DGS10").reindex(dates).ffill()
    tlt = load_etf("TLT").reindex(dates).ffill()
    shy = load_etf("SHY").reindex(dates).ffill()
    trend = (yld - yld.shift(126)).shift(1)
    sig = (trend < 0).astype(float)
    w = pd.DataFrame({"TLT": sig, "SHY": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["TLT"] * tlt.pct_change().fillna(0) + w["SHY"] * shy.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s3_bond_dur")


def s4_credit_trend(dates):
    """HYG regime: long when HYG>200dma & HY OAS below its 63d mean."""
    hyg = load_etf("HYG").reindex(dates).ffill()
    shy = load_etf("SHY").reindex(dates).ffill()
    oas = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    trend_ok = hyg > hyg.rolling(200).mean()
    oas_ok = oas < oas.rolling(63).mean()
    sig = (trend_ok & oas_ok).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"HYG": sig, "SHY": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["HYG"] * hyg.pct_change().fillna(0) + w["SHY"] * shy.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s4_credit")


def s5_curve_carry(dates):
    """Long IEF when curve (T10Y3M) > +1% (steep = carry trade): earn roll-down.
    Flat/inverted: short IEF (rates expected to fall to inversion point? actually
    just go to SHY defensively)."""
    slope = load_fred("T10Y3M").reindex(dates).ffill().shift(1)
    ief = load_etf("IEF").reindex(dates).ffill()
    shy = load_etf("SHY").reindex(dates).ffill()
    sig = (slope > 1.0).astype(float)
    w = pd.DataFrame({"IEF": sig, "SHY": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["IEF"] * ief.pct_change().fillna(0) + w["SHY"] * shy.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s5_curve_carry")


# ==================== COMMODITY (2) ====================

def s6_commodity_trend(dates):
    """Long DBC when DBC>200dma. Captures commodity cycles."""
    dbc = load_etf("DBC").reindex(dates).ffill()
    bil = load_etf("BIL").reindex(dates).ffill()
    ma = dbc.rolling(200).mean()
    sig = (dbc > ma).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"DBC": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["DBC"] * dbc.pct_change().fillna(0) + w["BIL"] * bil.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s6_cmdty")


def s7_gold_silver_regime(dates):
    """Gold/silver ratio momentum: when GLD/SLV ratio rising for 3m (risk-off),
    long GLD; when falling (risk-on), long SLV. This is regime momentum, not
    mean-reversion."""
    gld = load_etf("GLD").reindex(dates).ffill()
    slv = load_etf("SLV").reindex(dates).ffill()
    ratio = gld / slv
    mom = (ratio / ratio.shift(63) - 1).shift(1)
    sig = (mom > 0).astype(float)
    r_g = gld.pct_change().fillna(0); r_s = slv.pct_change().fillna(0)
    w = pd.DataFrame({"GLD": sig, "SLV": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["GLD"] * r_g + w["SLV"] * r_s
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s7_gld_slv")


# ==================== CURRENCY (2) ====================

def s8_safe_haven_jpy(dates):
    """Long FXY (Yen) as crisis safe-haven: activate when VIX 10d avg > 22
    (elevated-vol regime). Yen rallies during equity risk-off episodes —
    genuinely negative-beta payoff (2008, 2015, 2020). Cash otherwise."""
    fxy = load_etf("FXY")
    bil = load_etf("BIL").reindex(dates).ffill()
    vix = load_fred("VIXCLS")
    if fxy is None or vix is None:
        return pd.Series(0.0, index=dates).rename("s8_fxy_sh")
    f = fxy.reindex(dates).ffill()
    v = vix.reindex(dates).ffill().rolling(10).mean()
    live = f.notna()
    sig = ((v > 22) & live).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"FXY": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["FXY"] * f.pct_change().fillna(0) + w["BIL"] * bil.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s8_fxy_sh")


def s9_dollar_regime(dates):
    """UUP (long USD) momentum. Long when UUP 6m trend > 0."""
    uup = load_etf("UUP")
    bil = load_etf("BIL").reindex(dates).ffill()
    if uup is None: return pd.Series(0.0, index=dates).rename("s9_usd_reg")
    u = uup.reindex(dates).ffill()
    mom = (u / u.shift(126) - 1).shift(1)
    sig = (mom > 0).fillna(False).astype(float)
    live = u.notna().astype(float)
    sig = sig * live
    w = pd.DataFrame({"UUP": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["UUP"] * u.pct_change().fillna(0) + w["BIL"] * bil.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s9_usd_reg")


# ==================== VOLATILITY (2) ====================

def s10_vix_carry(dates):
    """Short VIXY when VIX > realized SPY vol by 3pts AND VIX < 30."""
    vixy = load_etf("VIXY")
    spy = load_etf("SPY")
    vix = load_fred("VIXCLS")
    if vixy is None: return pd.Series(0.0, index=dates).rename("s10_vix_carry")
    px = vixy.reindex(dates).ffill()
    spy_px = spy.reindex(dates).ffill()
    vix_d = vix.reindex(dates).ffill()
    rv = spy_px.pct_change().rolling(20).std() * np.sqrt(252) * 100
    contango = (vix_d - rv).shift(1)
    sig = ((contango > 3) & (vix_d.shift(1) < 30)).astype(float) * (-1.0)
    ret = px.pct_change().fillna(0)
    live = px.notna().astype(float)
    w = (sig * live).to_frame("VIXY")
    gross = w["VIXY"] * ret
    net = apply_tc(w, gross, ret.index)
    r, _ = vol_target(net)
    return r.rename("s10_vix_carry")


def s11_vix_spike_fade(dates):
    """When VIX > 30 AND SPY trending down: long UVXY for 5 days (crisis hedge).
    When VIX spike reverses (VIX dropped >20% from peak): exit.
    This is LONG vol, a tail hedge that we accept will lose most of the time
    but pays off in crises. Diversifies with the short-vol carry."""
    uvxy = load_etf("UVXY")
    spy = load_etf("SPY")
    vix = load_fred("VIXCLS")
    if uvxy is None: return pd.Series(0.0, index=dates).rename("s11_vix_spike")
    u = uvxy.reindex(dates).ffill()
    spy_px = spy.reindex(dates).ffill()
    vix_d = vix.reindex(dates).ffill()
    spy_ma = spy_px.rolling(20).mean()
    # Signal: enter long UVXY when VIX crosses above 25 and SPY below 20d MA
    enter = (vix_d > 25) & (spy_px < spy_ma)
    sig = enter.shift(1).fillna(False).astype(float)
    live = u.notna().astype(float)
    w = (sig * live).to_frame("UVXY")
    # Rebalance weekly to limit turnover
    w = rebal(w, monthly_mask(dates))
    gross = w["UVXY"] * u.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s11_vix_spike")


# ==================== CRYPTO (1) ====================

def s12_btc_trend(dates):
    """BTC_USD when BTC > 200dma, else cash. Crypto is genuinely uncorrelated
    with equity factors."""
    btc = load_etf("BTC_USD")
    bil = load_etf("BIL").reindex(dates).ffill()
    if btc is None: return pd.Series(0.0, index=dates).rename("s12_btc")
    b = btc.reindex(dates).ffill()
    ma = b.rolling(200).mean()
    sig = (b > ma).shift(1).fillna(False).astype(float)
    live = b.notna().astype(float)
    sig = sig * live
    w = pd.DataFrame({"BTC": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["BTC"] * b.pct_change().fillna(0) + w["BIL"] * bil.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s12_btc")


# ==================== CROSS-ASSET (1) ====================

def s13_cross_asset_ensemble(dates):
    """Classic GEM-style: SPY if SPY 12m > TLT 12m & > 0 else TLT if TLT 12m > 0 else cash.
    """
    spy = load_etf("SPY").reindex(dates).ffill()
    tlt = load_etf("TLT").reindex(dates).ffill()
    bil = load_etf("BIL").reindex(dates).ffill()
    m_s = (spy / spy.shift(252) - 1).shift(1)
    m_t = (tlt / tlt.shift(252) - 1).shift(1)
    pick_spy = ((m_s > m_t) & (m_s > 0)).astype(float)
    pick_tlt = ((m_s <= m_t) & (m_t > 0)).astype(float)
    pick_bil = 1 - pick_spy - pick_tlt
    w = pd.DataFrame({"SPY": pick_spy, "TLT": pick_tlt, "BIL": pick_bil})
    w = rebal(w, monthly_mask(dates))
    gross = (w["SPY"] * spy.pct_change().fillna(0) +
             w["TLT"] * tlt.pct_change().fillna(0) +
             w["BIL"] * bil.pct_change().fillna(0))
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s13_xa_gem")


# ==================== ALTERNATIVE (2) ====================

def s14_turn_of_month(dates):
    """Long SPY on last 3 trading days of month + first 3 of next; cash otherwise.
    Well-documented calendar anomaly."""
    spy = load_etf("SPY").reindex(dates).ffill()
    bil = load_etf("BIL").reindex(dates).ffill()
    r_s = spy.pct_change().fillna(0); r_b = bil.pct_change().fillna(0)
    s = pd.Series(dates, index=dates)
    # Day-of-month from end: count days remaining in month
    days_left = s.dt.days_in_month - s.dt.day
    day_of_mo = s.dt.day
    is_tom = (days_left < 3) | (day_of_mo <= 3)
    sig = is_tom.shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"SPY": sig, "BIL": 1 - sig})
    # No rebalance masking - signal already discretizes; but TC becomes high (entering/leaving SPY ~24x/yr)
    gross = w["SPY"] * r_s + w["BIL"] * r_b
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s14_tom")


def s15_defensive_rotation(dates):
    """Long XLU/XLP/XLV equal-weight when sector breadth < 40% above 200dma;
    else SHY. Defensive rotation into low-beta sectors when market internals weak."""
    tickers_all = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB"]
    defensive = ["XLU", "XLP", "XLV"]
    px = pd.DataFrame({t: load_etf(t) for t in tickers_all}).reindex(dates).ffill()
    ma = px.rolling(200).mean()
    breadth = (px > ma).astype(float).mean(axis=1).shift(1)
    # Signal: when breadth is weak (< 40%), go long defensives
    sig = (breadth < 0.4).astype(float)
    shy = load_etf("SHY").reindex(dates).ffill()
    r_shy = shy.pct_change().fillna(0)
    r_def = px[defensive].pct_change().fillna(0).mean(axis=1)
    w_def = pd.DataFrame({t: sig / len(defensive) for t in defensive})
    w_shy = (1 - sig).to_frame("SHY")
    w = pd.concat([w_def, w_shy], axis=1)
    w = rebal(w, monthly_mask(dates))
    gross = (w[defensive] * px[defensive].pct_change().fillna(0)).sum(axis=1) + w["SHY"] * r_shy
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s15_defensive")


def s16_tlt_meanrev(dates):
    """TLT 5-day mean reversion: long TLT when TLT 5d return < -3% AND
    10y yield has spiked above 63d mean. Hold 5 days. Captures bond
    oversold bounces, orthogonal to equity signals."""
    tlt = load_etf("TLT").reindex(dates).ffill()
    bil = load_etf("BIL").reindex(dates).ffill()
    yld = load_fred("DGS10").reindex(dates).ffill()
    r5 = tlt.pct_change(5)
    ytrend = yld > yld.rolling(63).mean()
    trigger = ((r5 < -0.03) & ytrend).shift(1).fillna(False)
    sig_arr = np.zeros(len(dates))
    hold_end = 0
    trig_arr = trigger.values
    for i in range(len(dates)):
        if i < hold_end:
            sig_arr[i] = 1.0
        elif trig_arr[i]:
            sig_arr[i] = 1.0
            hold_end = i + 5
    sig = pd.Series(sig_arr, index=dates)
    r_t = tlt.pct_change().fillna(0); r_b = bil.pct_change().fillna(0)
    w = pd.DataFrame({"TLT": sig, "BIL": 1 - sig})
    gross = w["TLT"] * r_t + w["BIL"] * r_b
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s16_tlt_rev")


def s17_semi_trend(dates):
    """SMH semis when SMH>200dma. Volatile momentum-rich factor."""
    smh = load_etf("SMH"); bil = load_etf("BIL")
    if smh is None: return pd.Series(0.0, index=dates).rename("s17_semi")
    s = smh.reindex(dates).ffill()
    b = bil.reindex(dates).ffill()
    ma = s.rolling(200).mean()
    sig = (s > ma).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"SMH": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["SMH"] * s.pct_change().fillna(0) + w["BIL"] * b.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s17_semi")


def s18_spy_meanrev(dates):
    """Enter SPY long after 5d return < -3% AND VIX > 20. Exit after 5 trading
    days or +2% recovery. Short-horizon mean reversion with regime filter."""
    spy = load_etf("SPY").reindex(dates).ffill()
    bil = load_etf("BIL").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    r5 = spy.pct_change(5)
    trigger = ((r5 < -0.03) & (vix > 20)).shift(1).fillna(False)
    # Hold for 5 trading days
    sig = pd.Series(0.0, index=dates)
    hold_end = 0
    trig_arr = trigger.values
    sig_arr = np.zeros(len(dates))
    for i in range(len(dates)):
        if i < hold_end:
            sig_arr[i] = 1.0
        elif trig_arr[i]:
            sig_arr[i] = 1.0
            hold_end = i + 5
    sig = pd.Series(sig_arr, index=dates)
    r_s = spy.pct_change().fillna(0); r_b = bil.pct_change().fillna(0)
    w = pd.DataFrame({"SPY": sig, "BIL": 1 - sig})
    gross = w["SPY"] * r_s + w["BIL"] * r_b
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s18_spy_rev")


def s19_emerging_mkt(dates):
    """EEM trend: long when EEM > 200dma, else cash. EM is often
    lead-lag with USD regime, partially orthogonal to SPY."""
    eem = load_etf("EEM"); bil = load_etf("BIL")
    if eem is None: return pd.Series(0.0, index=dates).rename("s19_em")
    e = eem.reindex(dates).ffill()
    b = bil.reindex(dates).ffill()
    ma = e.rolling(200).mean()
    sig = (e > ma).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"EEM": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["EEM"] * e.pct_change().fillna(0) + w["BIL"] * b.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s19_em")


def s20_inflation_hedge(dates):
    """TIP vs IEF: long TIP when 10y breakeven (T10YIE) is rising; else IEF.
    Rotates inflation-linked vs nominal duration."""
    tip = load_etf("TIP"); ief = load_etf("IEF")
    be = load_fred("T10YIE")
    if tip is None or ief is None or be is None:
        return pd.Series(0.0, index=dates).rename("s20_infl")
    be_d = be.reindex(dates).ffill()
    be_trend = (be_d - be_d.shift(63)).shift(1)
    sig = (be_trend > 0).astype(float)
    t = tip.reindex(dates).ffill(); i = ief.reindex(dates).ffill()
    r_t = t.pct_change().fillna(0); r_i = i.pct_change().fillna(0)
    w = pd.DataFrame({"TIP": sig, "IEF": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["TIP"] * r_t + w["IEF"] * r_i
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s20_infl")


def s21_eth_trend(dates):
    """ETH trend: long ETH-linked when ETH > 50dma, else cash.
    Separate from BTC — ETH has different drivers (smart-contract demand)."""
    eth = load_etf("ETH_USD")
    bil = load_etf("BIL")
    if eth is None:
        return pd.Series(0.0, index=dates).rename("s21_eth")
    e = eth.reindex(dates).ffill()
    b = bil.reindex(dates).ffill()
    ma = e.rolling(50).mean()
    sig = (e > ma).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"ETH": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["ETH"] * e.pct_change().fillna(0) + w["BIL"] * b.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s21_eth")


def s22_energy_regime(dates):
    """XLE energy regime: long XLE when USO 20d>60d MA AND XLE above 200dma;
    else XLP (staples defence). Energy leads inflation shocks — orthogonal
    to rates/tech regimes."""
    xle = load_etf("XLE"); xlp = load_etf("XLP"); uso = load_etf("USO")
    if xle is None or xlp is None or uso is None:
        return pd.Series(0.0, index=dates).rename("s22_energy")
    xe = xle.reindex(dates).ffill()
    xp = xlp.reindex(dates).ffill()
    o = uso.reindex(dates).ffill()
    oil_mom = (o.rolling(20).mean() > o.rolling(60).mean())
    xle_trend = xe > xe.rolling(200).mean()
    sig = (oil_mom & xle_trend).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"XLE": sig, "XLP": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["XLE"] * xe.pct_change().fillna(0) + w["XLP"] * xp.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s22_energy")


def s23_btal_hedge(dates):
    """BTAL anti-beta: always long when available. BTAL is long low-beta /
    short high-beta — genuinely anti-correlated to SPY drawdowns."""
    btal = load_etf("BTAL"); bil = load_etf("BIL")
    if btal is None:
        return pd.Series(0.0, index=dates).rename("s23_btal")
    bt = btal.reindex(dates).ffill()
    b = bil.reindex(dates).ffill()
    # Sleeve runs only when BTAL data live
    live = bt.notna()
    sig = live.shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"BTAL": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["BTAL"] * bt.pct_change().fillna(0) + w["BIL"] * b.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s23_btal")


def s24_em_bond_carry(dates):
    """EMB EM bond trend: long EMB when > 200dma AND DGS10 trend not sharply
    rising (yields not spiking); else BIL. Captures EM credit carry."""
    emb = load_etf("EMB"); bil = load_etf("BIL")
    yld = load_fred("DGS10")
    if emb is None or yld is None:
        return pd.Series(0.0, index=dates).rename("s24_emb")
    e = emb.reindex(dates).ffill()
    b = bil.reindex(dates).ffill()
    y = yld.reindex(dates).ffill()
    trend = e > e.rolling(200).mean()
    yspike = (y - y.shift(20)) > 0.5  # 10y up 50bp in 20d = stress
    sig = (trend & ~yspike).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"EMB": sig, "BIL": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["EMB"] * e.pct_change().fillna(0) + w["BIL"] * b.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s24_emb")


def s25_reit_regime(dates):
    """IYR REITs: long IYR when IYR > 200dma AND 10y yield < 12m mean;
    else SHY. REITs are rate-sensitive — this captures rate-tailwind regimes."""
    iyr = load_etf("IYR"); shy = load_etf("SHY")
    yld = load_fred("DGS10")
    if iyr is None or yld is None:
        return pd.Series(0.0, index=dates).rename("s25_reit")
    i = iyr.reindex(dates).ffill()
    s = shy.reindex(dates).ffill()
    y = yld.reindex(dates).ffill()
    trend = i > i.rolling(200).mean()
    rate_tw = y < y.rolling(252).mean()
    sig = (trend & rate_tw).shift(1).fillna(False).astype(float)
    w = pd.DataFrame({"IYR": sig, "SHY": 1 - sig})
    w = rebal(w, monthly_mask(dates))
    gross = w["IYR"] * i.pct_change().fillna(0) + w["SHY"] * s.pct_change().fillna(0)
    net = apply_tc(w, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s25_reit")


def s26_sector_ls(dates):
    """Market-neutral sector momentum: long top-2 / short bottom-2 of 9
    sectors by 6m momentum, rebalanced monthly. Dollar-neutral → ~0 beta."""
    tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB"]
    px = pd.DataFrame({t: load_etf(t) for t in tickers}).reindex(dates).ffill()
    ret = px.pct_change().fillna(0)
    mom = (px / px.shift(126) - 1).shift(1)
    rnk = mom.rank(axis=1, ascending=False)
    long_w = (rnk <= 2).astype(float) * 0.5
    short_w = (rnk >= 8).astype(float) * 0.5
    w_net = (long_w - short_w)
    w_net = rebal(w_net, monthly_mask(dates))
    w_abs = w_net.abs()
    gross = (w_net * ret).sum(axis=1)
    net = apply_tc(w_abs, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s26_sec_ls")


def s27_risk_onoff_ls(dates):
    """Long-short cross-asset rotation: rank 6 risk assets (SPY, EFA, EEM,
    HYG, GLD, DBC) by 6m momentum. Long top-2, short bottom-2. Market-neutral
    pair trade capturing regime dispersion."""
    tickers = ["SPY", "EFA", "EEM", "HYG", "GLD", "DBC"]
    px_map = {t: load_etf(t) for t in tickers}
    if any(p is None for p in px_map.values()):
        return pd.Series(0.0, index=dates).rename("s27_xa_ls")
    px = pd.DataFrame(px_map).reindex(dates).ffill()
    ret = px.pct_change().fillna(0)
    mom = (px / px.shift(126) - 1).shift(1)
    rnk = mom.rank(axis=1, ascending=False)
    long_w = (rnk <= 2).astype(float) * 0.5
    short_w = (rnk >= 5).astype(float) * 0.5
    w_net = (long_w - short_w)
    w_net = rebal(w_net, monthly_mask(dates))
    w_abs = w_net.abs()
    gross = (w_net * ret).sum(axis=1)
    net = apply_tc(w_abs, gross, dates)
    r, _ = vol_target(net)
    return r.rename("s27_xa_ls")


def s28_stocks_bonds_ls(dates):
    """Stocks vs bonds long-short: if SPY 6m return > TLT 6m return, long
    SPY short TLT (50/50 dollar); else reverse. Classic asset-class momentum
    spread, market-neutral over a full cycle."""
    spy = load_etf("SPY").reindex(dates).ffill()
    tlt = load_etf("TLT").reindex(dates).ffill()
    sm = (spy / spy.shift(126) - 1).shift(1)
    tm = (tlt / tlt.shift(126) - 1).shift(1)
    sig = (sm > tm).astype(float) * 2 - 1  # +1 or -1
    w_spy = 0.5 * sig
    w_tlt = -0.5 * sig
    w = pd.DataFrame({"SPY": w_spy, "TLT": w_tlt})
    w = rebal(w, monthly_mask(dates))
    r_s = spy.pct_change().fillna(0); r_t = tlt.pct_change().fillna(0)
    gross = w["SPY"] * r_s + w["TLT"] * r_t
    net = apply_tc(w.abs(), gross, dates)
    r, _ = vol_target(net)
    return r.rename("s28_sb_ls")


SLEEVES = [
    s1_vol_contingent_spy, s2_sector_top3,
    s3_bond_duration, s4_credit_trend, s5_curve_carry,
    s6_commodity_trend, s7_gold_silver_regime,
    s8_safe_haven_jpy, s9_dollar_regime,
    s10_vix_carry,
    s12_btc_trend,
    s13_cross_asset_ensemble,
    s15_defensive_rotation,
    s17_semi_trend,
    s18_spy_meanrev, s19_emerging_mkt,
    s20_inflation_hedge,
    s22_energy_regime,
    s24_em_bond_carry,
    s27_risk_onoff_ls,
]
