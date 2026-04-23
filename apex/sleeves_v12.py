"""APEX v12 sleeves — NEW sleeves from 10-agent research.

Key additions:
  SL_DUALBEAR   — 2022-defense macro gate (SPY-TLT corr flip + HY + real yield + CPI)
  SL_REGIME     — Rule-based regime classifier with per-regime allocation
  SL_CALENDAR   — Calendar stack (pre-FOMC + TOM + pre-holiday + Santa)
  SL_VRP        — SVXY when vol-term-structure in contango
  SL_DISPERSION — Dispersion-gated cross-sectional momentum (survives 2022!)
  SL_CPPI       — CPPI floor via SSO + cash (convex upside, floor protection)

All follow the "weights DataFrame (T x N) that sums to <= 1" interface.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import util

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"


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


# ==========================================================================
# DUAL-BEAR SCORE — 2022 defense signal (5-point composite)
# ==========================================================================

def dual_bear_score(cp: pd.DataFrame) -> pd.Series:
    """Returns a 0-5 score. Triggers per agent research:
    1. HY OAS widening >50bp from 252d min AND > 3.50
    2. Real 10Y yield rising sharply (proxy: DGS10 - inflation breakeven)
    3. SPY-TLT 60d correlation > 0
    4. CPI YoY > 5% AND accelerating
    5. DXY up, SPY down, TLT down simultaneously (20d)
    """
    idx = cp.index
    score = pd.Series(0.0, index=idx)

    # 1. HY OAS widening (using HYG/LQD proxy since BAML HY only from 2023)
    hyg = _etf_close("HYG", idx)
    lqd = _etf_close("LQD", idx)
    if not hyg.isna().all() and not lqd.isna().all():
        cr = hyg / lqd   # ratio (FALLS when HY widens)
        cr_252_max = cr.rolling(252, min_periods=30).max()
        widening = (cr_252_max - cr) / cr_252_max > 0.05  # 5%+ below recent high
        score = score + widening.astype(float).fillna(0)

    # 2. Real yield rising (DGS10 - TIP breakeven proxy using 2y/10y change)
    dgs10 = _fred("DGS10", idx)
    dgs10_change_126d = dgs10 - dgs10.rolling(126, min_periods=30).min()
    real_yield_rising = dgs10_change_126d > 0.60   # 60bp rise from 126d low
    score = score + real_yield_rising.astype(float).fillna(0)

    # 3. SPY-TLT correlation flip (KEY 2022 signal)
    spy_r = cp["SPY"].pct_change()
    tlt_r = cp.get("TLT", cp["SPY"]).pct_change()
    corr = spy_r.rolling(60).corr(tlt_r)
    corr_flipped = corr > 0.0
    score = score + corr_flipped.astype(float).fillna(0)

    # 4. CPI hot and accelerating
    cpi = _fred("CPIAUCSL", idx)
    if not cpi.isna().all():
        cpi_yoy = cpi.pct_change(252) * 100   # monthly data forward-filled
        cpi_accel = cpi_yoy > cpi_yoy.shift(63)
        cpi_hot_accel = (cpi_yoy > 5.0) & cpi_accel
        score = score + cpi_hot_accel.astype(float).fillna(0)

    # 5. DXY up + SPY down + TLT down (20d)
    uup = _etf_close("UUP", idx)
    if not uup.isna().all():
        uup_up = uup.pct_change(20) > 0
        spy_down = cp["SPY"].pct_change(20) < 0
        tlt_down = cp.get("TLT", cp["SPY"]).pct_change(20) < 0
        triple = uup_up & spy_down & tlt_down
        score = score + triple.astype(float).fillna(0)

    return score.fillna(0)


def sleeve_dualbear_defense(cp: pd.DataFrame) -> pd.DataFrame:
    """When DBS >= 3: hold defensive basket (UUP + UCO + UGL + cash).
    When DBS < 3: 0 weight (this sleeve sits out; other sleeves handle risk-on).
    """
    dbs = dual_bear_score(cp)
    armed = (dbs >= 3).astype(float).shift(1).fillna(0)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "UUP" in cp.columns:
        W["UUP"] = armed * 0.40
    if "UCO" in cp.columns:
        W["UCO"] = armed * 0.25
    if "UGL" in cp.columns:
        W["UGL"] = armed * 0.15
    # Remaining 20% sits in cash (0 weight means 0 return contribution)
    return _scale_to_vol(W, cp, target_vol=0.15)


# ==========================================================================
# REGIME CLASSIFIER — rule-based, 5 regimes, per-regime allocation
# ==========================================================================

def classify_regime(cp: pd.DataFrame) -> pd.Series:
    """Returns a Series of regime labels: BULL, BEAR, SIDEWAYS, CRISIS, INFLATION."""
    idx = cp.index
    spy = cp["SPY"]
    vix = _fred("VIXCLS", idx)
    vix_slope20 = vix - vix.shift(20)
    spy_ma200 = spy.rolling(200).mean()
    spy_dist_200 = (spy - spy_ma200) / spy_ma200

    # HY proxy
    hyg = _etf_close("HYG", idx)
    lqd = _etf_close("LQD", idx)
    cr = hyg / lqd if (not hyg.isna().all() and not lqd.isna().all()) else None
    cr_252 = cr.rolling(252, min_periods=30).max() if cr is not None else None
    hy_widening = ((cr_252 - cr) / cr_252 > 0.05) if cr is not None else pd.Series(False, index=idx)

    # SPY-TLT corr
    tlt_r = cp.get("TLT", spy).pct_change()
    corr_flip = (spy.pct_change().rolling(60).corr(tlt_r) > 0.3)

    # CPI
    cpi = _fred("CPIAUCSL", idx)
    cpi_yoy = cpi.pct_change(252) * 100 if not cpi.isna().all() else pd.Series(np.nan, index=idx)
    cpi_hot = cpi_yoy > 4.0

    # TLT 3m return
    tlt_3m = cp.get("TLT", spy).pct_change(63)

    regime = pd.Series("BULL", index=idx)
    # CRISIS: vol spike + credit widening + SPY below 200
    regime[(vix > 30) & hy_widening & (spy_dist_200 < -0.10)] = "CRISIS"
    # INFLATION: corr flip + CPI hot + TLT falling
    regime[corr_flip & cpi_hot & (tlt_3m < -0.03)] = "INFLATION"
    # BEAR: trending down + rising vol
    regime[(spy_dist_200 < -0.05) & (vix > 22) & (vix_slope20 > 0) & (regime == "BULL")] = "BEAR"
    # SIDEWAYS: range-bound
    regime[(spy_dist_200.abs() < 0.03) & (vix < 22) & (vix > 15) & (regime == "BULL")] = "SIDEWAYS"

    # Hysteresis: require 5-day confirmation
    # (Skip for simplicity — just use instantaneous regime with 1-day lag)
    return regime.shift(1).fillna("BULL")


def sleeve_regime(cp: pd.DataFrame) -> pd.DataFrame:
    """Per-regime allocation from agent research."""
    regime = classify_regime(cp)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    # BULL: 60% UPRO + 30% TQQQ + 10% cash
    bull = (regime == "BULL").astype(float)
    if "UPRO" in cp.columns: W["UPRO"] = bull * 0.60
    if "TQQQ" in cp.columns: W["TQQQ"] = bull * 0.30

    # BEAR: 30% TMF
    bear = (regime == "BEAR").astype(float)
    if "TMF" in cp.columns: W["TMF"] = W.get("TMF", 0) + bear * 0.30

    # SIDEWAYS: 50% UPRO (short-term holds, simplified)
    side = (regime == "SIDEWAYS").astype(float)
    if "UPRO" in cp.columns: W["UPRO"] = W["UPRO"] + side * 0.30

    # CRISIS: 25% UGL
    crisis = (regime == "CRISIS").astype(float)
    if "UGL" in cp.columns: W["UGL"] = W.get("UGL", 0) + crisis * 0.25

    # INFLATION: 40% UCO + 30% UGL
    infl = (regime == "INFLATION").astype(float)
    if "UCO" in cp.columns: W["UCO"] = W.get("UCO", 0) + infl * 0.40
    if "UGL" in cp.columns: W["UGL"] = W["UGL"] + infl * 0.30

    return _scale_to_vol(W, cp, target_vol=0.15)


# ==========================================================================
# CALENDAR STACK — 8 anomalies summed (TOM + pre-holiday + Santa + Sell-in-May)
# ==========================================================================

def sleeve_calendar(cp: pd.DataFrame) -> pd.DataFrame:
    """Calendar stack combining TOM, pre-holiday, Santa rally, Sell-in-May (TMF)."""
    idx = cp.index
    day = pd.Series(idx.day, index=idx)
    month = pd.Series(idx.month, index=idx)
    year = pd.Series(idx.year, index=idx)
    weekday = pd.Series(idx.weekday, index=idx)

    # Turn-of-month: last 2 + first 3 trading days of month
    group = year.astype(str) + "-" + month.astype(str).str.zfill(2)
    g = pd.DataFrame({"g": group.values}, index=idx)
    fwd_rank = g.groupby("g").cumcount()
    back_rank = g.groupby("g").cumcount(ascending=False)
    tom = ((fwd_rank < 3) | (back_rank < 2)).astype(float)

    # Pre-holiday (2 days before major US holidays, approximated)
    # Thanksgiving ~4th Thu Nov; Xmas Dec 25; July 4; Memorial Day ~last Mon May;
    # Labor Day ~first Mon Sep; New Year Jan 1
    pre_holiday = pd.Series(0.0, index=idx)
    # Simple: days 23-25 of Dec, days 2-3 of July, days 23-24 Nov, day 25 May, etc.
    pre_holiday[((month == 12) & (day >= 22) & (day <= 25)) |
                ((month == 7) & (day >= 2) & (day <= 3)) |
                ((month == 11) & (day >= 22) & (day <= 24)) |
                ((month == 5) & (day >= 25) & (day <= 27)) |
                ((month == 9) & (day >= 1) & (day <= 3))] = 1.0

    # Santa rally: Dec 15 - Jan 3
    santa = pd.Series(0.0, index=idx)
    santa[((month == 12) & (day >= 15)) | ((month == 1) & (day <= 3))] = 1.0

    # Summer: May 1 - Oct 31, long bonds (TMF)
    summer = ((month >= 5) & (month <= 10)).astype(float)

    # Market regime: only active when SPY > 200MA
    spy = cp["SPY"]
    mkt_ok = (spy > spy.rolling(200).mean()).astype(float)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    # UPRO: TOM + pre-holiday + Santa (voting, cap)
    upro_vote = 0.30 * tom + 0.30 * pre_holiday + 0.25 * santa
    upro_vote = upro_vote.clip(upper=0.60)
    if "UPRO" in cp.columns:
        W["UPRO"] = upro_vote * mkt_ok
    # TMF: summer window
    if "TMF" in cp.columns:
        W["TMF"] = summer * 0.20

    return _scale_to_vol(W, cp, target_vol=0.15)


# ==========================================================================
# VRP SLEEVE — SVXY conditional on vol term structure in contango
# ==========================================================================

def sleeve_vrp(cp: pd.DataFrame) -> pd.DataFrame:
    """Long SVXY when: RV_5d/RV_21d < 0.92 AND SPY RV < VIX (VRP > 2)
    AND VIX < 25 AND SPY > 200MA AND no VIX spike 5d.
    Circuit breakers built in."""
    idx = cp.index
    if "SVXY" not in cp.columns:
        return pd.DataFrame(0.0, index=idx, columns=cp.columns)

    spy = cp["SPY"]
    vix = _fred("VIXCLS", idx)

    rv_5d = spy.pct_change().rolling(5).std() * np.sqrt(util.DPY) * 100
    rv_21d = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY) * 100
    term_slope = rv_5d / rv_21d
    vrp_proxy = vix - rv_21d

    # VIX 5d max change
    vix_5d_max = vix.pct_change(5).rolling(5).max()
    vix_spiked = vix_5d_max > 0.15

    conditions = (
        (term_slope < 0.92) &
        (vrp_proxy > 2.0) &
        (vix < 25) &
        (spy > spy.rolling(200).mean()) &
        (~vix_spiked) &
        (vix.pct_change(1) < 0.20)
    )
    # Circuit breakers
    vix_spike_1d = vix.pct_change(1) > 0.20
    vix_danger = vix > 30
    halt_5d = vix_spike_1d | vix_danger
    halt_extended = halt_5d.rolling(10, min_periods=1).max().astype(bool)

    on = (conditions & ~halt_extended).astype(float).shift(1).fillna(0.0)

    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    W["SVXY"] = on * 0.25   # max 25% allocation (cap tail risk)

    return _scale_to_vol(W, cp, target_vol=0.12)


# ==========================================================================
# DISPERSION-GATED MOMENTUM — survives 2022 via USO/UCO pick
# ==========================================================================

def sleeve_dispersion_mom(cp: pd.DataFrame) -> pd.DataFrame:
    """When cross-sectional dispersion is high, go long top-2 by 63d momentum
    across diversified asset classes. Survives 2022 because UCO/UGL have
    positive momentum while equity/bonds are dumping."""
    # Underlyings (not LETFs) for cleaner dispersion measure
    univ_under = [u for u in ["SPY", "QQQ", "TLT", "GLD", "USO", "VNQ", "EEM"] if u in cp.columns]
    # If USO not directly, use UCO as commodity proxy
    if "USO" not in cp.columns and "UCO" in cp.columns:
        pass  # skip
    # Use LETF-equivalent set
    univ_letf = {"SPY": "UPRO", "QQQ": "TQQQ", "TLT": "TMF", "GLD": "UGL",
                 "USO": "UCO", "VNQ": "DRN", "EEM": "EDC"}
    # Filter to what's available
    map_ok = {u: l for u, l in univ_letf.items() if u in cp.columns and l in cp.columns}
    if len(map_ok) < 3:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    unlev_df = cp[list(map_ok.keys())]
    r21 = unlev_df.pct_change(21)

    # Dispersion: cross-sectional std of 21d returns
    disp = r21.std(axis=1)
    disp_z = (disp - disp.rolling(252, min_periods=60).mean()) / disp.rolling(252, min_periods=60).std()

    # Momentum top-2 by 63d
    mom63 = unlev_df.pct_change(63)
    rnk = mom63.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2) & (mom63 > 0)

    # Only trade when dispersion > 0.5 z
    high_disp = (disp_z > 0.5).astype(float).fillna(0)

    # Weekly rebal
    is_r = pd.Series(cp.index.weekday, index=cp.index) == 4
    sel_wk = sel.where(is_r.values[:, None].repeat(len(sel.columns), axis=1)
                      if False else is_r, axis=0).ffill().fillna(False)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for u, l in map_ok.items():
        W[l] = 0.5 * sel_wk[u].astype(float) * high_disp

    return _scale_to_vol(W, cp, target_vol=0.15)


# ==========================================================================
# CPPI — convex upside, floor protection via SSO + cash
# ==========================================================================

def sleeve_cppi(cp: pd.DataFrame, multiplier: float = 3.0,
                floor_pct: float = 0.85) -> pd.DataFrame:
    """Constant Proportion Portfolio Insurance.
    Allocation to SSO = multiplier * (NAV - floor), capped at 1.0.
    Floor resets every 252 days to maintain rolling protection.
    """
    idx = cp.index
    if "SSO" not in cp.columns:
        return pd.DataFrame(0.0, index=idx, columns=cp.columns)

    # Simulate NAV trajectory naively
    # Floor = 85% of rolling 252d high
    sso = cp["SSO"]
    # Rolling 252d max of SSO price as a "NAV proxy"
    nav_max = sso.rolling(252, min_periods=30).max()
    floor = floor_pct * nav_max
    cushion = (sso - floor) / sso
    cushion = cushion.clip(lower=0, upper=1.0)

    # Allocation
    alloc_sso = (multiplier * cushion).clip(upper=1.0, lower=0.0)

    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    W["SSO"] = alloc_sso.shift(1).fillna(0.5)

    return _scale_to_vol(W, cp, target_vol=0.15)
