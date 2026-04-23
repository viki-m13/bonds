"""APEX — production sleeves (v4, properly vol-scaled with no-margin cap).

CONSTRAINTS:
  • Each sleeve's weights sum to ≤ 1 at all times
  • Portfolio blend of sleeve weights ≤ 1 (no margin)
  • Sleeve vol scaling only DECREASES exposure (never inflates above 100%)

Each sleeve returns a T×N weights DataFrame. The blend then averages sleeves.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import util

ROOT = Path("/home/user/bonds")
FRED = ROOT / "data/fred"


def _fred(name: str, idx: pd.DatetimeIndex) -> pd.Series:
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def _weights_to_returns(W: pd.DataFrame, cp: pd.DataFrame) -> pd.Series:
    """Daily return from weights (no overlay, simple TC)."""
    w = W.fillna(0.0)
    rets = cp.pct_change()
    r = (w.shift(1).fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def _scale_down(W: pd.DataFrame, r: pd.Series, target_vol: float = 0.20,
                win: int = 60) -> pd.DataFrame:
    """Multiply all weights by a scalar ≤ 1 so realized vol → target.
    Never scales up (no margin). Lagged by 1 day to avoid lookahead."""
    rv = r.rolling(win, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


def _finalize_sleeve(W: pd.DataFrame, cp: pd.DataFrame, target_vol: float = 0.20) -> pd.DataFrame:
    """Build the sleeve: run weights, compute realized vol, scale down to target."""
    r = _weights_to_returns(W, cp)
    return _scale_down(W, r, target_vol=target_vol)


# ----------------------------------------------------------------------------
# S1 TSMOM
# ----------------------------------------------------------------------------

def sleeve_tsmom(cp: pd.DataFrame, target_vol: float = 0.12) -> pd.DataFrame:
    universe = ["UPRO", "TQQQ", "TECL", "SOXL", "FAS", "EDC", "YINN",
                "TMF", "UBT", "TYD", "UGL", "UCO"]
    universe = [a for a in universe if a in cp.columns]
    p = cp[universe]
    rv60 = p.pct_change().rolling(60).std() * np.sqrt(util.DPY)

    cons = pd.DataFrame(0.0, index=cp.index, columns=universe)
    for L in (21, 63, 126, 252):
        cons = cons + np.sign(p.pct_change(L)).fillna(0.0)
    cons = (cons / 4).clip(lower=0.0)

    raw = cons * (0.12 / rv60.replace(0, np.nan))
    raw = raw.fillna(0.0).clip(upper=0.33)
    s = raw.sum(axis=1)
    scale = np.minimum(1.0, 1.0 / s.replace(0, np.nan)).fillna(0.0)
    w = raw.mul(scale, axis=0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = w[a]
    return _finalize_sleeve(W, cp, target_vol=target_vol)


# ----------------------------------------------------------------------------
# S2 XSMOM
# ----------------------------------------------------------------------------

def sleeve_xsmom(cp: pd.DataFrame, target_vol: float = 0.12,
                 top_n: int = 2, rebal_every: int = 21,
                 lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    universe = ["UPRO", "TQQQ", "TECL", "SOXL", "FAS", "EDC", "YINN",
                "TMF", "UBT", "UGL", "DRN", "UCO"]
    universe = [a for a in universe if a in cp.columns]
    p = cp[universe]
    mom = p.shift(skip).pct_change(lookback - skip)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel_day = (rnk <= top_n) & (mom > 0)
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % rebal_every == 0
    sel_m = sel_day.where(is_rebal).ffill().fillna(False)
    n_sel = sel_m.sum(axis=1)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = (sel_m[a].astype(float) / n_sel.replace(0, np.nan)).fillna(0.0)
    return _finalize_sleeve(W, cp, target_vol=target_vol)


# ----------------------------------------------------------------------------
# S3 RPAR
# ----------------------------------------------------------------------------

def sleeve_rpar(cp: pd.DataFrame, target_vol: float = 0.20,
                sleeves_list: tuple = ("UPRO", "TMF", "UGL"),
                vol_win: int = 60) -> pd.DataFrame:
    sleeves_list = [a for a in sleeves_list if a in cp.columns]
    rv = cp[sleeves_list].pct_change().rolling(vol_win).std()
    iv = 1.0 / rv.replace(0, np.nan)
    iv = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in sleeves_list:
        W[a] = iv[a]
    return _finalize_sleeve(W, cp, target_vol=target_vol)


# ----------------------------------------------------------------------------
# Sn TREND_<X>
# ----------------------------------------------------------------------------

def sleeve_trend(cp: pd.DataFrame, letf: str, under: str,
                 fast: int = 50, slow: int = 200, ret_win: int = 126,
                 target_vol: float = 0.20) -> pd.DataFrame:
    if letf not in cp.columns or under not in cp.columns:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    u = cp[under]
    ma_s = u.rolling(slow).mean()
    ma_f = u.rolling(fast).mean()
    r = u.pct_change(ret_win)
    on = ((u > ma_s) & (ma_f > ma_s) & (r > 0)).astype(float)
    rv = cp[letf].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    sc = (0.25 / rv.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W[letf] = on * sc
    return _finalize_sleeve(W, cp, target_vol=target_vol)


def sleeve_trend_eq(cp):        return sleeve_trend(cp, "TQQQ", "QQQ")
def sleeve_trend_eq_spy(cp):    return sleeve_trend(cp, "UPRO", "SPY")
def sleeve_trend_bd(cp):        return sleeve_trend(cp, "TMF", "TLT")
def sleeve_trend_gd(cp):        return sleeve_trend(cp, "UGL", "GLD")
def sleeve_trend_intl(cp):
    if "EEM" in cp.columns:
        return sleeve_trend(cp, "EDC", "EEM")
    return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)


# ----------------------------------------------------------------------------
# S_CREDIT
# ----------------------------------------------------------------------------

def sleeve_credit(cp: pd.DataFrame, risky: str = "UPRO",
                  target_vol: float = 0.20) -> pd.DataFrame:
    hy = _fred("BAMLH0A0HYM2", cp.index)
    if hy.isna().all():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    pct80 = hy.rolling(504, min_periods=60).quantile(0.80)
    ma60 = hy.rolling(60).mean()
    on = ((hy < pct80) & (hy < ma60)).astype(float)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        rv = cp[risky].pct_change().rolling(60).std() * np.sqrt(util.DPY)
        sc = (0.25 / rv.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
        W[risky] = on * sc
    return _finalize_sleeve(W, cp, target_vol=target_vol)


# ----------------------------------------------------------------------------
# S_VOLREG
# ----------------------------------------------------------------------------

def sleeve_volreg(cp: pd.DataFrame, risky: str = "SSO",
                  target_vol: float = 0.20) -> pd.DataFrame:
    spy = cp["SPY"]
    rv21 = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    rv63 = spy.pct_change().rolling(63).std() * np.sqrt(util.DPY)
    med_rv = rv21.rolling(504, min_periods=60).median()
    ma200 = spy.rolling(200).mean()
    on = ((rv21 < rv63) & (rv21 < med_rv) & (spy > ma200)).astype(float)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        rv = cp[risky].pct_change().rolling(60).std() * np.sqrt(util.DPY)
        sc = (0.25 / rv.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
        W[risky] = on * sc
    return _finalize_sleeve(W, cp, target_vol=target_vol)


# ----------------------------------------------------------------------------
# S_CRISIS (go to UGL in stress)
# ----------------------------------------------------------------------------

def sleeve_crisis(cp: pd.DataFrame, target_vol: float = 0.20) -> pd.DataFrame:
    spy = cp["SPY"]
    rv60 = spy.pct_change().rolling(60).std() * np.sqrt(util.DPY)
    r60 = spy.pct_change(60)
    crisis = ((rv60 > 0.30) | (r60 < -0.10)).astype(float)
    calm = 1.0 - crisis

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    # Calm: inv-vol UPRO + TMF
    if "UPRO" in cp.columns and "TMF" in cp.columns:
        u_v = cp["UPRO"].pct_change().rolling(60).std()
        t_v = cp["TMF"].pct_change().rolling(60).std()
        iv_u = 1 / u_v.replace(0, np.nan)
        iv_t = 1 / t_v.replace(0, np.nan)
        wu = iv_u / (iv_u + iv_t)
        wt = iv_t / (iv_u + iv_t)
        W["UPRO"] = calm * wu.fillna(0.5) * 0.9
        W["TMF"] = calm * wt.fillna(0.5) * 0.9
    # Crisis: UGL scaled
    if "UGL" in cp.columns:
        rv = cp["UGL"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
        sc = (0.20 / rv.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
        W["UGL"] = crisis * sc
    return _finalize_sleeve(W, cp, target_vol=target_vol)


# ----------------------------------------------------------------------------
# S_COMMOD (commodity trend — UCO/UGL)
# ----------------------------------------------------------------------------

def sleeve_commod(cp: pd.DataFrame, target_vol: float = 0.20) -> pd.DataFrame:
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "UGL" in cp.columns and "GLD" in cp.columns:
        gld = cp["GLD"]
        on = ((gld > gld.rolling(200).mean()) &
              (gld.rolling(50).mean() > gld.rolling(200).mean())).astype(float)
        rv = cp["UGL"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
        sc = (0.20 / rv.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
        W["UGL"] = 0.5 * on * sc
    if "UCO" in cp.columns and "USO" in cp.columns:
        uso = cp["USO"]
        on = ((uso > uso.rolling(200).mean()) &
              (uso.rolling(50).mean() > uso.rolling(200).mean())).astype(float)
        rv = cp["UCO"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
        sc = (0.20 / rv.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
        W["UCO"] = 0.5 * on * sc
    return _finalize_sleeve(W, cp, target_vol=target_vol)
