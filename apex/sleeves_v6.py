"""APEX v6 sleeves — designed for orthogonality, not redundancy.

Each sleeve uses a DIFFERENT signal structure to ensure near-zero correlation:
  V1  MOM_LEV_189 — 189d mom on 4-LETF safe/risk basket (Phoenix VANGUARD clone)
  V2  RISK_SAFE   — 50/50 RISK (eq-LETFs) + SAFE (bond/gold), VIX-gated, weekly (Phoenix ORION)
  V3  UNLEV_MOM   — 6m mom on unlevered {SPY,QQQ,TLT,GLD,IEF,EEM}, expressed via 3× LETF (Phoenix HELIOS)
  V4  ML_SHORT    — XGBoost 5d-horizon, top-3, weekly rebal
  V5  ML_LONG     — XGBoost 63d-horizon, top-3, monthly rebal (different horizon = orthogonal)
  V6  SHORT_MR    — RSI(2)<5 + uptrend dip-buy, 3d hold
  V7  CALENDAR    — turn-of-month equity tilt
  V8  CARRY_CURVE — TMF/TYD based on yield-curve slope change
  V9  VOL_REGIME  — SSO when SPY RV regime LOW+DECLINING
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


def _etf(t, idx):
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


def _scale_to_vol(W, cp, target_vol=0.20, win=60):
    """Scale weights DOWN so realized vol ≤ target (never up past 1.0 cap)."""
    r = _weights_to_ret(W, cp)
    rv = r.rolling(win, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


# V1 — Phoenix VANGUARD clone
def s_v1_mom_lev(cp, target_vol=0.18):
    universe = [a for a in ["QLD", "UGL", "TMF", "TYD"] if a in cp.columns]
    p = cp[universe]
    # 189d momentum skip 42
    mom = p.shift(42).pct_change(189 - 42)
    ma200 = p.rolling(200).mean()
    eligible = (p > ma200) & (mom > 0)
    rnk = mom.where(eligible).rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2).fillna(False)
    # Inverse-60d-vol among selected
    vol60 = p.pct_change().rolling(60).std()
    iv = 1.0 / vol60.replace(0, np.nan)
    raw = sel.astype(float) * iv
    rowsum = raw.sum(axis=1).replace(0, np.nan)
    w = raw.div(rowsum, axis=0).fillna(0.0)
    # VIX gate
    vix = _fred("VIXCLS", cp.index)
    vix_ok = ((vix < 30)).astype(float).ffill().fillna(1.0)
    # Monthly rebal (21 days)
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_r = mask % 21 == 0
    w_m = w.where(is_r).ffill().fillna(0.0)
    w_m = w_m.mul(vix_ok, axis=0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = w_m[a]
    return _scale_to_vol(W, cp, target_vol=target_vol)


# V2 — Phoenix ORION clone (RISK + SAFE, weekly)
def s_v2_risk_safe(cp, target_vol=0.18):
    risk_u = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","YINN"] if a in cp.columns]
    safe_u = [a for a in ["TMF","UBT","TYD","UGL"] if a in cp.columns]
    p_r = cp[risk_u]
    p_s = cp[safe_u]
    mom_r = p_r.pct_change(252).shift(21)
    rnk_r = mom_r.rank(axis=1, ascending=False, method="first")
    sel_r = (rnk_r <= 3) & (mom_r > 0)
    w_r = sel_r.astype(float) / 3.0

    mom_s = p_s.pct_change(252).shift(21)
    vol_s = p_s.pct_change().rolling(60).std()
    mom_rank = mom_s.rank(axis=1, pct=True)
    lv_rank = 1 - vol_s.rank(axis=1, pct=True)
    comp = 0.7 * mom_rank + 0.3 * lv_rank
    rnk_s = comp.rank(axis=1, ascending=False, method="first")
    sel_s = (rnk_s <= 2)
    w_s = sel_s.astype(float) / 2.0

    # Weekly rebal
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_r = mask % 5 == 0
    w_r_w = w_r.where(is_r).ffill().fillna(0.0)
    w_s_w = w_s.where(is_r).ffill().fillna(0.0)

    # VIX gate on RISK
    vix = _fred("VIXCLS", cp.index)
    vix_ok = ((vix < 30)).astype(float).ffill().fillna(1.0)
    w_r_w = w_r_w.mul(vix_ok, axis=0)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in risk_u:
        W[a] = 0.5 * w_r_w[a]
    for a in safe_u:
        W[a] = 0.5 * w_s_w[a]
    return _scale_to_vol(W, cp, target_vol=target_vol)


# V3 — SECTOR ROTATION using underlying sector ETF mom → LETF
def s_v3_sector(cp, target_vol=0.18):
    """Rotate among sector LETFs {TECL, FAS, SOXL, ERX, DRN} using 63d mom
    on their UNDERLYINGS. Top-1 weekly, market filter SPY>200MA."""
    # Map sector LETFs to underlying close-proxy available via cp (SPY stand-in is fine)
    SECTORS = [("TECL", "QQQ"), ("FAS", "SPY"), ("SOXL", "QQQ"),
               ("ERX", "SPY"), ("DRN", "SPY")]
    # Use LETF itself to compute momentum (since XLK/XLF/XLE aren't in cp)
    sectors = [s for s, _ in SECTORS if s in cp.columns]
    p = cp[sectors]
    mom = p.shift(5).pct_change(63 - 5)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 1) & (mom > 0)

    # Weekly rebal
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_r = mask % 5 == 0
    sel_w = sel.where(is_r).ffill().fillna(False)

    # Market regime: SPY > 200MA AND VIX < 30
    spy = cp["SPY"]
    mkt_ok = (spy > spy.rolling(200).mean()).astype(float)
    vix = _fred("VIXCLS", cp.index)
    vix_ok = ((vix < 30)).astype(float).ffill().fillna(1.0)
    gate = mkt_ok * vix_ok

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for s in sectors:
        W[s] = sel_w[s].astype(float) * gate
    return _scale_to_vol(W, cp, target_vol=target_vol)


# V4 — Load pre-trained ML5 weights
def s_v4_ml5(cp, target_vol=0.18):
    fp = Path("/home/user/bonds/data/apex/ml5_weights.csv")
    if not fp.exists():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W = pd.read_csv(fp, parse_dates=["Date"], index_col="Date")
    W = W.reindex(cp.index).fillna(0.0)
    full = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in W.columns:
        if c in full.columns:
            full[c] = W[c]
    return _scale_to_vol(full, cp, target_vol=target_vol)


# V5 — Load pre-trained ML63 weights
def s_v5_ml63(cp, target_vol=0.18):
    fp = Path("/home/user/bonds/data/apex/ml63_weights.csv")
    if not fp.exists():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W = pd.read_csv(fp, parse_dates=["Date"], index_col="Date")
    W = W.reindex(cp.index).fillna(0.0)
    full = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in W.columns:
        if c in full.columns:
            full[c] = W[c]
    return _scale_to_vol(full, cp, target_vol=target_vol)


# V6 — Short-term mean reversion
def s_v6_short_mr(cp, target_vol=0.18):
    """RSI(2) < 5 AND SPY > 200MA AND asset > 200MA → buy for 3 days."""
    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS"] if a in cp.columns]
    p = cp[universe]
    # RSI(2) proxy
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta).clip(lower=0).rolling(2).mean()
    rsi2 = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # Asset's own 200MA
    ma200 = p.rolling(200).mean()
    # Signal
    trig = ((rsi2 < 5) & (p > ma200)).astype(float)
    held = trig.rolling(3, min_periods=1).sum().clip(upper=1.0)
    # Market regime: SPY > 200MA
    spy = cp["SPY"]
    mkt_ok = (spy > spy.rolling(200).mean()).astype(float)
    # Equal weight among triggered
    n_held = held.sum(axis=1).replace(0, np.nan)
    w = held.div(n_held, axis=0).fillna(0.0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = w[a] * mkt_ok
    return _scale_to_vol(W, cp, target_vol=target_vol)


# V7 — Calendar / turn-of-month
def s_v7_calendar(cp, target_vol=0.18):
    idx = cp.index
    day = pd.Series(idx.day, index=idx)
    month = pd.Series(idx.month, index=idx)
    year = pd.Series(idx.year, index=idx)
    group = year.astype(str) + "-" + month.astype(str).str.zfill(2)
    g = pd.DataFrame({"g": group.values}, index=idx)
    fwd = g.groupby("g").cumcount()
    back = g.groupby("g").cumcount(ascending=False)
    tom = ((fwd < 3) | (back < 2))
    santa = (((month == 12) & (day >= 15)) | ((month == 1) & (day <= 3)))
    on = (tom | santa)

    # Pick top-2 equity LETF by 63d momentum during the window
    eq_u = [a for a in ["UPRO", "TQQQ", "TECL", "FAS"] if a in cp.columns]
    p = cp[eq_u]
    mom = p.pct_change(63)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= 2) & (mom > 0)

    # Market regime: SPY > 200MA
    spy = cp["SPY"]
    mkt_ok = (spy > spy.rolling(200).mean()).astype(float)

    sel_on = sel.astype(float).mul(on.astype(float), axis=0).mul(mkt_ok, axis=0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in eq_u:
        W[a] = 0.5 * sel_on[a]
    return _scale_to_vol(W, cp, target_vol=target_vol)


# V8 — Carry / curve
def s_v8_curve(cp, target_vol=0.18):
    """Long TMF when T10Y2Y slope rising and positive; TYD when flat;
    cash when deeply inverted."""
    slope = _fred("T10Y2Y", cp.index)
    if slope.isna().all():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    ma20 = slope.rolling(20).mean()
    ma60 = slope.rolling(60).mean()
    steep = (slope > 0) & (ma20 > ma60)
    flat = (slope.between(-0.5, 0.25)) & ~steep
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "TMF" in cp.columns:
        W["TMF"] = steep.astype(float)
    if "TYD" in cp.columns:
        W["TYD"] = flat.astype(float) * 0.5
    # Apply bond correlation filter
    if "TLT" in cp.columns:
        sr = cp["SPY"].pct_change()
        tr = cp["TLT"].pct_change()
        corr = sr.rolling(20).corr(tr)
        pos_streak = (corr > 0).rolling(10).sum()
        corr_ok = (pos_streak < 8).astype(float).fillna(1.0)
        for c in ["TMF", "TYD"]:
            if c in W.columns:
                W[c] = W[c] * corr_ok
    return _scale_to_vol(W, cp, target_vol=target_vol)


# V9 — Volatility regime
def s_v9_vol_regime(cp, target_vol=0.18):
    """Long SSO when SPY 21d RV in bottom 50% AND < 63d RV."""
    spy = cp["SPY"]
    rv21 = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    rv63 = spy.pct_change().rolling(63).std() * np.sqrt(util.DPY)
    med = rv21.rolling(504, min_periods=60).median()
    ma200 = spy.rolling(200).mean()
    on = ((rv21 < rv63) & (rv21 < med) & (spy > ma200)).astype(float)
    # Also gate by VIX
    vix = _fred("VIXCLS", cp.index)
    vix_ok = ((vix < 25)).astype(float).ffill().fillna(1.0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "SSO" in cp.columns:
        W["SSO"] = on * vix_ok
    return _scale_to_vol(W, cp, target_vol=target_vol)
