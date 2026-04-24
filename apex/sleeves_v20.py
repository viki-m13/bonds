"""APEX v20 — truly orthogonal sleeves targeting specific low-corr niches.

Insight from diagnostic: Phoenix sleeves have 0.04 avg correlation because
they operate on fundamentally DIFFERENT signal types (value/carry/rate-ML/
crypto). Mine overlap because most use LETF-price momentum.

Focus: signals that NEVER look at LETF prices directly.

  SL_RATE_MOMENTUM  — Pure rate-change signal: long TMF when 10Y yield
                       5d diff < 0 AND 20d diff < 0 (rate rally starting).
  SL_USD_STRENGTH   — UUP momentum: long UUP when DXY strong → correlated
                       w/ risk-off which hurts equity. Unique signal.
  SL_SECTOR_ROT     — Long strongest sector LETF based on UNDERLYING sector
                       3m/6m mom (not LETF itself), rotating monthly.
  SL_SMART_MONEY    — Tracks SMH (semis) as leading indicator: when SMH
                       breaks 20d high while SPY doesn't → bullish breadth.
  SL_CREDIT_TIGHT   — HYG/TLT ratio rising (credit premium tightening) →
                       risk on, long UPRO.
  SL_COMMODITY      — DBC/GLD trend: when commodities outperform bonds,
                       inflation regime → long UCO + UGL.
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


def sleeve_rate_momentum(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Long TMF when 10Y yield FALLING (5d diff < 0 AND 20d diff < 0).
    Long TBF when 10Y yield RISING (both diffs > 0).
    Pure interest-rate signal, orthogonal to LETF price."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    dgs10 = _fred("DGS10", idx)
    d5 = dgs10 - dgs10.shift(5)
    d20 = dgs10 - dgs10.shift(20)

    rate_falling = ((d5 < -0.05) & (d20 < -0.10)).astype(float).shift(1).fillna(0)
    rate_rising = ((d5 > 0.05) & (d20 > 0.10)).astype(float).shift(1).fillna(0)

    if "TMF" in cp.columns:
        W["TMF"] = rate_falling * 0.5
    if "TBF" in cp.columns:
        W["TBF"] = rate_rising * 0.5   # long inverse bond ETF when rates rise

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_usd_strength(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """UUP momentum as uncorrelated signal.
    Long UUP when DXY strong AND rising (risk-off regime).
    Long UGL when DXY weakening AND below MA (inflation regime).
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    uup = _etf_close("UUP", idx)
    if uup.isna().all():
        return W

    uup_ma = uup.rolling(200).mean()
    uup_mom63 = uup.pct_change(63)

    strong = ((uup > uup_ma) & (uup_mom63 > 0)).astype(float).shift(1).fillna(0)
    weak = ((uup < uup_ma) & (uup_mom63 < 0)).astype(float).shift(1).fillna(0)

    if "UUP" in cp.columns:
        W["UUP"] = strong * 0.6
    elif uup.sum() > 0:
        cp["UUP"] = uup   # add if not present
        W["UUP"] = strong * 0.6

    if "UGL" in cp.columns:
        W["UGL"] = weak * 0.4

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_smart_money(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Semiconductor index (SMH) as smart-money leading indicator.
    When SMH breaks 20d high AND SPY is flat/lagging → bullish early signal.
    Long SOXL/TQQQ at those moments.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    smh = _etf_close("SMH", idx)
    if smh.isna().all() or "SPY" not in cp.columns:
        return W

    smh_high20 = smh.rolling(20).max()
    smh_breaking = (smh >= smh_high20 * 0.995).astype(float)   # within 0.5% of 20d high

    spy = cp["SPY"]
    spy_below_highs = (spy < spy.rolling(20).max() * 0.98).astype(float)

    # Smart-money signal: SMH breakout + SPY lagging
    signal = (smh_breaking * spy_below_highs).shift(1).fillna(0)

    if "SOXL" in cp.columns:
        W["SOXL"] = signal * 0.4
    if "TQQQ" in cp.columns:
        W["TQQQ"] = signal * 0.3

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_credit_tight(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """HYG/TLT ratio (credit premium) rising → credit spreads tightening → risk-on.

    Distinct from HYG/LQD because TLT is pure-duration, capturing both credit
    and rate effects separately.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    hyg = _etf_close("HYG", idx)
    if hyg.isna().all() or "TLT" not in cp.columns:
        return W

    tlt = cp["TLT"]
    ratio = hyg / tlt
    ratio_ma = ratio.rolling(60).mean()
    ratio_mom = ratio / ratio.shift(20) - 1

    # Signal: ratio above 60d MA AND rising
    risk_on = ((ratio > ratio_ma) & (ratio_mom > 0.01)).astype(float).shift(1).fillna(0)

    if "UPRO" in cp.columns:
        W["UPRO"] = risk_on * 0.4
    if "TQQQ" in cp.columns:
        W["TQQQ"] = risk_on * 0.3

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_commodity_regime(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Commodity-vs-bond momentum regime.

    DBC/TLT or USO/TLT ratio rising = inflation/growth regime → long UCO + UGL.
    Pure macro rotation, orthogonal to equity.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    dbc = _etf_close("DBC", idx)
    uso = _etf_close("USO", idx)
    commod = dbc if not dbc.isna().all() else uso
    if commod.isna().all() or "TLT" not in cp.columns:
        return W

    tlt = cp["TLT"]
    ratio = commod / tlt
    ratio_ma = ratio.rolling(126).mean()
    ratio_mom = ratio.pct_change(63)

    commod_leading = ((ratio > ratio_ma) & (ratio_mom > 0.05)).astype(float).shift(1).fillna(0)

    if "UCO" in cp.columns:
        W["UCO"] = commod_leading * 0.4
    if "UGL" in cp.columns:
        W["UGL"] = commod_leading * 0.3

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_bond_flight(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Bond flight-to-quality: when SPY drops >3% in 5d AND TLT rises,
    it's flight-to-quality. Long TMF during these episodes."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    if "SPY" not in cp.columns or "TLT" not in cp.columns:
        return W

    spy = cp["SPY"]
    tlt = cp["TLT"]
    spy_r5 = spy.pct_change(5)
    tlt_r5 = tlt.pct_change(5)

    flight = ((spy_r5 < -0.03) & (tlt_r5 > 0.01)).astype(float)
    # Hold 10 days
    held = flight.rolling(10, min_periods=1).sum().clip(upper=1.0).shift(1).fillna(0)

    if "TMF" in cp.columns:
        W["TMF"] = held * 0.5

    return _scale_to_vol(W, cp, target_vol=target_vol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()
    # Load extra ETFs
    for t in ["UUP", "DBC", "HYG", "TBF"]:
        if t not in cp.columns:
            s = _etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    sleeves = {
        "RATE_MOM":       sleeve_rate_momentum(cp),
        "USD_STRENGTH":   sleeve_usd_strength(cp),
        "SMART_MONEY":    sleeve_smart_money(cp),
        "CREDIT_TIGHT":   sleeve_credit_tight(cp),
        "COMMOD_REGIME":  sleeve_commodity_regime(cp),
        "BOND_FLIGHT":    sleeve_bond_flight(cp),
    }
    print(f"{'Sleeve':18s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sleeves.items():
        r = _weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:18s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")
