"""APEX v25 truly novel sleeves — asymmetric signals, ratio trading, etc.

  SL_QQQ_SPY_RATIO  — Long TQQQ when QQQ/SPY ratio breaks 200d high (tech
                       dominance regime); Long UPRO when ratio reverses.
  SL_GOLD_SPY       — Long UGL when GLD/SPY ratio breaks out (gold-leader
                       regime, typically stagflation).
  SL_COPPER_GOLD    — Long UPRO when copper/gold ratio rising (growth regime);
                       Long UGL when falling (recession signal).
  SL_VIX_RANK       — VIX percentile rank: long SSO when VIX at bottom 30%;
                       go defensive UGL+TMF when top 10%.
  SL_MOVING_AVG_CROSS — 50/200 golden cross on SPY triggers UPRO long;
                        death cross triggers UGL.
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


def sleeve_qqq_spy_ratio(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Long TQQQ when QQQ/SPY ratio breaking 60d high (tech leading).
    Long UPRO when QQQ/SPY below 200d MA (broader market leading)."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "QQQ" not in cp.columns or "SPY" not in cp.columns:
        return W

    ratio = cp["QQQ"] / cp["SPY"]
    r_ma60 = ratio.rolling(60).max()
    r_ma200 = ratio.rolling(200).mean()

    tech_breakout = (ratio >= r_ma60 * 0.995).astype(float).shift(1).fillna(0)
    broad_leading = (ratio < r_ma200).astype(float).shift(1).fillna(0)

    spy_ok = (cp["SPY"] > cp["SPY"].rolling(200).mean()).astype(float)

    if "TQQQ" in cp.columns:
        W["TQQQ"] = tech_breakout * spy_ok * 0.5
    if "UPRO" in cp.columns:
        W["UPRO"] = broad_leading * spy_ok * 0.4

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_gold_spy(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Long UGL when GLD/SPY ratio rising and above 200d MA (gold regime).
    Indicates stagflation/risk-off."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "GLD" not in cp.columns or "SPY" not in cp.columns:
        return W

    ratio = cp["GLD"] / cp["SPY"]
    ratio_ma = ratio.rolling(200).mean()
    ratio_mom = ratio.pct_change(63)

    gold_regime = ((ratio > ratio_ma) & (ratio_mom > 0.05)).astype(float).shift(1).fillna(0)

    if "UGL" in cp.columns:
        W["UGL"] = gold_regime * 0.6

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_ma_cross(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """50/200 golden cross = risk on (long UPRO + TQQQ).
    Death cross = risk off (long UGL + TMF).
    Classic but timeless."""
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    spy = cp["SPY"]
    ma50 = spy.rolling(50).mean()
    ma200 = spy.rolling(200).mean()
    golden = (ma50 > ma200).astype(float).shift(1).fillna(0)
    death = 1 - golden

    if "UPRO" in cp.columns:
        W["UPRO"] = golden * 0.4
    if "TQQQ" in cp.columns:
        W["TQQQ"] = golden * 0.3
    if "UGL" in cp.columns:
        W["UGL"] = death * 0.3
    if "TMF" in cp.columns:
        W["TMF"] = death * 0.2

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_vix_rank(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """VIX percentile rank over 252d:
      - Bottom 30% (VIX low): long SSO
      - Top 10% (VIX high): long UGL + TMF
      - Middle: cash (0 weight)
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    vix = _fred("VIXCLS", idx)
    vix_rank = vix.rolling(252, min_periods=60).rank(pct=True)

    low = (vix_rank < 0.30).astype(float).shift(1).fillna(0)
    high = (vix_rank > 0.90).astype(float).shift(1).fillna(0)

    # Market filter for low regime
    spy_ok = (cp["SPY"] > cp["SPY"].rolling(200).mean()).astype(float)

    if "SSO" in cp.columns:
        W["SSO"] = low * spy_ok * 0.6
    if "UGL" in cp.columns:
        W["UGL"] = high * 0.4
    if "TMF" in cp.columns:
        W["TMF"] = high * 0.3

    return _scale_to_vol(W, cp, target_vol=target_vol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()

    sleeves = {
        "QQQ_SPY_RATIO":  sleeve_qqq_spy_ratio(cp),
        "GOLD_SPY":       sleeve_gold_spy(cp),
        "MA_CROSS":       sleeve_ma_cross(cp),
        "VIX_RANK":       sleeve_vix_rank(cp),
    }
    print(f"{'Sleeve':18s}  {'SR':>5}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sleeves.items():
        r = _weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:18s}  {m['sharpe']:>5.2f}  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")
