"""Market regime features (daily, causal) for regime-conditional layers.

All series are computed from data through each row's close. Sources: SPY/QQQ
benchmark files, FRED (VIX, HY OAS), and panel breadth.
"""
import os

import numpy as np
import pandas as pd

import data as data_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fred(series: str) -> pd.Series:
    df = pd.read_csv(os.path.join(ROOT, "data", "fred", f"{series}.csv"))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["value"]
    return pd.to_numeric(s, errors="coerce")


def build_regime(index: pd.DatetimeIndex) -> pd.DataFrame:
    spy = data_mod.load_benchmark("SPY")["Close"]
    P = data_mod.build_panel()
    close, member = P["close"], P["member"]

    out = pd.DataFrame(index=index)
    spy = spy.reindex(index).ffill()
    out["spy_above_200"] = (spy > spy.rolling(200).mean()).astype(float)
    out["spy_above_100"] = (spy > spy.rolling(100).mean()).astype(float)
    out["spy_ret_63"] = spy.pct_change(63)
    out["spy_dd"] = spy / spy.cummax() - 1

    # breadth: share of current members above their own 200dma
    above = close > close.rolling(200).mean()
    memb = member & close.notna()
    out["breadth_200"] = (above & memb).sum(axis=1) / memb.sum(axis=1)
    out["breadth_200_ma10"] = out["breadth_200"].rolling(10).mean()

    vix = _fred("VIXCLS").reindex(index).ffill()
    out["vix"] = vix
    # trailing percentile of VIX vs its own past 3y (causal)
    out["vix_pct3y"] = vix.rolling(756, min_periods=252).rank(pct=True)

    try:
        oas = _fred("BAMLH0A0HYM2").reindex(index).ffill()
        out["hy_oas"] = oas
        out["hy_oas_chg63"] = oas - oas.rolling(63).mean()
    except FileNotFoundError:
        pass
    return out


def risk_on(index: pd.DatetimeIndex, mode: str = "trend_breadth") -> pd.Series:
    """Boolean risk-on series. Conservative default: SPY>200dma OR breadth
    recovering. All causal."""
    R = build_regime(index)
    if mode == "trend":
        return R["spy_above_200"] > 0
    if mode == "trend_breadth":
        return (R["spy_above_200"] > 0) | (R["breadth_200_ma10"] > 0.6)
    if mode == "strict":
        return ((R["spy_above_200"] > 0) & (R["hy_oas_chg63"] < 0.5)
                if "hy_oas_chg63" in R else R["spy_above_200"] > 0)
    raise ValueError(mode)
