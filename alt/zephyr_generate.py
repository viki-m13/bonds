"""Generate blend_factsheet_data.json matching blend.html schema.

Produces ZEPHYR strategy data in the format the HTML expects
(keys like "Blend", DICHS used as drawdown column name, etc.).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

PORTFOLIO = {
    "JAAA": 0.32,
    "JPST": 0.28,
    "MINT": 0.15,
    "BKLN": 0.10,
    "SRLN": 0.05,
    "FLOT": 0.05,
    "GLD":  0.05,
}
ETF_META = {
    "JAAA": ("Janus Henderson AAA CLO ETF", "CLO (AAA)", "CLO / Structured"),
    "JPST": ("JPMorgan Ultra-Short Income ETF", "Ultra-short IG credit", "Ultra-Short Credit"),
    "MINT": ("PIMCO Enhanced Short Maturity ETF", "Ultra-short IG credit", "Ultra-Short Credit"),
    "BKLN": ("Invesco Senior Loan ETF", "Senior leveraged loans", "Floating Rate"),
    "SRLN": ("SPDR Blackstone Senior Loan ETF", "Active senior loans", "Floating Rate"),
    "FLOT": ("iShares Floating Rate Bond ETF", "IG floating-rate notes", "Floating Rate"),
    "GLD":  ("SPDR Gold Shares ETF", "Physical gold crisis hedge", "Gold"),
}
REBALANCE_DAYS = 21
TC_BPS = 5.0
FEE_ANNUAL = 0.01


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def build_regime(dates):
    hy = load_fred("BAMLH0A0HYM2")
    if hy is not None:
        h = hy.reindex(dates).ffill()
        hy_g = ((8.0 - h) / (8.0 - 5.0)).clip(0, 1)
    else:
        hy_g = pd.Series(1.0, index=dates)
    y = load_fred("DGS10")
    if y is not None:
        yv = y.reindex(dates).ffill()
        chg = yv - yv.shift(63)
        rt_g = (chg < 0.7).astype(float)
    else:
        rt_g = pd.Series(1.0, index=dates)
    return (hy_g * rt_g).shift(1).fillna(1.0)


def backtest(weights, start=None, end=None, use_regime=True):
    prices = pd.DataFrame({t: load_etf(t) for t in weights}).dropna()
    if start: prices = prices.loc[start:]
    if end: prices = prices.loc[:end]
    rets = prices.pct_change().fillna(0)
    dates = rets.index
    target = pd.Series(weights); target = target / target.sum()
    current = pd.Series(0.0, index=weights.keys())
    port = pd.Series(0.0, index=dates)
    last_idx = -REBALANCE_DAYS
    rebal_dates = []
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    regime = build_regime(dates) if use_regime else pd.Series(1.0, index=dates)
    for i, d in enumerate(dates):
        if i - last_idx >= REBALANCE_DAYS:
            tc = (target - current).abs().sum() * (TC_BPS / 1e4)
            port.iloc[i] -= tc
            current = target.copy()
            last_idx = i
            rebal_dates.append(d)
        r = (rets.iloc[i] * current).sum()
        g = float(regime.get(d, 1.0))
        r = g * r + (1 - g) * bil.iloc[i]
        port.iloc[i] += r
    return {"returns": port, "prices": prices, "regime": regime,
            "rebal_dates": rebal_dates, "target": target}


def main():
    res = backtest(PORTFOLIO)
    ret = res["returns"]
    if (ret != 0).any():
        ret = ret.loc[ret.ne(0).idxmax():]
    print(f"Inception: {ret.index[0].date()}  End: {ret.index[-1].date()}  Rows: {len(ret)}")
    ar = ret.mean() * 252; av = ret.std() * np.sqrt(252)
    print(f"Sharpe: {ar/av:.3f}  Ann Ret: {ar:.2%}  Vol: {av:.2%}")
    # Save returns series for downstream
    ret.to_csv(RESULTS / "zephyr_returns.csv", header=["Close"])
    # Also save regime
    res["regime"].loc[ret.index[0]:].to_csv(RESULTS / "zephyr_regime.csv", header=["gate"])
    print("Wrote zephyr_returns.csv and zephyr_regime.csv")


if __name__ == "__main__":
    main()
