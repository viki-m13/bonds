"""Step 3: Multi-engine approach. Each engine is a trend-following switch
between a leveraged ETF and BIL, gated by its underlying's 200dma.
Combine with ZEPHYR core into a portfolio.

Engine: hold leveraged_etf when underlying > its 200dma, else BIL.
No vol scaling. Monthly rebalance restores weights.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data")
ETF = DATA / "etfs"
FRED = DATA / "fred"


def load(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def metrics(r):
    r = r.loc[r.ne(0).idxmax():] if (r != 0).any() else r
    if r.std() == 0 or len(r) == 0: return None
    ar, av = r.mean() * 252, r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    neg = r[r < 0]
    sor = ar / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 0
    return {"sharpe": ar/av if av > 0 else 0, "ret": ar, "vol": av, "mdd": dd,
            "sortino": sor, "calmar": ar/abs(dd) if dd < 0 else 0,
            "n": len(r)/252, "start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def trend_engine(levered, underlying, dates, ma_len=200):
    """Hold `levered` when `underlying` > its MA, else BIL."""
    u = load(underlying).reindex(dates).ffill()
    lev = load(levered).reindex(dates).ffill().pct_change().fillna(0)
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    ma = u.rolling(ma_len).mean()
    gate = (u > ma).shift(1).fillna(False).astype(float)
    return gate * lev + (1 - gate) * bil


def zephyr_backtest(dates):
    """Approximate ZEPHYR backtest on a common date index."""
    portfolio = {"JAAA": 0.32, "JPST": 0.28, "MINT": 0.15,
                 "BKLN": 0.10, "SRLN": 0.05, "FLOT": 0.05, "GLD": 0.05}
    prices = pd.DataFrame({t: load(t) for t in portfolio}).dropna()
    rets = prices.pct_change().fillna(0).reindex(dates).fillna(0)
    w = pd.Series(portfolio)
    port = (rets * w).sum(axis=1)
    # Regime gate
    hy = load_fred("BAMLH0A0HYM2")
    y = load_fred("DGS10")
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    hy_g = ((8.0 - hy.reindex(dates).ffill()) / (8.0 - 5.0)).clip(0, 1).shift(1).fillna(1.0)
    rt_g = (y.reindex(dates).ffill() - y.reindex(dates).ffill().shift(63) < 0.7).shift(1).fillna(True).astype(float)
    g = hy_g * rt_g
    return g * port + (1 - g) * bil


# Build a common date index from the shortest-history required asset
# Leverage ETFs have varied inception: TQQQ 2010, UPRO 2009, SOXL 2010, TECL 2008,
# TMF 2009, UGL 2008, FAS 2008. JAAA is 2020.
# Use the JAAA-inception start for ZEPHYR-compatible backtests.
# For longer-history, we can backtest without ZEPHYR and use ETF-based proxies.

def run_combo(weights, start="2014-01-01"):
    """weights: dict of sleeve_name -> (weight, returns_func_of_dates)."""
    # First align dates
    tmp = load("SPY")
    dates = tmp.loc[start:].index
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    # Build each sleeve
    sleeves = {}
    for name, (w, func) in weights.items():
        if name == "BIL":
            sleeves[name] = bil
        elif name == "ZEPHYR":
            sleeves[name] = zephyr_backtest(dates)
        else:
            lev, und, ma = func
            sleeves[name] = trend_engine(lev, und, dates, ma)
    # Combine
    df = pd.DataFrame(sleeves)
    wser = pd.Series({k: v[0] for k, v in weights.items()})
    port = (df * wser).sum(axis=1)
    return port, df


# === EXPERIMENTS ===
# "weights" = {name: (weight, (levered, underlying, MA_len))}
experiments = [
    ("eq5_simple", {
        "eq_trend":    (0.2, ("QLD", "SPY", 200)),
        "bond_trend":  (0.2, ("TMF", "TLT", 200)),
        "gold_trend":  (0.2, ("UGL", "GLD", 200)),
        "tech_trend":  (0.2, ("TQQQ", "QQQ", 200)),
        "fin_trend":   (0.2, ("FAS", "XLF", 200)),
    }),
    ("eq5_faster_ma", {
        "eq_trend":    (0.2, ("QLD", "SPY", 100)),
        "bond_trend":  (0.2, ("TMF", "TLT", 100)),
        "gold_trend":  (0.2, ("UGL", "GLD", 100)),
        "tech_trend":  (0.2, ("TQQQ", "QQQ", 100)),
        "fin_trend":   (0.2, ("FAS", "XLF", 100)),
    }),
    ("eq_weighted_6", {
        "eq_trend":    (0.17, ("QLD", "SPY", 200)),
        "bond_trend":  (0.17, ("TMF", "TLT", 200)),
        "gold_trend":  (0.17, ("UGL", "GLD", 200)),
        "tech_trend":  (0.17, ("TQQQ", "QQQ", 200)),
        "fin_trend":   (0.17, ("FAS", "XLF", 200)),
        "semi_trend":  (0.15, ("SOXL", "SMH", 200)),
    }),
    ("zephyr_plus_trends", {
        "ZEPHYR":      (0.50, None),
        "eq_trend":    (0.125, ("QLD", "SPY", 200)),
        "tech_trend":  (0.125, ("TQQQ", "QQQ", 200)),
        "gold_trend":  (0.125, ("UGL", "GLD", 200)),
        "semi_trend":  (0.125, ("SOXL", "SMH", 200)),
    }),
    ("zephyr_light_trends", {
        "ZEPHYR":      (0.30, None),
        "eq_trend":    (0.14, ("QLD", "SPY", 200)),
        "tech_trend":  (0.14, ("TQQQ", "QQQ", 200)),
        "bond_trend":  (0.14, ("TMF", "TLT", 200)),
        "gold_trend":  (0.14, ("UGL", "GLD", 200)),
        "semi_trend":  (0.14, ("SOXL", "SMH", 200)),
    }),
    ("all_trends_no_zephyr", {
        "eq_trend":    (0.14, ("QLD", "SPY", 200)),
        "tech_trend":  (0.15, ("TQQQ", "QQQ", 200)),
        "bond_trend":  (0.14, ("TMF", "TLT", 200)),
        "gold_trend":  (0.14, ("UGL", "GLD", 200)),
        "semi_trend":  (0.15, ("SOXL", "SMH", 200)),
        "fin_trend":   (0.14, ("FAS", "XLF", 200)),
        "spy_trend":   (0.14, ("UPRO", "SPY", 200)),
    }),
]

print(f"{'Name':<25}{'Sharpe':>8}{'Ret':>8}{'Vol':>7}{'MDD':>8}{'Sor':>6}{'Cal':>6}{'Yrs':>6}")
for name, weights in experiments:
    try:
        port, df = run_combo(weights, start="2014-01-01")
        m = metrics(port)
        if m is None: continue
        print(f"{name:<25}{m['sharpe']:8.2f}{m['ret']*100:7.1f}%{m['vol']*100:6.1f}%"
              f"{m['mdd']*100:7.1f}%{m['sortino']:6.2f}{m['calmar']:6.2f}{m['n']:6.1f}")
    except Exception as e:
        print(f"{name:<25}  ERROR: {e}")
        import traceback; traceback.print_exc()
