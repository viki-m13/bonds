"""Step 5: NOVA — Diversified portfolio combining:
  (a) ZEPHYR core (ultra-short credit CLOs)       — Sharpe ~3.7, 4.4% ret, 1.2% vol
  (b) Covered-call income (JEPI/JEPQ/SPYI)        — Sharpe ~1.0, 12-15% ret, 12-17% vol
  (c) Trend engines on leveraged ETFs             — gated by 200dma of underlying
  (d) Crypto trend engine                         — only when BTC > 200dma

All weights FIXED. Monthly rebalance. ZERO vol scaling.
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
    u = load(underlying).reindex(dates).ffill()
    lev = load(levered).reindex(dates).ffill().pct_change().fillna(0)
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    ma = u.rolling(ma_len).mean()
    gate = (u > ma).shift(1).fillna(False).astype(float)
    return gate * lev + (1 - gate) * bil


def zephyr_returns(dates):
    portfolio = {"JAAA": 0.32, "JPST": 0.28, "MINT": 0.15,
                 "BKLN": 0.10, "SRLN": 0.05, "FLOT": 0.05, "GLD": 0.05}
    prices = pd.DataFrame({t: load(t) for t in portfolio}).dropna()
    rets = prices.pct_change().fillna(0).reindex(dates).fillna(0)
    w = pd.Series(portfolio)
    port = (rets * w).sum(axis=1)
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    hy = load_fred("BAMLH0A0HYM2")
    y = load_fred("DGS10")
    hy_g = ((8.0 - hy.reindex(dates).ffill()) / (8.0 - 5.0)).clip(0, 1).shift(1).fillna(1.0)
    rt_g = (y.reindex(dates).ffill() - y.reindex(dates).ffill().shift(63) < 0.7).shift(1).fillna(True).astype(float)
    g = hy_g * rt_g
    return g * port + (1 - g) * bil


def covered_call_sleeve(dates):
    """Average of JEPI, JEPQ, SPYI, DIVO (whichever are available at each date)."""
    pieces = {}
    for t in ["JEPI", "JEPQ", "SPYI", "DIVO"]:
        p = load(t)
        if p is None: continue
        pieces[t] = p.reindex(dates).ffill().pct_change().fillna(0)
    df = pd.DataFrame(pieces)
    # Equal weight among available funds each day (normalise by available count)
    avail = df.notna() & (df != 0)
    n = avail.sum(axis=1).replace(0, np.nan)
    ret = df.fillna(0).sum(axis=1) / n.fillna(1)
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    ret = ret.where(n > 0, bil)
    return ret


def build_portfolio(weights, start="2014-01-01"):
    tmp = load("SPY")
    dates = tmp.loc[start:].index
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    sleeves = {}
    for name, (w, spec) in weights.items():
        if spec == "ZEPHYR":
            sleeves[name] = zephyr_returns(dates)
        elif spec == "COV_CALL":
            sleeves[name] = covered_call_sleeve(dates)
        elif spec == "BIL":
            sleeves[name] = bil
        else:
            lev, und, ma = spec
            sleeves[name] = trend_engine(lev, und, dates, ma)
    df = pd.DataFrame(sleeves).fillna(0)
    wser = pd.Series({k: v[0] for k, v in weights.items()})
    port = (df * wser).sum(axis=1)
    return port, df


experiments = [
    ("N1_balanced", {
        "zephyr":   (0.50, "ZEPHYR"),
        "covcall":  (0.20, "COV_CALL"),
        "eq":       (0.08, ("QLD", "SPY", 200)),
        "tech":     (0.08, ("TQQQ", "QQQ", 200)),
        "bond":     (0.05, ("TMF", "TLT", 200)),
        "gold":     (0.05, ("UGL", "GLD", 200)),
        "semi":     (0.04, ("SOXL", "SMH", 200)),
    }),
    ("N2_return_tilt", {
        "zephyr":   (0.30, "ZEPHYR"),
        "covcall":  (0.30, "COV_CALL"),
        "tech":     (0.15, ("TQQQ", "QQQ", 200)),
        "eq":       (0.10, ("QLD", "SPY", 200)),
        "semi":     (0.08, ("SOXL", "SMH", 200)),
        "gold":     (0.04, ("UGL", "GLD", 200)),
        "bond":     (0.03, ("TMF", "TLT", 200)),
    }),
    ("N3_bold", {
        "zephyr":   (0.20, "ZEPHYR"),
        "covcall":  (0.30, "COV_CALL"),
        "tech":     (0.20, ("TQQQ", "QQQ", 200)),
        "eq":       (0.15, ("UPRO", "SPY", 200)),
        "semi":     (0.10, ("SOXL", "SMH", 200)),
        "bond":     (0.05, ("TMF", "TLT", 200)),
    }),
    ("N4_covcall_heavy", {
        "zephyr":   (0.30, "ZEPHYR"),
        "covcall":  (0.50, "COV_CALL"),
        "tech":     (0.10, ("TQQQ", "QQQ", 200)),
        "eq":       (0.10, ("QLD", "SPY", 200)),
    }),
    ("N5_maxreturn", {
        "zephyr":   (0.15, "ZEPHYR"),
        "covcall":  (0.35, "COV_CALL"),
        "tech":     (0.25, ("TQQQ", "QQQ", 200)),
        "eq":       (0.15, ("UPRO", "SPY", 200)),
        "semi":     (0.10, ("SOXL", "SMH", 200)),
    }),
    ("N6_allcrypto", {
        "zephyr":   (0.40, "ZEPHYR"),
        "covcall":  (0.30, "COV_CALL"),
        "tech":     (0.10, ("TQQQ", "QQQ", 200)),
        "eq":       (0.08, ("QLD", "SPY", 200)),
        "gold":     (0.05, ("UGL", "GLD", 200)),
        "crypto":   (0.05, ("BITO", "BITO", 200)),
        "bond":     (0.02, ("TMF", "TLT", 200)),
    }),
]

print(f"{'Name':<20}{'SR':>6}{'Ret':>8}{'Vol':>7}{'MDD':>8}{'Sor':>6}{'Cal':>6}{'Yrs':>6}  Period")
for name, w in experiments:
    port, df = build_portfolio(w, start="2014-01-01")
    m = metrics(port)
    if m is None: continue
    print(f"{name:<20}{m['sharpe']:6.2f}{m['ret']*100:7.1f}%{m['vol']*100:6.1f}%"
          f"{m['mdd']*100:7.1f}%{m['sortino']:6.2f}{m['calmar']:6.2f}{m['n']:6.1f}  {m['start']}..{m['end']}")
