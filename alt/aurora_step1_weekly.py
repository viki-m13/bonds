"""Step 1 of AURORA: weekly momentum rotation on leveraged ETFs
combined with ZEPHYR base. Sweep parameters."""
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


def weekly_momo(universe, lookback, top_n, dates, rebal_days=5,
                tc_bps=15.0, regime=None, abs_momo_floor=0.0, cash_ticker="BIL"):
    prices = pd.DataFrame({t: load(t) for t in universe}).dropna()
    prices = prices.reindex(dates).ffill().dropna()
    rets = prices.pct_change().fillna(0)
    d2 = rets.index
    cash = load(cash_ticker).reindex(d2).ffill().pct_change().fillna(0)
    current = pd.Series(0.0, index=prices.columns)
    port = pd.Series(0.0, index=d2)
    last_idx = -rebal_days
    for i in range(len(d2)):
        if i >= lookback and i - last_idx >= rebal_days:
            momo = prices.iloc[i] / prices.iloc[i - lookback] - 1
            ranked = momo.sort_values(ascending=False)
            # Abs-momo filter: must beat floor
            positive = [t for t in ranked.index if momo[t] > abs_momo_floor]
            top = positive[:top_n]
            new_target = pd.Series(0.0, index=prices.columns)
            if top:
                w = 1.0 / len(top)
                for t in top:
                    new_target[t] = w
            tc = (new_target - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = new_target
            last_idx = i
        r = (rets.iloc[i] * current).sum()
        g = float(regime.iloc[i]) if regime is not None else 1.0
        r = g * r + (1 - g) * cash.iloc[i]
        port.iloc[i] += r
    return port


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


def spy_regime(dates, vix_cap=30, ma_len=200):
    spy = load("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS")
    ok = (spy > spy.rolling(ma_len).mean())
    if vix is not None:
        v = vix.reindex(dates).ffill()
        ok = ok & (v < vix_cap)
    return ok.shift(1).fillna(False).astype(float)


# Build common date index
dates = load("SPY").loc["2014-01-01":].index

# Universes
U_3X = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]
U_BIG = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL", "BITO",
         "QLD", "SSO", "ERX", "UCO", "UBT"]
U_MIX = ["TQQQ", "UPRO", "SOXL", "TECL", "TMF", "UGL", "BITO", "QQQ", "GLD"]

# Sweep: universe, lookback (days), top_n, rebal_days, regime
regime = spy_regime(dates)
experiments = [
    ("3x_lb20_top2_rb5",    U_3X, 20, 2, 5),
    ("3x_lb20_top1_rb5",    U_3X, 20, 1, 5),
    ("3x_lb40_top2_rb5",    U_3X, 40, 2, 5),
    ("3x_lb60_top2_rb5",    U_3X, 60, 2, 5),
    ("3x_lb20_top3_rb5",    U_3X, 20, 3, 5),
    ("3x_lb20_top2_rb10",   U_3X, 20, 2, 10),
    ("3x_lb10_top2_rb5",    U_3X, 10, 2, 5),
    ("big_lb20_top2_rb5",   U_BIG, 20, 2, 5),
    ("big_lb20_top3_rb5",   U_BIG, 20, 3, 5),
    ("big_lb40_top3_rb5",   U_BIG, 40, 3, 5),
    ("mix_lb20_top2_rb5",   U_MIX, 20, 2, 5),
]

print(f"Weekly momentum rotation (with SPY>200dma + VIX<30 regime gate)")
print(f"{'Name':<24}{'SR':>6}{'Ret':>8}{'Vol':>7}{'MDD':>8}{'Cal':>6}{'Yrs':>6}")
for name, uni, lb, tn, rb in experiments:
    port = weekly_momo(uni, lb, tn, dates, rebal_days=rb, regime=regime)
    m = metrics(port)
    print(f"{name:<24}{m['sharpe']:6.2f}{m['ret']*100:7.1f}%{m['vol']*100:6.1f}%"
          f"{m['mdd']*100:7.1f}%{m['calmar']:6.2f}{m['n']:6.1f}")

# Now the combo: best weekly + ZEPHYR core
print("\n--- Best weekly momo + ZEPHYR base (different splits) ---")
best_port = weekly_momo(U_BIG, 20, 3, dates, rebal_days=5, regime=regime)
zeph = zephyr_returns(dates)

for w_z in [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    port = w_z * zeph + (1 - w_z) * best_port
    m = metrics(port)
    print(f"  ZEPHYR {w_z*100:.0f}% / Weekly-Momo {(1-w_z)*100:.0f}%:  "
          f"SR={m['sharpe']:.2f}  Ret={m['ret']*100:.1f}%  Vol={m['vol']*100:.1f}%  "
          f"MDD={m['mdd']*100:.1f}%  Cal={m['calmar']:.2f}")
