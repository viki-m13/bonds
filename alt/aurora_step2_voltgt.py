"""Step 2: Weekly momentum rotation with vol-targeting AT REBALANCE ONLY.
Each week, size the sleeve so its forward vol (est. from last 63d) is ~V target.
Weights are held fixed for 5 days — no daily adjustment.
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
            "sortino": sor, "calmar": ar/abs(dd) if dd < 0 else 0, "n": len(r)/252}


def momo_voltgt(universe, lookback, top_n, dates, rebal_days=5, vol_target=0.10,
                vol_window=63, max_exposure=1.5, tc_bps=15.0, regime=None,
                cash_ticker="BIL"):
    """Weekly momentum rotation with exposure scaled at rebalance to target vol."""
    prices = pd.DataFrame({t: load(t) for t in universe}).dropna()
    prices = prices.reindex(dates).ffill().dropna()
    rets = prices.pct_change().fillna(0)
    d2 = rets.index
    cash = load(cash_ticker).reindex(d2).ffill().pct_change().fillna(0)
    current = pd.Series(0.0, index=prices.columns)
    cash_w = 1.0
    port = pd.Series(0.0, index=d2)
    last_idx = -rebal_days
    for i in range(len(d2)):
        if i >= max(lookback, vol_window) and i - last_idx >= rebal_days:
            momo = prices.iloc[i] / prices.iloc[i - lookback] - 1
            ranked = momo.sort_values(ascending=False)
            positive = [t for t in ranked.index if momo[t] > 0]
            top = positive[:top_n]
            # Equal-weighted sleeve among selected, then scale whole sleeve to target vol
            new_target = pd.Series(0.0, index=prices.columns)
            if top:
                # Estimate sleeve vol from last vol_window days assuming equal weights
                w_eq = 1.0 / len(top)
                sleeve_rets = (rets[top].iloc[i - vol_window:i] * w_eq).sum(axis=1)
                sleeve_vol = sleeve_rets.std() * np.sqrt(252)
                scale = min(max_exposure, vol_target / max(sleeve_vol, 0.01))
                for t in top:
                    new_target[t] = w_eq * scale
            new_cash = 1.0 - new_target.sum()
            tc = (new_target - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = new_target
            cash_w = new_cash
            last_idx = i
        r_assets = (rets.iloc[i] * current).sum()
        r_cash = cash.iloc[i] * cash_w
        r = r_assets + r_cash
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


dates = load("SPY").loc["2014-01-01":].index
regime = spy_regime(dates)
U_3X = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]

# Sweep vol targets
print("Vol-targeted weekly momo on 3x universe (lb=20, top=3):")
print(f"{'VolTgt':<8}{'MaxExp':<8}{'SR':>6}{'Ret':>8}{'Vol':>7}{'MDD':>8}{'Cal':>6}")
for vt in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    for me in [1.0, 1.5, 2.0]:
        port = momo_voltgt(U_3X, 20, 3, dates, rebal_days=5, vol_target=vt,
                           max_exposure=me, regime=regime)
        m = metrics(port)
        print(f"  {vt:.2f}    {me:.1f}     {m['sharpe']:6.2f}{m['ret']*100:7.1f}%"
              f"{m['vol']*100:6.1f}%{m['mdd']*100:7.1f}%{m['calmar']:6.2f}")

# Best config + ZEPHYR combo
print("\nBest configs + ZEPHYR:")
best = momo_voltgt(U_3X, 20, 3, dates, rebal_days=5, vol_target=0.10, max_exposure=1.5, regime=regime)
zeph = zephyr_returns(dates)
for w_z in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
    port = w_z * zeph + (1 - w_z) * best
    m = metrics(port)
    print(f"  ZEPHYR {w_z*100:.0f}% / Momo 10%vol {(1-w_z)*100:.0f}%:  "
          f"SR={m['sharpe']:.2f}  Ret={m['ret']*100:.1f}%  Vol={m['vol']*100:.1f}%  MDD={m['mdd']*100:.1f}%")

# Higher-vol config
best2 = momo_voltgt(U_3X, 20, 3, dates, rebal_days=5, vol_target=0.20, max_exposure=2.0, regime=regime)
for w_z in [0.3, 0.4, 0.5, 0.6, 0.7]:
    port = w_z * zeph + (1 - w_z) * best2
    m = metrics(port)
    print(f"  ZEPHYR {w_z*100:.0f}% / Momo 20%vol {(1-w_z)*100:.0f}%:  "
          f"SR={m['sharpe']:.2f}  Ret={m['ret']*100:.1f}%  Vol={m['vol']*100:.1f}%  MDD={m['mdd']*100:.1f}%")
