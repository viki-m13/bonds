"""Step 5: Levered ZEPHYR core + momo overlay.
ZEPHYR has Sharpe 2.49 unlevered (2.1% ret, 0.8% vol) — lever the base 3-6x
with BIL financing, then add a momo overlay for return juice.
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
    return {"sharpe": ar/av if av > 0 else 0, "ret": ar, "vol": av, "mdd": dd,
            "n": len(r)/252}


def weekly_momo(universe, lookback, top_n, dates, rebal_days=5,
                tc_bps=15.0, regime=None, cash_ticker="BIL"):
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
            positive = [t for t in ranked.index if momo[t] > 0]
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


def zephyr_max_returns(dates):
    """Higher-return CLO variant: CLOI core, JBBB mezz, more float."""
    portfolio = {"CLOI": 0.40, "JBBB": 0.20, "JAAA": 0.15, "BKLN": 0.10,
                 "SRLN": 0.10, "GLD": 0.05}
    pieces = {}
    for t, _ in portfolio.items():
        p = load(t)
        if p is None: continue
        pieces[t] = p.reindex(dates).ffill().pct_change().fillna(0)
    df = pd.DataFrame(pieces)
    # Renormalize each row by available weight
    w_full = pd.Series({t: portfolio[t] for t in pieces})
    # Mask zero rows (not yet trading) — use only active tickers each day
    active = (df != 0).astype(float)
    w_day = active * w_full
    w_day_sum = w_day.sum(axis=1).replace(0, np.nan)
    port = (df * w_day).sum(axis=1) / w_day_sum
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    port = port.fillna(bil)
    # Regime gates
    hy = load_fred("BAMLH0A0HYM2")
    y = load_fred("DGS10")
    hy_g = ((8.0 - hy.reindex(dates).ffill()) / (8.0 - 5.0)).clip(0, 1).shift(1).fillna(1.0)
    rt_g = (y.reindex(dates).ffill() - y.reindex(dates).ffill().shift(63) < 0.7).shift(1).fillna(True).astype(float)
    g = hy_g * rt_g
    return g * port + (1 - g) * bil


def covered_call(dates):
    pieces = {}
    for t in ["JEPI", "JEPQ", "SPYI", "DIVO"]:
        p = load(t)
        if p is None: continue
        pieces[t] = p.reindex(dates).ffill().pct_change().fillna(0)
    df = pd.DataFrame(pieces)
    avail = (df != 0).astype(int).sum(axis=1).replace(0, np.nan)
    ret = df.sum(axis=1) / avail.fillna(1)
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    return ret.where(avail.notna(), bil).fillna(0)


def managed_futures(dates):
    pieces = {}
    for t in ["DBMF", "CTA", "KMLM"]:
        p = load(t)
        if p is None: continue
        pieces[t] = p.reindex(dates).ffill().pct_change().fillna(0)
    df = pd.DataFrame(pieces)
    avail = (df != 0).astype(int).sum(axis=1).replace(0, np.nan)
    ret = df.sum(axis=1) / avail.fillna(1)
    bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
    return ret.where(avail.notna(), bil).fillna(0)


def spy_regime(dates, vix_cap=30, ma_len=200):
    spy = load("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS")
    ok = (spy > spy.rolling(ma_len).mean())
    if vix is not None:
        v = vix.reindex(dates).ffill()
        ok = ok & (v < vix_cap)
    return ok.shift(1).fillna(False).astype(float)


def levered(sleeve, lev, bil):
    """Return levered sleeve with BIL financing on the extra notional."""
    return lev * sleeve - max(lev - 1.0, 0.0) * bil


print("Building sleeves...")
dates = load("SPY").loc["2014-01-01":].index
regime = spy_regime(dates)
U_3X = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]
bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)

sleeve_Z = zephyr_returns(dates)
sleeve_Zmax = zephyr_max_returns(dates)
sleeve_C = covered_call(dates)
sleeve_M = weekly_momo(U_3X, 20, 3, dates, rebal_days=5, regime=regime)
sleeve_F = managed_futures(dates)

print("\nUnlevered sleeves:")
for name, s in [("ZEPHYR", sleeve_Z), ("ZEPHYR-MAX", sleeve_Zmax),
                ("Income", sleeve_C), ("Momo", sleeve_M), ("MF", sleeve_F)]:
    m = metrics(s)
    print(f"  {name:<12} SR={m['sharpe']:.2f} Ret={m['ret']*100:.1f}% Vol={m['vol']*100:.1f}% MDD={m['mdd']*100:.1f}% n={m['n']:.1f}y")

print("\nLevered ZEPHYR (BIL financing):")
for lev in [1, 2, 3, 4, 5, 6, 8]:
    m = metrics(levered(sleeve_Z, lev, bil))
    print(f"  {lev}x: SR={m['sharpe']:.2f} Ret={m['ret']*100:.1f}% Vol={m['vol']*100:.1f}% MDD={m['mdd']*100:.1f}%")

print("\nLevered ZEPHYR-MAX (BIL financing):")
for lev in [1, 2, 3, 4, 5, 6]:
    m = metrics(levered(sleeve_Zmax, lev, bil))
    print(f"  {lev}x: SR={m['sharpe']:.2f} Ret={m['ret']*100:.1f}% Vol={m['vol']*100:.1f}% MDD={m['mdd']*100:.1f}%")

print("\n--- Levered-ZEPHYR + Momo combos ---")
for lev_z in [3, 4, 5, 6]:
    base = levered(sleeve_Z, lev_z, bil)
    for wm in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]:
        wz = 1.0 - wm
        port = wz * base + wm * sleeve_M
        m = metrics(port)
        print(f"  {lev_z}xZ {wz*100:.0f}% + Momo {wm*100:.0f}%: "
              f"SR={m['sharpe']:.2f} Ret={m['ret']*100:.1f}% Vol={m['vol']*100:.1f}% MDD={m['mdd']*100:.1f}%")

print("\n--- Levered-ZEPHYR-MAX + Momo combos ---")
for lev_z in [2, 3, 4, 5]:
    base = levered(sleeve_Zmax, lev_z, bil)
    for wm in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]:
        wz = 1.0 - wm
        port = wz * base + wm * sleeve_M
        m = metrics(port)
        print(f"  {lev_z}xZmax {wz*100:.0f}% + Momo {wm*100:.0f}%: "
              f"SR={m['sharpe']:.2f} Ret={m['ret']*100:.1f}% Vol={m['vol']*100:.1f}% MDD={m['mdd']*100:.1f}%")

print("\n--- 4-sleeve with levered ZEPHYR core ---")
best = []
for lev_z in [3, 4, 5]:
    base = levered(sleeve_Z, lev_z, bil)
    for wz in np.arange(0.3, 1.01, 0.1):
        for wm in np.arange(0.0, 1.01 - wz, 0.1):
            for wc in np.arange(0.0, 1.01 - wz - wm, 0.1):
                wf = 1.0 - wz - wm - wc
                if wf < -0.01: continue
                wf = max(0.0, wf)
                port = wz*base + wm*sleeve_M + wc*sleeve_C + wf*sleeve_F
                m = metrics(port)
                if m is None: continue
                best.append((m['sharpe'], m['ret'], m['vol'], m['mdd'],
                             lev_z, wz, wm, wc, wf))

best.sort(reverse=True)
print("\nTop 15 by Sharpe, Ret >= 20%:")
filtered = [x for x in best if x[1] >= 0.20]
for sr, ret, vol, mdd, lz, wz, wm, wc, wf in filtered[:15]:
    print(f"  SR={sr:.2f} Ret={ret*100:.1f}% Vol={vol*100:.1f}% MDD={mdd*100:.1f}% "
          f"| {lz}xZ={wz:.1f} M={wm:.1f} C={wc:.1f} F={wf:.1f}")

print("\nTop 15 by Sharpe, Ret >= 15%:")
filtered = [x for x in best if x[1] >= 0.15]
for sr, ret, vol, mdd, lz, wz, wm, wc, wf in filtered[:15]:
    print(f"  SR={sr:.2f} Ret={ret*100:.1f}% Vol={vol*100:.1f}% MDD={mdd*100:.1f}% "
          f"| {lz}xZ={wz:.1f} M={wm:.1f} C={wc:.1f} F={wf:.1f}")
