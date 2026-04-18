"""Step 7: Regime-conditional sleeve allocation.
In calm regime (VIX<18 + SPY>200dma): heavy momo + covered call (20%+ targets)
In normal regime: balanced mix
In stress (VIX>25 or SPY<200dma): all ZEPHYR + MF
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
            "n": len(r)/252, "start": str(r.index[0].date())}


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


def classify_regime(dates):
    """Return regime integer per day: 2=calm, 1=normal, 0=stress."""
    spy = load("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    ma200 = spy.rolling(200).mean()
    ma50 = spy.rolling(50).mean()
    calm = (vix < 18) & (spy > ma50) & (spy > ma200)
    stress = (vix > 25) | (spy < ma200)
    reg = pd.Series(1, index=dates)
    reg[calm] = 2
    reg[stress] = 0
    return reg.shift(1).fillna(1)


print("Building sleeves...")
dates = load("SPY").loc["2014-01-01":].index
basic_reg = spy_regime(dates)
regime3 = classify_regime(dates)
U_3X = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]

sleeve_Z = zephyr_returns(dates)
sleeve_C = covered_call(dates)
sleeve_M = weekly_momo(U_3X, 20, 3, dates, rebal_days=5, regime=basic_reg)
sleeve_F = managed_futures(dates)

print("Regime distribution:")
print(f"  Calm (2):   {(regime3 == 2).sum()} days  ({(regime3 == 2).mean()*100:.1f}%)")
print(f"  Normal (1): {(regime3 == 1).sum()} days  ({(regime3 == 1).mean()*100:.1f}%)")
print(f"  Stress (0): {(regime3 == 0).sum()} days  ({(regime3 == 0).mean()*100:.1f}%)")

# Define weight vectors per regime: (wZ, wC, wM, wF)
# Calm: heavy momo + CC
# Normal: balanced
# Stress: all Z + MF
configs = [
    ("aggressive_calm", {
        2: (0.10, 0.30, 0.50, 0.10),   # calm
        1: (0.30, 0.30, 0.25, 0.15),   # normal
        0: (0.70, 0.10, 0.00, 0.20),   # stress
    }),
    ("moderate", {
        2: (0.20, 0.30, 0.40, 0.10),
        1: (0.40, 0.25, 0.20, 0.15),
        0: (0.80, 0.10, 0.00, 0.10),
    }),
    ("balanced", {
        2: (0.30, 0.25, 0.30, 0.15),
        1: (0.50, 0.20, 0.15, 0.15),
        0: (0.80, 0.10, 0.00, 0.10),
    }),
    ("max_return", {
        2: (0.00, 0.30, 0.60, 0.10),
        1: (0.20, 0.30, 0.30, 0.20),
        0: (0.60, 0.10, 0.10, 0.20),
    }),
    ("defensive", {
        2: (0.40, 0.25, 0.25, 0.10),
        1: (0.60, 0.15, 0.10, 0.15),
        0: (0.90, 0.05, 0.00, 0.05),
    }),
]

def regime_portfolio(weight_map, regime_series):
    """Build return series using regime-conditional weights."""
    wZ = regime_series.map({k: v[0] for k, v in weight_map.items()}).astype(float)
    wC = regime_series.map({k: v[1] for k, v in weight_map.items()}).astype(float)
    wM = regime_series.map({k: v[2] for k, v in weight_map.items()}).astype(float)
    wF = regime_series.map({k: v[3] for k, v in weight_map.items()}).astype(float)
    port = wZ * sleeve_Z + wC * sleeve_C + wM * sleeve_M + wF * sleeve_F
    return port

print(f"\n{'Config':<20}{'SR':>6}{'Ret':>8}{'Vol':>7}{'MDD':>8}  calm/normal/stress")
for name, wmap in configs:
    port = regime_portfolio(wmap, regime3)
    m = metrics(port)
    cw = wmap[2]; nw = wmap[1]; sw = wmap[0]
    ws = f"(Z{cw[0]:.1f}C{cw[1]:.1f}M{cw[2]:.1f}F{cw[3]:.1f})/(Z{nw[0]:.1f}C{nw[1]:.1f}M{nw[2]:.1f}F{nw[3]:.1f})/(Z{sw[0]:.1f}C{sw[1]:.1f}M{sw[2]:.1f}F{sw[3]:.1f})"
    print(f"{name:<20}{m['sharpe']:6.2f}{m['ret']*100:7.1f}%{m['vol']*100:6.1f}%"
          f"{m['mdd']*100:7.1f}%  {ws}")

# Grid search over regime weights
print("\nGrid search over calm/normal regime weights (stress fixed at 80%Z/10%C/0%M/10%F):")
stress_w = (0.80, 0.10, 0.00, 0.10)
best = []
for cZ in np.arange(0.0, 0.41, 0.1):
    for cC in np.arange(0.0, 0.51, 0.1):
        for cM in np.arange(0.0, 0.71, 0.1):
            cF = 1.0 - cZ - cC - cM
            if cF < -0.01 or cF > 0.41: continue
            cF = max(0.0, cF)
            for nZ in np.arange(0.1, 0.71, 0.2):
                for nC in np.arange(0.0, 0.41, 0.2):
                    for nM in np.arange(0.0, 0.51, 0.2):
                        nF = 1.0 - nZ - nC - nM
                        if nF < -0.01 or nF > 0.51: continue
                        nF = max(0.0, nF)
                        wmap = {2: (cZ, cC, cM, cF), 1: (nZ, nC, nM, nF), 0: stress_w}
                        port = regime_portfolio(wmap, regime3)
                        m = metrics(port)
                        if m is None: continue
                        best.append((m['sharpe'], m['ret'], m['vol'], m['mdd'],
                                     (cZ, cC, cM, cF), (nZ, nC, nM, nF)))

best.sort(reverse=True)
print(f"\nTotal configs tested: {len(best)}")

print("\nTop 10 by Sharpe overall:")
for sr, ret, vol, mdd, cw, nw in best[:10]:
    print(f"  SR={sr:.2f} Ret={ret*100:.1f}% Vol={vol*100:.1f}% MDD={mdd*100:.1f}% "
          f"| Calm Z{cw[0]:.1f}C{cw[1]:.1f}M{cw[2]:.1f}F{cw[3]:.1f}  Normal Z{nw[0]:.1f}C{nw[1]:.1f}M{nw[2]:.1f}F{nw[3]:.1f}")

print("\nTop 10 by Sharpe with Ret >= 15%:")
filtered = [x for x in best if x[1] >= 0.15]
for sr, ret, vol, mdd, cw, nw in filtered[:10]:
    print(f"  SR={sr:.2f} Ret={ret*100:.1f}% Vol={vol*100:.1f}% MDD={mdd*100:.1f}% "
          f"| Calm Z{cw[0]:.1f}C{cw[1]:.1f}M{cw[2]:.1f}F{cw[3]:.1f}  Normal Z{nw[0]:.1f}C{nw[1]:.1f}M{nw[2]:.1f}F{nw[3]:.1f}")

print("\nTop 10 by Sharpe with Ret >= 20%:")
filtered = [x for x in best if x[1] >= 0.20]
for sr, ret, vol, mdd, cw, nw in filtered[:10]:
    print(f"  SR={sr:.2f} Ret={ret*100:.1f}% Vol={vol*100:.1f}% MDD={mdd*100:.1f}% "
          f"| Calm Z{cw[0]:.1f}C{cw[1]:.1f}M{cw[2]:.1f}F{cw[3]:.1f}  Normal Z{nw[0]:.1f}C{nw[1]:.1f}M{nw[2]:.1f}F{nw[3]:.1f}")
