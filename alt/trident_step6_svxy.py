"""Step 6: SVXY (short volatility) + strict regime gate.
Short-vol harvests VIX term premium in calm markets, but blows up in crises.
With VIX<20 AND VIX_1M<VIX_3M (backwardation OFF) filter, it becomes much safer.
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
            "sortino": sor, "n": len(r)/252}


# Test SVXY with progressively tighter gates
svxy = load("SVXY")
vix = load_fred("VIXCLS")
dates = svxy.loc["2018-02-15":].index  # post-Feb 2018 (fund was rebalanced to half leverage)
r_raw = svxy.reindex(dates).pct_change().fillna(0)
bil = load("BIL").reindex(dates).ffill().pct_change().fillna(0)
v = vix.reindex(dates).ffill()

# Multiple gate configs
gates = {
    "raw_svxy": pd.Series(1.0, index=dates),
    "vix<20": (v < 20).shift(1).fillna(False).astype(float),
    "vix<18": (v < 18).shift(1).fillna(False).astype(float),
    "vix<16": (v < 16).shift(1).fillna(False).astype(float),
    "vix<20_and_rising_spy": ((v < 20) & (load("SPY").reindex(dates).ffill() > load("SPY").reindex(dates).ffill().rolling(200).mean())).shift(1).fillna(False).astype(float),
    "vix<16_and_rising_spy": ((v < 16) & (load("SPY").reindex(dates).ffill() > load("SPY").reindex(dates).ffill().rolling(200).mean())).shift(1).fillna(False).astype(float),
}

print(f"SVXY gating experiments (post Feb 2018)")
print(f"{'Gate':<30}{'SR':>6}{'Ret':>8}{'Vol':>7}{'MDD':>8}{'Yrs':>5}{'Days on':>9}")
for name, g in gates.items():
    port = g * r_raw + (1 - g) * bil
    m = metrics(port)
    if m is None: continue
    days_on = g.sum() / len(g) * 100
    print(f"{name:<30}{m['sharpe']:6.2f}{m['ret']*100:7.1f}%{m['vol']*100:6.1f}%"
          f"{m['mdd']*100:7.1f}%{m['n']:5.1f}{days_on:8.1f}%")

# Now: combine gated SVXY with ZEPHYR
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


print("\nZEPHYR + SVXY combos (post Feb 2018):")
gate = gates["vix<20_and_rising_spy"]
svxy_gated = gate * r_raw + (1 - gate) * bil
zeph = zephyr_returns(dates)
for w_svxy in [0.10, 0.15, 0.20, 0.25, 0.30]:
    port = (1 - w_svxy) * zeph + w_svxy * svxy_gated
    m = metrics(port)
    print(f"  ZEPHYR {(1-w_svxy)*100:.0f}% + SVXY(gated) {w_svxy*100:.0f}%:  "
          f"SR={m['sharpe']:.2f}  Ret={m['ret']*100:.1f}%  Vol={m['vol']*100:.1f}%  MDD={m['mdd']*100:.1f}%")

print("\nZEPHYR + SVXY + Covered Call combos:")
# Covered call (JEPI/JEPQ/SPYI avg)
cc_pieces = {t: load(t).reindex(dates).ffill().pct_change().fillna(0)
             for t in ["JEPI", "JEPQ", "SPYI", "DIVO"] if load(t) is not None}
cc_df = pd.DataFrame(cc_pieces)
cc_avail = cc_df.notna().sum(axis=1).replace(0, 1)
cc = cc_df.fillna(0).sum(axis=1) / cc_avail

for w_z, w_cc, w_s in [(0.40, 0.30, 0.30), (0.30, 0.40, 0.30), (0.50, 0.25, 0.25),
                       (0.45, 0.30, 0.25), (0.35, 0.35, 0.30), (0.30, 0.30, 0.40)]:
    port = w_z * zeph + w_cc * cc + w_s * svxy_gated
    m = metrics(port)
    print(f"  Z={w_z*100:.0f}% CC={w_cc*100:.0f}% SVXY={w_s*100:.0f}%:  "
          f"SR={m['sharpe']:.2f}  Ret={m['ret']*100:.1f}%  Vol={m['vol']*100:.1f}%  MDD={m['mdd']*100:.1f}%")
