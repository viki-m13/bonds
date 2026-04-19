"""Verify that the chosen NOVA config (lookback=10, top_n=3, cap=0.33, btc_ma=200)
is not crypto-dominated over history. Print average weight per name and
fraction of time BTC/ETH are in the top-3."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"

EQUITY = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]
CRYPTO = ["BTC_USD", "ETH_USD"]

def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()

def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()

def spy_regime(dates, vix_cap=30, ma_len=200):
    spy = load_etf("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS")
    ok = (spy > spy.rolling(ma_len).mean())
    if vix is not None:
        v = vix.reindex(dates).ffill()
        ok = ok & (v < vix_cap)
    return ok.shift(1).fillna(False).astype(float)

def btc_trend(dates, ma_len=200):
    btc = load_etf("BTC_USD").reindex(dates).ffill()
    return (btc > btc.rolling(ma_len).mean()).shift(1).fillna(False).astype(float)

btc = load_etf("BTC_USD")
spy = load_etf("SPY")
dates = spy.loc[btc.index.min():].index

universe = EQUITY + CRYPTO
prices = pd.DataFrame({t: load_etf(t) for t in universe}).reindex(dates).ffill()
avail = prices.notna()
reg_eq = spy_regime(dates)
reg_bt = btc_trend(dates, 200)

lookback = 10
top_n = 3
cap = 0.33
weights = []  # effective weight per name on each day
current = pd.Series(0.0, index=universe)
last_idx = -5
for i in range(len(dates)):
    if i > lookback and i - last_idx >= 5:
        # Lag momentum by 1 bar (avoid look-ahead).
        live = avail.iloc[i - 1]
        momo = (prices.iloc[i - 1] / prices.iloc[i - 1 - lookback] - 1).where(live)
        ranked = momo.dropna().sort_values(ascending=False)
        positive = [t for t in ranked.index if momo[t] > 0]
        top = positive[:top_n]
        new = pd.Series(0.0, index=universe)
        if top:
            w = 1.0 / len(top)
            for t in top:
                new[t] = min(w, cap)
        current = new
        last_idx = i
    eff = current.copy()
    for t in EQUITY: eff[t] = current[t] * reg_eq.iloc[i]
    for t in CRYPTO: eff[t] = current[t] * reg_bt.iloc[i]
    weights.append(eff.values)

W = pd.DataFrame(weights, index=dates, columns=universe)
print(f"Window: {dates[0].date()} .. {dates[-1].date()} ({len(dates)/252:.1f}y)")
print("\n=== Average effective weight per name ===")
print((W.mean() * 100).round(2).sort_values(ascending=False).to_string())

print("\n=== Fraction of time each name had >0 weight ===")
print((W.gt(0).mean() * 100).round(1).sort_values(ascending=False).to_string())

crypto_w = W[CRYPTO].sum(axis=1)
equity_w = W[EQUITY].sum(axis=1)
cash_w = 1 - crypto_w - equity_w
print(f"\n=== Bucket-weight summary ===")
print(f"Crypto  avg={crypto_w.mean()*100:5.2f}%  max={crypto_w.max()*100:5.2f}%  median={crypto_w.median()*100:5.2f}%")
print(f"Equity  avg={equity_w.mean()*100:5.2f}%  max={equity_w.max()*100:5.2f}%  median={equity_w.median()*100:5.2f}%")
print(f"Cash    avg={cash_w.mean()*100:5.2f}%  max={cash_w.max()*100:5.2f}%  median={cash_w.median()*100:5.2f}%")

print(f"\nFraction of days crypto > 50% of book: {(crypto_w > 0.5).mean()*100:.1f}%")
print(f"Fraction of days crypto > 66% of book: {(crypto_w > 0.66).mean()*100:.1f}%")
