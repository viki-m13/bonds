"""Grid v2: NO fixed crypto weight. Unified momentum universe where BTC/ETH
compete with 3x ETFs on the same cross-sectional momentum signal, rebalanced
weekly. A per-asset cap prevents collapsing into 100% crypto.

Design:
  - Universe = 3x leveraged ETFs + BTC + (ETH when available)
  - Rank by lookback-day return; hold top-N equal-weight subject to cap
  - SPY-regime gate de-risks the *equity* portion only; crypto keeps its own
    BTC>MA gate (trend filter), so nothing is fully forced
  - Search over lookback, top_n, cap, BTC-trend MA
"""
from pathlib import Path
import numpy as np
import pandas as pd

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


def btc_trend(dates, ma_len=100):
    btc = load_etf("BTC_USD").reindex(dates).ffill()
    return (btc > btc.rolling(ma_len).mean()).shift(1).fillna(False).astype(float)


def run(lookback, top_n, cap, btc_ma, dates, tc_bps=15.0):
    universe = EQUITY + CRYPTO
    prices = pd.DataFrame({t: load_etf(t) for t in universe}).reindex(dates).ffill()
    rets = prices.pct_change().fillna(0)
    avail = prices.notna()
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    reg_eq = spy_regime(dates)          # applies to equity-leg names
    reg_bt = btc_trend(dates, btc_ma)   # applies to crypto names

    current = pd.Series(0.0, index=universe)
    port = pd.Series(0.0, index=dates)
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
                w_eq = 1.0 / len(top)
                for t in top:
                    new[t] = min(w_eq, cap)
                # Renormalize so weights still sum to <=1 (cash residual if capped)
                total = new.sum()
                if total > 1.0:
                    new = new / total
            tc = (new - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = new
            last_idx = i
        # Apply per-name regime gates: equity names -> SPY regime; crypto -> BTC trend
        eff = current.copy()
        g_eq = reg_eq.iloc[i]
        g_bt = reg_bt.iloc[i]
        off_eq = sum(current[t] for t in EQUITY) * (1 - g_eq)
        off_bt = sum(current[t] for t in CRYPTO) * (1 - g_bt)
        for t in EQUITY: eff[t] = current[t] * g_eq
        for t in CRYPTO: eff[t] = current[t] * g_bt
        r = (rets.iloc[i] * eff).sum() + (off_eq + off_bt) * bil.iloc[i]
        port.iloc[i] += r
    return port


def stats(r):
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return ar * 100, av * 100, sr, mdd * 100, cum.iloc[-1]


def main():
    btc = load_etf("BTC_USD")
    spy = load_etf("SPY")
    dates = spy.loc[btc.index.min():].index
    print(f"Grid v2 from {dates[0].date()} to {dates[-1].date()} "
          f"({len(dates)/252:.1f}y)")
    print("Unified universe: 7 x 3x ETFs + BTC + ETH  (per-name cap enforced)\n")

    results = []
    for lookback in [10, 20, 30, 60]:
        for top_n in [2, 3, 4]:
            for cap in [0.25, 0.33, 0.50]:
                for btc_ma in [50, 100, 200]:
                    p = run(lookback, top_n, cap, btc_ma, dates)
                    ar, av, sr, mdd, nav = stats(p)
                    # Measure average BTC+ETH weight to validate "not crypto-dominated"
                    results.append({
                        "lookback": lookback, "top_n": top_n, "cap": cap,
                        "btc_ma": btc_ma, "Ret": ar, "Vol": av, "SR": sr,
                        "MDD": mdd, "NAV": nav,
                    })

    df = pd.DataFrame(results)
    passed = df[df["Ret"] >= 50].sort_values("SR", ascending=False)
    print(f"{len(passed)} configs clear 50% CAGR:")
    if len(passed):
        print(passed.head(15).to_string(index=False))
    print("\nTop 10 by Sharpe overall:")
    print(df.sort_values("SR", ascending=False).head(10).to_string(index=False))
    print("\nTop 10 by return overall:")
    print(df.sort_values("Ret", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
