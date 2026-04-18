"""Grid search for a MAX-GROWTH strategy targeting 50%+ CAGR.

Core ideas:
  - Concentrated top-1 (or top-2) momentum on 3x leveraged ETFs
  - BTC core for uncorrelated high-return exposure
  - Regime gates to cut drawdowns

The search evaluates combinations and reports the ones with annual return
>= 50% and the highest Sharpe among those.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"

MOMO_UNIVERSE = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]


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


def btc_regime(dates, ma_len=50):
    btc = load_etf("BTC_USD")
    b = btc.reindex(dates).ffill()
    ok = (b > b.rolling(ma_len).mean()).shift(1).fillna(False).astype(float)
    return ok


def top_n_momo(universe, lookback, top_n, dates, rebal_days=5, regime=None,
               tc_bps=15.0):
    prices = pd.DataFrame({t: load_etf(t) for t in universe})
    prices = prices.reindex(dates).ffill()
    rets = prices.pct_change().fillna(0)
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    current = pd.Series(0.0, index=prices.columns)
    port = pd.Series(0.0, index=dates)
    last_idx = -rebal_days
    avail = prices.notna().all(axis=1)
    for i in range(len(dates)):
        if avail.iloc[i] and i >= lookback and i - last_idx >= rebal_days:
            momo = prices.iloc[i] / prices.iloc[i - lookback] - 1
            ranked = momo.dropna().sort_values(ascending=False)
            positive = [t for t in ranked.index if momo[t] > 0]
            top = positive[:top_n]
            new_target = pd.Series(0.0, index=prices.columns)
            if top:
                w = 1.0 / len(top)
                for t in top: new_target[t] = w
            tc = (new_target - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = new_target
            last_idx = i
        r = (rets.iloc[i] * current).sum()
        g = float(regime.iloc[i]) if regime is not None else 1.0
        r = g * r + (1 - g) * bil.iloc[i]
        port.iloc[i] += r
    return port


def btc_gated(dates, ma=50):
    btc = load_etf("BTC_USD").reindex(dates).ffill()
    r = btc.pct_change().fillna(0)
    g = btc_regime(dates, ma_len=ma)
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    return g * r + (1 - g) * bil


def stats(r):
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return ar * 100, av * 100, sr, mdd * 100, cum.iloc[-1]


def main():
    btc = load_etf("BTC_USD")
    start = btc.index.min()
    spy = load_etf("SPY")
    dates = spy.loc[start:].index
    print(f"Grid search from {dates[0].date()} to {dates[-1].date()} "
          f"({len(dates)/252:.1f}y)")

    reg_spy = spy_regime(dates)
    results = []

    for lookback in [10, 20, 30]:
        for top_n in [1, 2]:
            for rebal in [5]:
                momo = top_n_momo(MOMO_UNIVERSE, lookback, top_n, dates,
                                  rebal_days=rebal, regime=reg_spy)
                for btc_ma in [50, 100]:
                    btc_r = btc_gated(dates, ma=btc_ma)
                    for w_momo in [0.5, 0.6, 0.7, 0.8]:
                        w_btc = 1.0 - w_momo
                        port = w_momo * momo + w_btc * btc_r
                        ar, av, sr, mdd, nav = stats(port)
                        results.append({
                            "lookback": lookback, "top_n": top_n,
                            "btc_ma": btc_ma, "w_momo": w_momo, "w_btc": w_btc,
                            "Ret": ar, "Vol": av, "SR": sr, "MDD": mdd, "NAV": nav,
                        })

    df = pd.DataFrame(results)
    # Filter by target
    passed = df[df["Ret"] >= 50].sort_values("SR", ascending=False)
    print(f"\n{len(passed)} configs clear 50% CAGR:")
    if len(passed):
        print(passed.head(15).to_string(index=False))
    print("\nTop 10 by Sharpe overall:")
    print(df.sort_values("SR", ascending=False).head(10).to_string(index=False))
    print("\nTop 10 by return overall:")
    print(df.sort_values("Ret", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
