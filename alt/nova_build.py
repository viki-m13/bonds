"""NOVA — max-growth strategy. Unified cross-sectional momentum on a broad
bull-leveraged-ETF universe + BTC + ETH, weekly rebalanced, with per-name
cap and asset-class regime gates.

Config (picked from alt/nova_grid2.py, validated on nova_grid3.py):
  - Universe: 18 bull-leveraged ETFs + BTC_USD + ETH_USD (20 total)
    - Broad index:      TQQQ UPRO QLD SSO
    - Sector:           SOXL TECL FAS LABU ERX DRN NUGT
    - Country/region:   EDC YINN
    - Commodity:        UGL UCO
    - Rates:            TMF TYD UBT
    - Crypto:           BTC_USD ETH_USD
  - Signal:  10-day momentum, top-3 positive, equal-weight
  - Cap:     33% per name (prevents any single asset from dominating)
  - Gates:   SPY>200dma AND VIX<30 on the equity leg
             BTC>200dma on the crypto leg
  - Rebal:   weekly (5 trading days)
  - TC:      15bps round-trip

Full-window result (2014-09 → 2026-04, 11.5y):
  Return  ~67%    Vol  ~42%    Sharpe  ~1.59    MDD  ~-35%    NAV  832x

Output: data/results/nova_returns.csv
  columns: Close, Crypto, Equity, Cash, SPY, AGG
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

EQUITY = [
    # Original 7 (v1 universe)
    "TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL",
    # Expanded bull-leveraged set (v2 universe)
    "LABU", "EDC", "YINN", "ERX", "NUGT", "DRN", "UCO", "TYD",
    "QLD", "SSO", "UBT",
]
CRYPTO = ["BTC_USD", "ETH_USD"]

LOOKBACK = 10
TOP_N = 3
CAP = 0.33
BTC_MA = 200
SPY_MA = 200
VIX_CAP = 30.0
REBAL = 5
TC_BPS = 15.0


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


def spy_regime(dates):
    spy = load_etf("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    ok = (spy > spy.rolling(SPY_MA).mean()) & (vix < VIX_CAP)
    return ok.shift(1).fillna(False).astype(float)


def btc_regime(dates):
    btc = load_etf("BTC_USD").reindex(dates).ffill()
    return (btc > btc.rolling(BTC_MA).mean()).shift(1).fillna(False).astype(float)


def build():
    btc = load_etf("BTC_USD")
    spy = load_etf("SPY")
    dates = spy.loc[btc.index.min():].index
    print(f"NOVA build: {dates[0].date()} .. {dates[-1].date()} "
          f"({len(dates)/252:.1f}y)")

    universe = EQUITY + CRYPTO
    prices = pd.DataFrame({t: load_etf(t) for t in universe}).reindex(dates).ffill()
    rets = prices.pct_change().fillna(0)
    avail = prices.notna()
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    reg_eq = spy_regime(dates)
    reg_bt = btc_regime(dates)

    current = pd.Series(0.0, index=universe)
    port = pd.Series(0.0, index=dates)
    w_crypto = pd.Series(0.0, index=dates)
    w_equity = pd.Series(0.0, index=dates)
    last_idx = -REBAL
    rebal_rows = []

    for i in range(len(dates)):
        if i >= LOOKBACK and i - last_idx >= REBAL:
            live = avail.iloc[i]
            momo = (prices.iloc[i] / prices.iloc[i - LOOKBACK] - 1).where(live)
            ranked = momo.dropna().sort_values(ascending=False)
            positive = [t for t in ranked.index if momo[t] > 0]
            top = positive[:TOP_N]
            new = pd.Series(0.0, index=universe)
            if top:
                w = 1.0 / len(top)
                for t in top:
                    new[t] = min(w, CAP)
            tc = (new - current).abs().sum() * (TC_BPS / 1e4)
            port.iloc[i] -= tc
            current = new
            last_idx = i
            rebal_rows.append({
                "date": dates[i],
                "pick_1": top[0] if len(top) > 0 else "",
                "pick_2": top[1] if len(top) > 1 else "",
                "pick_3": top[2] if len(top) > 2 else "",
                "n_positive": len(positive),
            })

        eff = current.copy()
        geq = reg_eq.iloc[i]
        gbt = reg_bt.iloc[i]
        off_eq = sum(current[t] for t in EQUITY) * (1 - geq)
        off_bt = sum(current[t] for t in CRYPTO) * (1 - gbt)
        for t in EQUITY: eff[t] = current[t] * geq
        for t in CRYPTO: eff[t] = current[t] * gbt
        r = (rets.iloc[i] * eff).sum() + (off_eq + off_bt) * bil.iloc[i]
        port.iloc[i] += r
        w_crypto.iloc[i] = sum(eff[t] for t in CRYPTO)
        w_equity.iloc[i] = sum(eff[t] for t in EQUITY)

    r_spy = spy.reindex(dates).ffill().pct_change().fillna(0)
    agg = load_etf("AGG")
    r_agg = agg.reindex(dates).ffill().pct_change().fillna(0) if agg is not None else pd.Series(0.0, index=dates)

    out = pd.DataFrame({
        "Close": port,
        "Crypto": w_crypto,
        "Equity": w_equity,
        "Cash": 1 - w_crypto - w_equity,
        "SPY": r_spy,
        "AGG": r_agg,
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / "nova_returns.csv")

    pd.DataFrame(rebal_rows).to_csv(RESULTS / "nova_rebalances.csv", index=False)

    ar = port.mean() * 252
    av = port.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + port).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"  Full-window  SR={sr:.2f}  Ret={ar*100:.2f}%  Vol={av*100:.2f}%  "
          f"MDD={mdd*100:.1f}%  NAVx={cum.iloc[-1]:.1f}")
    print(f"  Avg exposure  crypto={w_crypto.mean()*100:.1f}%  "
          f"equity={w_equity.mean()*100:.1f}%  "
          f"cash={(1-w_crypto-w_equity).mean()*100:.1f}%")

    # SPY & AGG benchmarks on same window
    for name, r in [("SPY", r_spy), ("AGG", r_agg)]:
        a = r.mean() * 252; v = r.std() * np.sqrt(252)
        s = a / v if v > 0 else 0
        c = (1 + r).cumprod()
        m = (c / c.cummax() - 1).min()
        print(f"  {name:6s} SR={s:.2f}  Ret={a*100:.2f}%  Vol={v*100:.2f}%  MDD={m*100:.1f}%")


if __name__ == "__main__":
    build()
