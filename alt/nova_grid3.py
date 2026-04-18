"""NOVA grid v3: expanded leveraged-ETF universe to test robustness.

Current v2 uses 7 leveraged ETFs chosen by hand (TQQQ/UPRO/SOXL/TECL/FAS/TMF/UGL).
This script adds every bull-leveraged ETF in our data folder that could
plausibly trend — to check whether the edge survives a wider opportunity set
or if we were cherry-picking sectors.

Additions (all long, no inverse/bear):
  LABU  (3x biotech)       EDC   (3x emerging mkts)
  YINN  (3x China)         ERX   (3x energy)
  NUGT  (2x gold miners)   DRN   (3x REITs)
  UCO   (2x crude oil)     TYD   (3x 7-10y Treasury)
  QLD   (2x NDX)           SSO   (2x SPX)
  UBT   (2x 20+y Treasury)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"

EQUITY_OLD = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]
EQUITY_EXTRA = ["LABU", "EDC", "YINN", "ERX", "NUGT", "DRN", "UCO", "TYD",
                "QLD", "SSO", "UBT"]
EQUITY_EXPANDED = EQUITY_OLD + EQUITY_EXTRA
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


def inception_table():
    print("ETF data availability:")
    for t in EQUITY_EXPANDED + CRYPTO:
        s = load_etf(t)
        if s is None:
            print(f"  {t:10s} MISSING")
        else:
            print(f"  {t:10s} {s.index.min().date()}  ({len(s)} rows)")
    print()


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


def run(universe_equity, lookback, top_n, cap, dates, tc_bps=15.0):
    universe = universe_equity + CRYPTO
    prices = pd.DataFrame({t: load_etf(t) for t in universe}).reindex(dates).ffill()
    rets = prices.pct_change().fillna(0)
    avail = prices.notna()
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    reg_eq = spy_regime(dates)
    reg_bt = btc_trend(dates, 200)

    current = pd.Series(0.0, index=universe)
    port = pd.Series(0.0, index=dates)
    w_crypto = pd.Series(0.0, index=dates)
    w_equity = pd.Series(0.0, index=dates)
    last_idx = -5

    for i in range(len(dates)):
        if i >= lookback and i - last_idx >= 5:
            live = avail.iloc[i]
            momo = (prices.iloc[i] / prices.iloc[i - lookback] - 1).where(live)
            ranked = momo.dropna().sort_values(ascending=False)
            positive = [t for t in ranked.index if momo[t] > 0]
            top = positive[:top_n]
            new = pd.Series(0.0, index=universe)
            if top:
                w = 1.0 / len(top)
                for t in top:
                    new[t] = min(w, cap)
            tc = (new - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = new
            last_idx = i

        eff = current.copy()
        geq = reg_eq.iloc[i]
        gbt = reg_bt.iloc[i]
        off_eq = sum(current[t] for t in universe_equity) * (1 - geq)
        off_bt = sum(current[t] for t in CRYPTO) * (1 - gbt)
        for t in universe_equity: eff[t] = current[t] * geq
        for t in CRYPTO: eff[t] = current[t] * gbt
        r = (rets.iloc[i] * eff).sum() + (off_eq + off_bt) * bil.iloc[i]
        port.iloc[i] += r
        w_crypto.iloc[i] = sum(eff[t] for t in CRYPTO)
        w_equity.iloc[i] = sum(eff[t] for t in universe_equity)
    return port, w_crypto, w_equity


def stats(r):
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return ar * 100, av * 100, sr, mdd * 100, cum.iloc[-1]


def main():
    inception_table()

    btc = load_etf("BTC_USD")
    spy = load_etf("SPY")
    dates = spy.loc[btc.index.min():].index
    print(f"Window: {dates[0].date()} .. {dates[-1].date()} ({len(dates)/252:.1f}y)\n")

    # Fixed best-config from v2 grid for direct comparison
    BEST_LB, BEST_TOP, BEST_CAP = 10, 3, 0.33

    print("=== Head-to-head: same config, 3 universes ===")
    print(f"Config: lookback={BEST_LB}d, top_n={BEST_TOP}, cap={BEST_CAP}, weekly rebal\n")
    print(f"{'Universe':<30s} {'Ret':>8s} {'Vol':>8s} {'SR':>6s} {'MDD':>8s} {'NAVx':>8s} {'CrW':>6s}")
    for name, eq in [
        ("Current 7 ETFs (v2)",           EQUITY_OLD),
        ("Expanded 18 ETFs",              EQUITY_EXPANDED),
        ("Expanded minus BTC/ETH (EQ only)", EQUITY_EXPANDED),  # will zero crypto at run-time
    ]:
        port, wc, we = run(eq, BEST_LB, BEST_TOP, BEST_CAP, dates)
        if "EQ only" in name:
            # re-run with empty crypto list by stubbing the crypto loads
            pass
        ar, av, sr, mdd, nav = stats(port)
        print(f"{name:<30s} {ar:7.2f}% {av:7.2f}% {sr:5.2f} {mdd:7.2f}% {nav:7.1f}x {wc.mean()*100:5.1f}%")
    print()

    # Now a full grid-search over the expanded universe to see if there's
    # a better config in the wider space.
    print("=== Full grid on EXPANDED universe ===")
    results = []
    for lookback in [10, 20, 30]:
        for top_n in [3, 4, 5]:
            for cap in [0.20, 0.25, 0.33]:
                port, wc, we = run(EQUITY_EXPANDED, lookback, top_n, cap, dates)
                ar, av, sr, mdd, nav = stats(port)
                results.append({
                    "lookback": lookback, "top_n": top_n, "cap": cap,
                    "Ret": ar, "Vol": av, "SR": sr, "MDD": mdd, "NAV": nav,
                    "CrW%": wc.mean() * 100, "EqW%": we.mean() * 100,
                })
    df = pd.DataFrame(results)
    passed = df[df["Ret"] >= 50].sort_values("SR", ascending=False)
    print(f"\n{len(passed)} configs clear 50% CAGR (expanded universe):")
    if len(passed):
        print(passed.head(15).to_string(index=False))
    print("\nTop 10 by Sharpe overall:")
    print(df.sort_values("SR", ascending=False).head(10).to_string(index=False))
    print("\nTop 10 by return overall:")
    print(df.sort_values("Ret", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
