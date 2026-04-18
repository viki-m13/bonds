"""Build AURORA returns using proxies for pre-inception history.

Proxy map (used only before the real ETF's inception date):
  CoveredCall sleeve:
    JEPI (2020-05) -> XYLD (2013-06)   # S&P covered-call
    JEPQ (2022-05) -> QYLD (2013-12)   # Nasdaq covered-call
    SPYI (2022-08) -> XYLD (2013-06)
    DIVO (2016-12) -> SCHD (2011-10)   # dividend proxy (no calls before DIVO)

  Momentum universe (3x leveraged):
    All pre-2011 except SOXL (2010-03) and TQQQ (2010-02) - no proxies needed
    for the >=2014 window.

  Managed-Futures sleeve (trend replication):
    DBMF (2019-05), CTA (2022-03), KMLM (2020-12)
    Pre-inception: synthetic 4-asset-class 12m-momentum trend using
      DBC (commodities), TLT (20y rates), UUP (dollar), SPY (equities)
      Long if 12m return > 0, else BIL. Equal-weighted, monthly rebalance.

Output: data/results/aurora_proxy_returns.csv  (Date, Close, source)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

W_COVCALL = 0.40
W_MOMO = 0.40
W_MF = 0.20

COVCALL_TICKERS = ["JEPI", "JEPQ", "SPYI", "DIVO"]
COVCALL_PROXY = {"JEPI": "XYLD", "JEPQ": "QYLD", "SPYI": "XYLD", "DIVO": "SCHD"}

MF_TICKERS = ["DBMF", "CTA", "KMLM"]

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


def returns_or_proxy(ticker, proxy_ticker, dates):
    """Daily returns using live ETF after inception, proxy before.
    Returns (ret_series, used_proxy_mask)."""
    live = load_etf(ticker)
    proxy = load_etf(proxy_ticker) if proxy_ticker else None
    live_ret = (live.reindex(dates).ffill().pct_change().fillna(0)
                if live is not None else pd.Series(0.0, index=dates))
    proxy_ret = (proxy.reindex(dates).ffill().pct_change().fillna(0)
                 if proxy is not None else pd.Series(0.0, index=dates))
    live_avail = pd.Series(
        False if live is None else (live.index.min() <= dates), index=dates)
    out = live_ret.where(live_avail, proxy_ret)
    return out, ~live_avail


def covered_call_sleeve(dates):
    """Equal-weight JEPI/JEPQ/SPYI/DIVO with proxies pre-inception."""
    pieces = {}
    any_proxy = pd.Series(False, index=dates)
    for t in COVCALL_TICKERS:
        r, up = returns_or_proxy(t, COVCALL_PROXY[t], dates)
        pieces[t] = r
        any_proxy = any_proxy | up
    df = pd.DataFrame(pieces)
    sleeve = df.mean(axis=1)
    return sleeve, any_proxy


def synth_trend(dates):
    """Simple 4-asset 12-month trend. Long asset if 12m return > 0, else BIL.
    Equal-weight, daily-checked monthly signal."""
    assets = ["DBC", "TLT", "UUP", "SPY"]
    prices = {}
    for a in assets:
        p = load_etf(a)
        if p is None: continue
        prices[a] = p.reindex(dates).ffill()
    prices = pd.DataFrame(prices)
    rets = prices.pct_change().fillna(0)
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)

    # Signal: lookback 252 trading days, evaluated each day (shift 1 for no look-ahead)
    lookback = 252
    signals = {}
    for a in prices.columns:
        sig = (prices[a] / prices[a].shift(lookback) - 1 > 0).shift(1).fillna(False)
        signals[a] = sig
    sig_df = pd.DataFrame(signals)

    # Each asset contributes 1/N; when signal off, that slice goes to BIL.
    sleeve = pd.Series(0.0, index=dates)
    n = len(assets)
    for a in assets:
        if a not in rets.columns: continue
        sleeve = sleeve + (1.0 / n) * (sig_df[a].astype(float) * rets[a] +
                                       (1 - sig_df[a].astype(float)) * bil)
    return sleeve


def managed_futures_sleeve(dates):
    """Equal-weight DBMF/CTA/KMLM; use synth_trend proxy before first
    constituent is live. Blend smoothly as each comes online."""
    live = {}
    avail_masks = {}
    for t in MF_TICKERS:
        p = load_etf(t)
        if p is None: continue
        live[t] = p.reindex(dates).ffill().pct_change().fillna(0)
        avail_masks[t] = pd.Series(p.index.min() <= dates, index=dates)
    live_df = pd.DataFrame(live)
    avail_df = pd.DataFrame(avail_masks).astype(int)

    # Average of available live funds (1/N where N is live count)
    n_live = avail_df.sum(axis=1)
    live_avg = (live_df * avail_df).sum(axis=1) / n_live.replace(0, np.nan)

    # Proxy series for when n_live == 0
    proxy = synth_trend(dates)

    sleeve = live_avg.where(n_live > 0, proxy)
    used_proxy = (n_live == 0)
    return sleeve.fillna(0), used_proxy


def weekly_momo(universe, lookback, top_n, dates, rebal_days=5,
                tc_bps=15.0, regime=None, cash_ticker="BIL"):
    prices = pd.DataFrame({t: load_etf(t) for t in universe})
    prices = prices.reindex(dates).ffill()
    rets = prices.pct_change().fillna(0)
    cash = load_etf(cash_ticker).reindex(dates).ffill().pct_change().fillna(0)
    current = pd.Series(0.0, index=prices.columns)
    port = pd.Series(0.0, index=dates)
    last_idx = -rebal_days
    # Require all universe ETFs to have data before allowing rebalance
    avail = prices.notna().all(axis=1)
    for i in range(len(dates)):
        if avail.iloc[i] and i >= lookback and i - last_idx >= rebal_days:
            momo = prices.iloc[i] / prices.iloc[i - lookback] - 1
            momo = momo.dropna()
            ranked = momo.sort_values(ascending=False)
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
        r = g * r + (1 - g) * cash.iloc[i]
        port.iloc[i] += r
    return port


def spy_regime(dates, vix_cap=30, ma_len=200):
    spy = load_etf("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS")
    ok = (spy > spy.rolling(ma_len).mean())
    if vix is not None:
        v = vix.reindex(dates).ffill()
        ok = ok & (v < vix_cap)
    return ok.shift(1).fillna(False).astype(float)


def main():
    print("Building AURORA with proxies...")
    # Start when covered-call proxy (XYLD/QYLD) + momentum universe all live.
    # QYLD 2013-12, SOXL 2010-03, TQQQ 2010-02. Binding: QYLD + 1 year of trend
    # signal for synth MF = need 252 days after 2013-12 = ~2015-01.
    spy = load_etf("SPY")
    start = pd.Timestamp("2014-01-01")
    dates = spy.loc[start:].index

    sleeve_c, cc_proxy = covered_call_sleeve(dates)
    sleeve_f, mf_proxy = managed_futures_sleeve(dates)
    regime = spy_regime(dates)
    sleeve_m = weekly_momo(MOMO_UNIVERSE, 20, 3, dates, rebal_days=5, regime=regime)

    port = W_COVCALL * sleeve_c + W_MOMO * sleeve_m + W_MF * sleeve_f

    # Mark any day using any proxy input.
    any_proxy = cc_proxy | mf_proxy

    # Live-only period: when every constituent is live. Live-onset is the
    # date when the LAST covered-call ETF (SPYI 2022-08) and LAST managed-
    # futures ETF (CTA 2022-03) are both live.
    spyi_live = pd.Timestamp("2022-08-30")
    cta_live = pd.Timestamp("2022-03-08")
    live_onset = max(spyi_live, cta_live)
    live_only = dates >= live_onset
    any_proxy = any_proxy & ~pd.Series(live_only, index=dates)

    out = pd.DataFrame({
        "Close": port,
        "CoveredCall": sleeve_c,
        "Momo": sleeve_m,
        "MF": sleeve_f,
        "source": np.where(any_proxy, "proxy", "live"),
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / "aurora_proxy_returns.csv")

    n_total = len(out)
    n_proxy = (out["source"] == "proxy").sum()
    n_live = (out["source"] == "live").sum()
    live_start = out[out["source"] == "live"].index.min() if n_live else None
    ar = port.mean() * 252
    av = port.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + port).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"\nAURORA (with proxies) from {dates[0].date()} to {dates[-1].date()}")
    print(f"  n_total={n_total}, n_proxy={n_proxy} ({n_proxy/n_total*100:.0f}%), "
          f"n_live={n_live} ({n_live/n_total*100:.0f}%)")
    if live_start is not None:
        print(f"  live segment starts: {live_start.date()}")
    print(f"  Sharpe={sr:.2f}  Ret={ar*100:.2f}%  Vol={av*100:.2f}%  MDD={mdd*100:.1f}%")
    print(f"  NAV multiple: {cum.iloc[-1]:.3f}x")


if __name__ == "__main__":
    main()
