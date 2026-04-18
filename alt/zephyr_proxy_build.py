"""Build ZEPHYR returns using proxies for pre-inception history.

Proxy map (used only when the real ETF has no data on a given date):
  JAAA (2020-10)  -> 60% FLOT + 40% BKLN
  JPST (2017-05)  -> MINT
  MINT (2009-11)  -> MINT (real)
  BKLN (2011-03)  -> BKLN (real)
  SRLN (2013-04)  -> SRLN (real) - this is the binding constraint
  FLOT (2011-06)  -> FLOT (real)
  GLD  (2004-11)  -> GLD  (real)

ZEPHYR weights (unchanged): JAAA 32 / JPST 28 / MINT 15 / BKLN 10
                             SRLN 5 / FLOT 5 / GLD 5

Output:
  data/results/zephyr_proxy_returns.csv  (Date, Close, source) where
    source = 'live'  when every constituent has its real ETF available
    source = 'proxy' when at least one proxy is in use
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

WEIGHTS = {
    "JAAA": 0.32, "JPST": 0.28, "MINT": 0.15, "BKLN": 0.10,
    "SRLN": 0.05, "FLOT": 0.05, "GLD": 0.05,
}

# Proxy -> replacement series (as daily returns). If first choice is missing,
# fall back to next. MINT as last resort for the credit sleeves.
PROXY_RECIPE = {
    "JAAA": [("FLOT", 0.6), ("BKLN", 0.4)],
    "JPST": [("MINT", 1.0)],
    "MINT": [("MINT", 1.0)],
    "BKLN": [("BKLN", 1.0)],
    "SRLN": [("SRLN", 1.0)],
    "FLOT": [("FLOT", 1.0)],
    "GLD":  [("GLD",  1.0)],
}


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


def daily_returns_for(ticker, dates):
    """Return a daily-return series for `ticker` over `dates`, where missing
    live data is filled from PROXY_RECIPE. Also returns a boolean series
    flagging dates where proxy fallback was used."""
    live = load_etf(ticker)
    live_ret = (live.reindex(dates).ffill().pct_change().fillna(0)
                if live is not None else pd.Series(0.0, index=dates))
    live_avail = pd.Series(False, index=dates) if live is None else \
                 pd.Series(live.index.min() <= dates, index=dates)

    # Build proxy blend on the same date index.
    proxy_ret = pd.Series(0.0, index=dates)
    for comp, w in PROXY_RECIPE[ticker]:
        comp_px = load_etf(comp)
        if comp_px is None: continue
        proxy_ret = proxy_ret + w * comp_px.reindex(dates).ffill().pct_change().fillna(0)

    # Use live where available, proxy otherwise.
    used_proxy = ~live_avail
    out = live_ret.where(live_avail, proxy_ret)
    return out, used_proxy


def regime_gate(dates):
    """ZEPHYR regime gate: HY OAS < 8% AND dDGS10(63d) < 0.7%, both T-1."""
    hy = load_fred("BAMLH0A0HYM2")
    dgs = load_fred("DGS10")
    hy_d = hy.reindex(dates).ffill() if hy is not None else pd.Series(5.0, index=dates)
    dg_d = dgs.reindex(dates).ffill() if dgs is not None else pd.Series(3.0, index=dates)
    g_credit = np.clip((8.0 - hy_d) / (8.0 - 5.0), 0, 1)
    g_rate = ((dg_d - dg_d.shift(63)) < 0.7).astype(float)
    g = (g_credit * g_rate).shift(1).fillna(0)
    return g


def main():
    print("Building ZEPHYR with proxies...")
    # Date grid: start from first date where SRLN exists (binding constraint)
    srln = load_etf("SRLN")
    start = srln.index.min()
    spy = load_etf("SPY")
    dates = spy.loc[start:].index

    combined = pd.Series(0.0, index=dates)
    any_proxy = pd.Series(False, index=dates)

    for tkr, w in WEIGHTS.items():
        ret, used_proxy = daily_returns_for(tkr, dates)
        combined = combined + w * ret
        any_proxy = any_proxy | used_proxy
        print(f"  {tkr} @ {w:.0%}: proxy days = {used_proxy.sum()}")

    # Apply regime gate + BIL complement.
    g = regime_gate(dates)
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    gated = g * combined + (1 - g) * bil

    # 21-day rebalance with 5 bps cost on turnover (static weights means ~drift only)
    # For proxy purposes we just apply a small fixed 2 bps / month drag.
    monthly_drag = 0.0002 / 21  # ~0.24%/yr haircut (vs 5bps per rebalance)
    gated = gated - monthly_drag

    out = pd.DataFrame({
        "Close": gated,
        "source": np.where(any_proxy, "proxy", "live"),
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / "zephyr_proxy_returns.csv")

    # Summary
    n_total = len(out)
    n_live = (out["source"] == "live").sum()
    n_proxy = (out["source"] == "proxy").sum()
    live_start = out[out["source"] == "live"].index.min() if n_live else None
    ar = gated.mean() * 252
    av = gated.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + gated).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"\nZEPHYR (with proxies) from {dates[0].date()} to {dates[-1].date()}")
    print(f"  n_total={n_total}, n_proxy={n_proxy} ({n_proxy/n_total*100:.0f}%), "
          f"n_live={n_live} ({n_live/n_total*100:.0f}%)")
    if live_start is not None:
        print(f"  live segment starts: {live_start.date()}")
    print(f"  Sharpe={sr:.2f}  Ret={ar*100:.2f}%  Vol={av*100:.2f}%  MDD={mdd*100:.1f}%")
    print(f"  NAV multiple: {cum.iloc[-1]:.3f}x")


if __name__ == "__main__":
    main()
