"""Build AURORA returns with proxies, extended to 2008-05 (match Growth page).

Proxy tiers (most recent wins):
  TIER 1 - Live (2022-08+): every real ETF available (JEPI/JEPQ/SPYI/DIVO/
           DBMF/CTA/KMLM/TQQQ/UPRO/SOXL/TECL/FAS/TMF/UGL)
  TIER 2 - Mid-proxy (2013-06 to 2022-08):
           JEPI/SPYI -> XYLD, JEPQ -> QYLD, DIVO -> SCHD
           DBMF/CTA/KMLM -> synthetic 12m trend on DBC/TLT/UUP/SPY
           3x momentum universe uses real ETFs
  TIER 3 - Early-proxy (2008-05 to 2013-06):
           Covered-call sleeve: SPY minus 3% annualised haircut
           (buywrite drag proxy — XYLD/QYLD not yet live)
           Managed-futures: synthetic trend on whatever of DBC/TLT/UUP/SPY
           is live (UUP from 2007, DBC from 2006 — OK)
           3x momentum universe: daily-return * 3 on underlying index ETFs:
             TQQQ->QQQ*3, UPRO->SPY*3, SOXL->XLK*3, TECL->XLK*3,
             FAS->XLF*3, TMF->TLT*3, UGL->GLD*2
           (approximation; real 3x ETFs would differ due to daily reset drag
            and higher expense ratios — TIER 3 is a conservative estimate)

Pre-2008-12 there are no 3x proxies with enough history anyway — synthetic
3x is used throughout TIER 3. SPY starts 2005-01 here, so earliest possible
is 2005-01 + 252-day trend lookback = 2006-01. We still start at 2008-05-13
to match the Growth page.

Output: data/results/aurora_proxy_returns.csv
  columns: Close, CoveredCall, Momo, MF, SPY, AGG, source, tier
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
MID_PROXY_CC = {"JEPI": "XYLD", "JEPQ": "QYLD", "SPYI": "XYLD", "DIVO": "SCHD"}

MF_TICKERS = ["DBMF", "CTA", "KMLM"]

MOMO_UNIVERSE = ["TQQQ", "UPRO", "SOXL", "TECL", "FAS", "TMF", "UGL"]
MOMO_EARLY_UNDERLIER = {
    "TQQQ": ("QQQ", 3), "UPRO": ("SPY", 3), "SOXL": ("XLK", 3),
    "TECL": ("XLK", 3), "FAS": ("XLF", 3), "TMF": ("TLT", 3), "UGL": ("GLD", 2),
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


def pct(s, dates):
    return s.reindex(dates).ffill().pct_change().fillna(0)


def avail(s, dates):
    return pd.Series(False if s is None else (s.index.min() <= dates), index=dates)


def synthetic_3x(underlier_ticker, leverage, dates):
    """Daily-return * leverage, minus 1%/yr expense ratio (approx real 3x ETF fee)."""
    s = load_etf(underlier_ticker)
    if s is None: return pd.Series(0.0, index=dates)
    r = pct(s, dates)
    return leverage * r - (0.01 / 252)


def covered_call_sleeve(dates):
    """JEPI/JEPQ/SPYI/DIVO equal-weight. TIER 1 -> real; TIER 2 -> proxy ETFs;
    TIER 3 -> SPY with 3%/yr haircut (pre-XYLD/QYLD buywrite proxy)."""
    r_spy_haircut = pct(load_etf("SPY"), dates) - (0.03 / 252)
    pieces = []
    for t in COVCALL_TICKERS:
        real = load_etf(t)
        proxy_ticker = MID_PROXY_CC[t]
        proxy = load_etf(proxy_ticker)
        real_m = avail(real, dates)
        proxy_m = avail(proxy, dates) if proxy is not None else pd.Series(False, index=dates)
        real_r = pct(real, dates) if real is not None else pd.Series(0.0, index=dates)
        proxy_r = pct(proxy, dates) if proxy is not None else pd.Series(0.0, index=dates)
        # TIER 3 fallback: SPY minus buywrite drag
        ret = proxy_r.where(proxy_m, r_spy_haircut)
        ret = real_r.where(real_m, ret)
        pieces.append(ret)
    return pd.concat(pieces, axis=1).mean(axis=1)


def synth_trend(dates):
    """Equal-weight 12m-momentum trend across DBC/TLT/UUP/SPY, fall back to BIL."""
    assets = ["DBC", "TLT", "UUP", "SPY"]
    bil = load_etf("BIL")
    r_bil = pct(bil, dates) if bil is not None else pd.Series(0.0, index=dates)
    sleeve = pd.Series(0.0, index=dates)
    live_n = pd.Series(0, index=dates)
    for a in assets:
        px = load_etf(a)
        if px is None: continue
        m = avail(px, dates)
        live_n = live_n + m.astype(int)
    for a in assets:
        px = load_etf(a)
        if px is None: continue
        pxd = px.reindex(dates).ffill()
        r = pxd.pct_change().fillna(0)
        sig = (pxd / pxd.shift(252) - 1 > 0).shift(1).fillna(False).astype(float)
        m = avail(px, dates).astype(float)
        contrib = m * (sig * r + (1 - sig) * r_bil)
        sleeve = sleeve + contrib
    return sleeve / live_n.replace(0, np.nan).fillna(1)


def managed_futures_sleeve(dates):
    live = {}
    masks = {}
    for t in MF_TICKERS:
        p = load_etf(t)
        if p is None: continue
        live[t] = pct(p, dates)
        masks[t] = avail(p, dates)
    live_df = pd.DataFrame(live)
    mask_df = pd.DataFrame(masks).astype(int)
    n_live = mask_df.sum(axis=1)
    live_avg = (live_df * mask_df).sum(axis=1) / n_live.replace(0, np.nan)
    proxy = synth_trend(dates)
    return live_avg.where(n_live > 0, proxy).fillna(0)


def weekly_momo(dates, lookback=20, top_n=3, rebal_days=5, tc_bps=15.0, regime=None):
    """Top-3 momentum on MOMO_UNIVERSE, with synthetic 3x daily-return fallback
    when the real 3x ETF is not yet live. Price series are reconstructed by
    cumulative product of daily returns (synthetic when needed)."""
    # Build a return series for each momentum ticker (real if live else synthetic)
    rets = {}
    for t in MOMO_UNIVERSE:
        real = load_etf(t)
        real_m = avail(real, dates)
        real_r = pct(real, dates) if real is not None else pd.Series(0.0, index=dates)
        und, lev = MOMO_EARLY_UNDERLIER[t]
        synth_r = synthetic_3x(und, lev, dates)
        rets[t] = real_r.where(real_m, synth_r)
    rets_df = pd.DataFrame(rets)
    # Reconstruct prices for momentum ranking
    prices = (1 + rets_df).cumprod() * 100

    bil = load_etf("BIL")
    cash = pct(bil, dates) if bil is not None else pd.Series(0.0, index=dates)
    current = pd.Series(0.0, index=rets_df.columns)
    port = pd.Series(0.0, index=dates)
    last_idx = -rebal_days
    for i in range(len(dates)):
        if i >= lookback and i - last_idx >= rebal_days:
            momo = prices.iloc[i] / prices.iloc[i - lookback] - 1
            ranked = momo.dropna().sort_values(ascending=False)
            positive = [t for t in ranked.index if momo[t] > 0]
            top = positive[:top_n]
            new_target = pd.Series(0.0, index=rets_df.columns)
            if top:
                w = 1.0 / len(top)
                for t in top: new_target[t] = w
            tc = (new_target - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = new_target
            last_idx = i
        r = (rets_df.iloc[i] * current).sum()
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
    print("Building AURORA extended to Growth-page start...")
    spy = load_etf("SPY")
    start = max(pd.Timestamp("2008-05-13"), spy.index.min())
    # Need 252 days of SPY history for the trend signal
    dates = spy.loc[start:].index

    sleeve_c = covered_call_sleeve(dates)
    sleeve_f = managed_futures_sleeve(dates)
    regime = spy_regime(dates)
    sleeve_m = weekly_momo(dates, regime=regime)

    port = W_COVCALL * sleeve_c + W_MOMO * sleeve_m + W_MF * sleeve_f

    # Tier tracking
    # TIER 1: all covered-call real AND all MF real AND all 3x real
    cc_real = pd.DataFrame({t: avail(load_etf(t), dates) for t in COVCALL_TICKERS}).all(axis=1)
    mf_real = pd.DataFrame({t: avail(load_etf(t), dates) for t in MF_TICKERS}).all(axis=1)
    momo_real = pd.DataFrame({t: avail(load_etf(t), dates) for t in MOMO_UNIVERSE}).all(axis=1)
    all_live = cc_real & mf_real & momo_real

    # TIER 2: XYLD/QYLD/SCHD live AND momentum universe live (but not all cc/mf real)
    tier2_cc = pd.DataFrame({p: avail(load_etf(p), dates) for p in set(MID_PROXY_CC.values())}).all(axis=1)
    tier2 = tier2_cc & momo_real & ~all_live

    tier = pd.Series(3, index=dates, dtype=int)
    tier[tier2] = 2
    tier[all_live] = 1
    source = np.where(all_live, "live", "proxy")

    # Benchmarks
    r_spy = pct(spy, dates)
    agg = load_etf("AGG")
    r_agg = pct(agg, dates) if agg is not None else pd.Series(0.0, index=dates)

    out = pd.DataFrame({
        "Close": port, "CoveredCall": sleeve_c, "Momo": sleeve_m, "MF": sleeve_f,
        "SPY": r_spy, "AGG": r_agg, "source": source, "tier": tier.values,
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / "aurora_proxy_returns.csv")

    for label, mask in [("full", slice(None)),
                        ("tier1 live", out["tier"] == 1),
                        ("tier2 mid-proxy", out["tier"] == 2),
                        ("tier3 early-proxy", out["tier"] == 3)]:
        sub = out.index[mask] if not isinstance(mask, slice) else out.index
        r = port.loc[sub]
        if len(r) < 2: continue
        ar = r.mean() * 252
        av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        d1, d2 = r.index[0].date(), r.index[-1].date()
        print(f"  {label:20s} {d1}..{d2} ({len(r)/252:.1f}y): "
              f"SR={sr:.2f} Ret={ar*100:.2f}% Vol={av*100:.2f}% MDD={mdd*100:.1f}%")


if __name__ == "__main__":
    main()
