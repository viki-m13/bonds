"""Build NOVA returns with proxies, extended to 2005-01.

Tiering (newest wins on each date):
  TIER 1 - Live (2017-11+): every real 3x/2x ETF + BTC + ETH available
  TIER 2 - Crypto-partial (2014-09 → 2017-11): real leveraged ETFs + BTC,
           ETH not yet live so the universe has 1 fewer crypto slot
  TIER 3 - No-crypto (2010-12 → 2014-09): real leveraged ETFs (most live
           by 2010-12 when NUGT launched), no crypto yet
  TIER 4 - Synthetic-leverage (2005-01 → 2010-12): real ETFs where live,
           synthetic daily×leverage on underlier otherwise:
             TQQQ→QQQ×3, UPRO→SPY×3, QLD→QQQ×2, SSO→SPY×2,
             SOXL→SMH×3, TECL→XLK×3, FAS→XLF×3, LABU→XBI×3,
             ERX→XLE×2, NUGT→GLD×2, DRN→IYR×3,
             EDC→EEM×3, YINN→FXI×3, UGL→GLD×2, UCO→USO×2,
             TMF→TLT×3, TYD→IEF×3, UBT→TLT×2
           1%/yr expense haircut on each synthetic. Instruments without
           a live underlier on a given date are dropped from the ranking.
           No crypto in this tier.

Same momentum/cap/gate mechanics as production nova_build.py
(lookback=10, top_n=3, cap=0.33, SPY+VIX equity gate, BTC trend
crypto gate when BTC is live).

Output: data/results/nova_proxy_returns.csv
  columns: Close, SPY, AGG, source, tier
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

# Leveraged ETF → (underlier, leverage)
EQUITY_MAP = {
    "TQQQ": ("QQQ", 3), "UPRO": ("SPY", 3),
    "QLD":  ("QQQ", 2), "SSO":  ("SPY", 2),
    "SOXL": ("SMH", 3), "TECL": ("XLK", 3), "FAS":  ("XLF", 3),
    "LABU": ("XBI", 3), "ERX":  ("XLE", 2), "NUGT": ("GLD", 2),
    "DRN":  ("IYR", 3), "EDC":  ("EEM", 3), "YINN": ("FXI", 3),
    "UGL":  ("GLD", 2), "UCO":  ("USO", 2),
    "TMF":  ("TLT", 3), "TYD":  ("IEF", 3), "UBT":  ("TLT", 2),
}
EQUITY = list(EQUITY_MAP.keys())
CRYPTO = ["BTC_USD", "ETH_USD"]
UNIVERSE = EQUITY + CRYPTO

LOOKBACK = 10
TOP_N = 3
CAP = 0.33
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


def pct(s, dates):
    return s.reindex(dates).ffill().pct_change().fillna(0)


def avail(s, dates):
    return pd.Series(False if s is None else (s.index.min() <= dates), index=dates)


def synthetic_lev(underlier, leverage, dates):
    """Daily-return × leverage, minus 1%/yr expense ratio (approx real 3x ETF fee)."""
    s = load_etf(underlier)
    if s is None: return None
    r = pct(s, dates)
    # Only valid once underlier is live
    m = avail(s, dates)
    synth = leverage * r - (0.01 / 252)
    return synth.where(m, 0.0), m


def spy_regime(dates, vix_cap=30, ma_len=200):
    spy = load_etf("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS")
    ok = (spy > spy.rolling(ma_len).mean())
    if vix is not None:
        v = vix.reindex(dates).ffill()
        ok = ok & (v < vix_cap)
    return ok.shift(1).fillna(False).astype(float)


def btc_regime(dates, ma_len=200):
    btc = load_etf("BTC_USD")
    if btc is None: return pd.Series(0.0, index=dates)
    b = btc.reindex(dates).ffill()
    return (b > b.rolling(ma_len).mean()).shift(1).fillna(False).astype(float)


def build():
    spy = load_etf("SPY")
    dates = spy.index  # 2005-01-03 onwards
    print(f"NOVA proxy build: {dates[0].date()} .. {dates[-1].date()} "
          f"({len(dates)/252:.1f}y)")

    # Build return series for each equity name: real when live else synthetic
    rets_eq = {}
    avail_eq = {}  # "does this name have *any* valid return series today?"
    for name, (und, lev) in EQUITY_MAP.items():
        real = load_etf(name)
        real_r = pct(real, dates) if real is not None else pd.Series(0.0, index=dates)
        real_m = avail(real, dates)
        synth_r_full = synthetic_lev(und, lev, dates)
        if synth_r_full is None:
            # No underlier at all — only real ETF contributes
            rets_eq[name] = real_r
            avail_eq[name] = real_m
        else:
            synth_r, synth_m = synth_r_full
            rets_eq[name] = real_r.where(real_m, synth_r)
            avail_eq[name] = real_m | synth_m

    # Reconstruct prices for momentum ranking
    rets_df = pd.DataFrame(rets_eq)
    avail_eq_df = pd.DataFrame(avail_eq)

    # Crypto: real returns where live, else simply NaN (drop from ranking)
    btc = load_etf("BTC_USD")
    eth = load_etf("ETH_USD")
    btc_r = pct(btc, dates) if btc is not None else pd.Series(0.0, index=dates)
    eth_r = pct(eth, dates) if eth is not None else pd.Series(0.0, index=dates)
    btc_m = avail(btc, dates)
    eth_m = avail(eth, dates)

    # Prices for ranking: use cum-product of returns (initial level arbitrary).
    prices_eq = (1 + rets_df).cumprod() * 100
    # Before a name's own availability, set price to NaN so momentum can't see it
    for name in EQUITY:
        prices_eq[name] = prices_eq[name].where(avail_eq_df[name])

    btc_px = (1 + btc_r.where(btc_m, 0)).cumprod() * 100
    btc_px = btc_px.where(btc_m)
    eth_px = (1 + eth_r.where(eth_m, 0)).cumprod() * 100
    eth_px = eth_px.where(eth_m)

    prices = pd.concat([prices_eq, btc_px.rename("BTC_USD"), eth_px.rename("ETH_USD")], axis=1)
    rets_all = pd.concat([rets_df,
                          btc_r.where(btc_m, 0).rename("BTC_USD"),
                          eth_r.where(eth_m, 0).rename("ETH_USD")], axis=1)

    bil = load_etf("BIL")
    shy = load_etf("SHY")
    bil_r = pct(bil, dates) if bil is not None else pd.Series(0.0, index=dates)
    shy_r = pct(shy, dates) if shy is not None else pd.Series(0.0, index=dates)
    # Pre-BIL cash: use SHY
    cash = bil_r.where(avail(bil, dates), shy_r)

    reg_eq = spy_regime(dates)
    reg_bt = btc_regime(dates)

    current = pd.Series(0.0, index=UNIVERSE)
    port = pd.Series(0.0, index=dates)
    w_crypto = pd.Series(0.0, index=dates)
    w_equity = pd.Series(0.0, index=dates)
    last_idx = -REBAL

    for i in range(len(dates)):
        if i >= LOOKBACK and i - last_idx >= REBAL:
            # Only names that are live TODAY and were live LOOKBACK days ago
            live_now = prices.iloc[i].notna()
            live_then = prices.iloc[i - LOOKBACK].notna()
            live = live_now & live_then
            momo = (prices.iloc[i] / prices.iloc[i - LOOKBACK] - 1).where(live)
            ranked = momo.dropna().sort_values(ascending=False)
            positive = [t for t in ranked.index if momo[t] > 0]
            top = positive[:TOP_N]
            new = pd.Series(0.0, index=UNIVERSE)
            if top:
                w = 1.0 / len(top)
                for t in top:
                    new[t] = min(w, CAP)
            tc = (new - current).abs().sum() * (TC_BPS / 1e4)
            port.iloc[i] -= tc
            current = new
            last_idx = i

        eff = current.copy()
        geq = reg_eq.iloc[i]
        gbt = reg_bt.iloc[i]
        off_eq = sum(current[t] for t in EQUITY) * (1 - geq)
        off_bt = sum(current[t] for t in CRYPTO) * (1 - gbt)
        for t in EQUITY: eff[t] = current[t] * geq
        for t in CRYPTO: eff[t] = current[t] * gbt
        r = (rets_all.iloc[i] * eff).sum() + (off_eq + off_bt) * cash.iloc[i]
        port.iloc[i] += r
        w_crypto.iloc[i] = sum(eff[t] for t in CRYPTO)
        w_equity.iloc[i] = sum(eff[t] for t in EQUITY)

    # Tier assignment
    all_eq_real = pd.DataFrame({t: avail(load_etf(t), dates) for t in EQUITY}).all(axis=1)
    tier = pd.Series(4, index=dates, dtype=int)
    tier[all_eq_real] = 3                       # all equity real, no crypto
    tier[all_eq_real & btc_m] = 2               # add BTC
    tier[all_eq_real & btc_m & eth_m] = 1       # add ETH (live)
    source = np.where(tier == 1, "live", "proxy")

    # Benchmarks
    r_spy = pct(spy, dates)
    agg = load_etf("AGG")
    r_agg = pct(agg, dates) if agg is not None else pd.Series(0.0, index=dates)

    out = pd.DataFrame({
        "Close": port,
        "SPY": r_spy,
        "AGG": r_agg,
        "source": source,
        "tier": tier.values,
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / "nova_proxy_returns.csv")

    for label, mask in [("full",              slice(None)),
                        ("t1 live",           out["tier"] == 1),
                        ("t2 btc-only",       out["tier"] == 2),
                        ("t3 no-crypto",      out["tier"] == 3),
                        ("t4 synth-lev",      out["tier"] == 4)]:
        sub = out.index[mask] if not isinstance(mask, slice) else out.index
        r = port.loc[sub]
        if len(r) < 2: continue
        ar = r.mean() * 252; av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        d1, d2 = r.index[0].date(), r.index[-1].date()
        print(f"  {label:15s} {d1}..{d2} ({len(r)/252:.1f}y): "
              f"SR={sr:.2f} Ret={ar*100:.2f}% Vol={av*100:.2f}% MDD={mdd*100:.1f}%")

    # Compare to SPY / AGG on same full window
    print()
    for name, r in [("SPY full", r_spy), ("AGG full", r_agg)]:
        ar = r.mean() * 252; av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        print(f"  {name:15s} {r.index[0].date()}..{r.index[-1].date()}: "
              f"SR={sr:.2f} Ret={ar*100:.2f}% Vol={av*100:.2f}% MDD={mdd*100:.1f}%")


if __name__ == "__main__":
    build()
