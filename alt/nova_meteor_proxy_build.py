"""Build NOVA METEOR returns with proxies, extended to 2005-01.

Same tiering as the v1 nova_proxy_build (TIER1 live / TIER2 BTC-only /
TIER3 no-crypto / TIER4 synthetic-leverage) but applies the METEOR
mechanics on top:
  - LB=120, TOP_N=3, CAP=1.0, REBAL=21 (monthly)
  - 5.5x base overlay throttled by PDOT (378d rolling HWM, floor=0.30)
  - 15d NAV-trend asymmetric multiplier (floor = 0.30 * 0.40 = 0.12)
  - SPY>200d & VIX<30 equity gate, BTC>200d crypto gate
  - DGS3MO financing on the overlay notional
  - 15bps transaction cost on turnover × overlay

Outputs:
  data/results/nova_meteor_proxy_returns.csv           (with crypto)
  data/results/nova_meteor_proxy_nocrypto_returns.csv  (equity-only)

Columns: Close, SPY, AGG, source, tier, Overlay
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

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

LOOKBACK = 120
TOP_N = 3
CAP = 1.00
REBAL = 21
OVERLAY_BASE = 5.5
DD_FLOOR = 0.30
NAV_WIN = 15
PDOT_WIN = 378
NAV_FLOOR_MULT = 0.40
BTC_MA = 200
SPY_MA = 200
VIX_CAP = 30.0
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


def avail_series(s, dates):
    return pd.Series(False if s is None else (s.index.min() <= dates), index=dates)


def financing(dates):
    d = load_fred("DGS3MO").reindex(dates).ffill() / 100.0
    return d.fillna(0) / 252.0


def synthetic_lev(underlier, leverage, dates, fin):
    s = load_etf(underlier)
    if s is None: return None, None
    r = pct(s, dates)
    m = avail_series(s, dates)
    expense = 0.0095 / 252
    synth = leverage * r - expense - (leverage - 1) * fin
    return synth.where(m, 0.0), m


def build(include_crypto=True, out_name="nova_meteor_proxy_returns.csv",
          label="METEOR proxy"):
    spy = load_etf("SPY")
    dates = spy.index
    print(f"{label}: {dates[0].date()} .. {dates[-1].date()} "
          f"({len(dates)/252:.1f}y)  include_crypto={include_crypto}")

    fin = financing(dates)

    rets_eq = {}; avail_eq = {}
    for name, (und, lev) in EQUITY_MAP.items():
        real = load_etf(name)
        real_r = pct(real, dates) if real is not None else pd.Series(0.0, index=dates)
        real_m = avail_series(real, dates)
        synth_r, synth_m = synthetic_lev(und, lev, dates, fin)
        if synth_r is None:
            rets_eq[name] = real_r
            avail_eq[name] = real_m
        else:
            rets_eq[name] = real_r.where(real_m, synth_r)
            avail_eq[name] = real_m | synth_m

    rets_df = pd.DataFrame(rets_eq)
    avail_eq_df = pd.DataFrame(avail_eq)

    btc = load_etf("BTC_USD"); eth = load_etf("ETH_USD")
    btc_r = pct(btc, dates) if btc is not None else pd.Series(0.0, index=dates)
    eth_r = pct(eth, dates) if eth is not None else pd.Series(0.0, index=dates)
    btc_m = avail_series(btc, dates)
    eth_m = avail_series(eth, dates)

    prices_eq = (1 + rets_df).cumprod() * 100
    for name in EQUITY:
        prices_eq[name] = prices_eq[name].where(avail_eq_df[name])

    if include_crypto:
        btc_px = (1 + btc_r.where(btc_m, 0)).cumprod() * 100
        btc_px = btc_px.where(btc_m)
        eth_px = (1 + eth_r.where(eth_m, 0)).cumprod() * 100
        eth_px = eth_px.where(eth_m)
        prices = pd.concat([prices_eq, btc_px.rename("BTC_USD"),
                            eth_px.rename("ETH_USD")], axis=1)
        rets_all = pd.concat([rets_df,
                              btc_r.where(btc_m, 0).rename("BTC_USD"),
                              eth_r.where(eth_m, 0).rename("ETH_USD")], axis=1)
        universe = EQUITY + CRYPTO
    else:
        prices = prices_eq.copy()
        rets_all = rets_df.copy()
        universe = list(EQUITY)

    bil = load_etf("BIL"); shy = load_etf("SHY")
    bil_r = pct(bil, dates) if bil is not None else pd.Series(0.0, index=dates)
    shy_r = pct(shy, dates) if shy is not None else pd.Series(0.0, index=dates)
    cash = bil_r.where(avail_series(bil, dates), shy_r).values

    vix = load_fred("VIXCLS")
    spy_a = spy.reindex(dates).ffill()
    reg_eq_s = (spy_a > spy_a.rolling(SPY_MA).mean())
    if vix is not None:
        reg_eq_s = reg_eq_s & (vix.reindex(dates).ffill() < VIX_CAP)
    reg_eq = reg_eq_s.shift(1).fillna(False).astype(float).values

    if include_crypto:
        btc_a = btc.reindex(dates).ffill() if btc is not None else pd.Series(np.nan, index=dates)
        reg_bt = (btc_a > btc_a.rolling(BTC_MA).mean()).shift(1).fillna(False).astype(float).values
    else:
        reg_bt = np.zeros(len(dates))

    rf = financing(dates).values

    n = len(dates); m_u = len(universe)
    eq_idx = np.array([universe.index(t) for t in EQUITY])
    cr_idx = np.array([universe.index(t) for t in CRYPTO if t in universe]) if include_crypto else np.array([], dtype=int)

    P = prices.values
    avail_np = prices.notna().values
    R = rets_all.values

    current = np.zeros(m_u)
    port = np.zeros(n)
    nav = np.ones(n + 1)
    overlay_series = np.zeros(n)
    last_idx = -REBAL
    tc_pending = 0.0
    start_req = max(LOOKBACK + 2, 2)

    for i in range(n):
        if i > start_req and i - last_idx >= REBAL:
            live_now = avail_np[i - 1]
            live_then = avail_np[i - 1 - LOOKBACK]
            live = live_now & live_then
            with np.errstate(invalid="ignore", divide="ignore"):
                mom = np.where(live, P[i - 1] / P[i - 1 - LOOKBACK] - 1.0, np.nan)
            valid = ~np.isnan(mom) & (mom > 0)
            order = np.argsort(-np.where(valid, mom, -np.inf))
            picks = [int(k) for k in order if valid[k]][:TOP_N]
            new = np.zeros(m_u)
            if picks:
                w = min(1.0 / len(picks), CAP)
                for k in picks: new[k] = w
            tc_pending = float(np.abs(new - current).sum())
            current = new
            last_idx = i

        eff = current.copy()
        geq = reg_eq[i]; gbt = reg_bt[i]
        off_eq = current[eq_idx].sum() * (1 - geq)
        off_bt = current[cr_idx].sum() * (1 - gbt) if len(cr_idx) else 0.0
        eff[eq_idx] = current[eq_idx] * geq
        if len(cr_idx): eff[cr_idx] = current[cr_idx] * gbt
        gross = float(np.nansum(R[i] * eff)) + (off_eq + off_bt) * cash[i]
        invested = float(eff.sum())

        if i > 0:
            lo = max(1, i - (PDOT_WIN - 1))
            hwm = nav[lo:i + 1].max()
            dd = (nav[i] / hwm) - 1.0
        else:
            dd = 0.0
        pdot = max(0.0, 1.0 + dd / DD_FLOOR)

        if i > NAV_WIN and nav[i - NAV_WIN] > 0:
            nav_mom = nav[i] / nav[i - NAV_WIN] - 1.0
        else:
            nav_mom = 0.0
        nav_floor = DD_FLOOR * NAV_FLOOR_MULT
        trend_mult = max(0.0, min(1.0, 1.0 + nav_mom / nav_floor)) if nav_mom < 0 else 1.0

        overlay_t = OVERLAY_BASE * pdot * trend_mult
        overlay_series[i] = overlay_t

        tc_today = 0.0
        if last_idx == i and tc_pending > 0:
            tc_today = tc_pending * (TC_BPS / 1e4) * overlay_t
            tc_pending = 0.0

        levered = overlay_t * gross - (overlay_t - 1.0) * invested * rf[i] - tc_today
        port[i] = levered
        nav[i + 1] = nav[i] * (1 + levered)

    port_s = pd.Series(port, index=dates)
    r_spy = pct(spy, dates)
    agg = load_etf("AGG")
    r_agg = pct(agg, dates) if agg is not None else pd.Series(0.0, index=dates)

    all_eq_real = pd.DataFrame({t: avail_series(load_etf(t), dates) for t in EQUITY}).all(axis=1)
    if include_crypto:
        tier = pd.Series(4, index=dates, dtype=int)
        tier[all_eq_real] = 3
        tier[all_eq_real & btc_m] = 2
        tier[all_eq_real & btc_m & eth_m] = 1
        source = np.where(tier == 1, "live", "proxy")
    else:
        tier = pd.Series(4, index=dates, dtype=int)
        tier[all_eq_real] = 1
        source = np.where(tier == 1, "live", "proxy")

    out = pd.DataFrame({
        "Close": port_s,
        "SPY": r_spy,
        "AGG": r_agg,
        "source": source,
        "tier": tier.values,
        "Overlay": pd.Series(overlay_series, index=dates),
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / out_name)

    rows = [("full", slice(None))]
    if include_crypto:
        rows += [("t1 live", out["tier"] == 1),
                 ("t2 btc-only", out["tier"] == 2),
                 ("t3 no-crypto", out["tier"] == 3),
                 ("t4 synth-lev", out["tier"] == 4)]
    else:
        rows += [("t1 all-real", out["tier"] == 1),
                 ("t4 synth-lev", out["tier"] == 4)]
    for lbl, mask in rows:
        sub = out.index[mask] if not isinstance(mask, slice) else out.index
        r = port_s.loc[sub]
        if len(r) < 2: continue
        ar = r.mean() * 252; av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        d1, d2 = r.index[0].date(), r.index[-1].date()
        print(f"  {lbl:15s} {d1}..{d2} ({len(r)/252:.1f}y): "
              f"SR={sr:.2f} Ret={ar*100:.2f}% Vol={av*100:.2f}% MDD={mdd*100:.1f}% "
              f"NAVx={cum.iloc[-1]:.1f}")

    print()
    for name, r in [("SPY full", r_spy), ("AGG full", r_agg)]:
        ar = r.mean() * 252; av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        print(f"  {name:15s} {r.index[0].date()}..{r.index[-1].date()}: "
              f"SR={sr:.2f} Ret={ar*100:.2f}% Vol={av*100:.2f}% MDD={mdd*100:.1f}%")

    print(f"  mean overlay applied (post-warmup): "
          f"{overlay_series[LOOKBACK:].mean():.2f}x (base {OVERLAY_BASE}x)")


if __name__ == "__main__":
    build(include_crypto=True,
          out_name="nova_meteor_proxy_returns.csv",
          label="METEOR proxy (with crypto)")
    print()
    build(include_crypto=False,
          out_name="nova_meteor_proxy_nocrypto_returns.csv",
          label="METEOR proxy (no crypto)")
