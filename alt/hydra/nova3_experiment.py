"""NOVA3 — aggressive SR 2+ attempt WITHOUT vol scaling.

Adds three big ideas that NOVA2 didn't explore:

1. **Volatility risk premium (VRP)**: long SVXY when VIX term-structure
   proxy is calm (VIX level below its 63d median + slope). Regime
   filter only, no vol scaling. This is the single biggest historical
   SR lift on ETF data. Sleeve is live from 2011 when SVXY begins.

2. **ML regime classifier**: XGBoost trained on macro features
   (yield curve, credit spreads, VIX, SPY trend, dollar) to predict
   whether next-month SPY return is positive. Signal re-fits monthly
   on rolling 5y window. Output drives a binary long-SPY / long-IEF
   switch (no vol scaling; pure regime-binary).

3. **Weekly cross-sectional mean reversion on sectors (CSMR-W)**:
   rank 10 sectors by last week's return, long the 2 worst, short
   the 2 best. Weekly rebal (5 trading days) keeps TC bearable.
   5 bps per leg.

Plus NOVA2's survivors (Faber trend sleeves, GEM, Halloween, TOY,
gold-crisis). Equal weight across live sleeves, monthly rebalance
of weights, no vol scaling anywhere."""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, load_fred, stats


TC_BPS = 15.0
TC_BPS_ETF = 5.0


# ---------- helpers ----------

def monthly_first_flag(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def _rebal_to_monthly(raw):
    first = monthly_first_flag(raw.index)
    return raw.where(first, np.nan).ffill().fillna(False).astype(bool)


def _rank_within_month(dates):
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    df["mk"] = df["date"].dt.to_period("M")
    df["fwd"] = df.groupby("mk").cumcount() + 1
    df["rev"] = df.groupby("mk").cumcount(ascending=False) + 1
    return df["fwd"].values, df["rev"].values


def binary_long(tic, signal, dates, off_tic=None, tc_bps=TC_BPS):
    L = load_etf(tic).reindex(dates).ffill().pct_change().fillna(0)
    sig = signal.reindex(dates).fillna(False).astype(bool).shift(1).fillna(False)
    if off_tic is None:
        r = pd.Series(np.where(sig, L, 0), index=dates)
    else:
        B = load_etf(off_tic).reindex(dates).ffill().pct_change().fillna(0)
        r = pd.Series(np.where(sig, L, B), index=dates)
    changes = sig.astype(int).diff().abs().fillna(0)
    tc = changes * (tc_bps / 1e4) * (2 if off_tic else 1)
    return r - tc


# ---------- trend-following (Faber 10m SMA) ----------

def ttm_signal(tic, dates, months=10):
    p = load_etf(tic).reindex(dates).ffill()
    sma = p.rolling(months * 21).mean()
    raw = p > sma
    return _rebal_to_monthly(raw)


def s_trend_spy(dates):
    return binary_long("SPY", ttm_signal("SPY", dates), dates, off_tic="IEF").rename("s_trend_spy")


def s_trend_qqq(dates):
    return binary_long("QQQ", ttm_signal("QQQ", dates), dates, off_tic="IEF").rename("s_trend_qqq")


def s_trend_eem(dates):
    return binary_long("EEM", ttm_signal("EEM", dates), dates, off_tic="IEF").rename("s_trend_eem")


def s_trend_efa(dates):
    return binary_long("EFA", ttm_signal("EFA", dates), dates, off_tic="IEF").rename("s_trend_efa")


def s_trend_vnq(dates):
    return binary_long("VNQ", ttm_signal("VNQ", dates), dates, off_tic="IEF").rename("s_trend_vnq")


def s_trend_gld(dates):
    return binary_long("GLD", ttm_signal("GLD", dates), dates, off_tic="BIL").rename("s_trend_gld")


def s_trend_tlt(dates):
    return binary_long("TLT", ttm_signal("TLT", dates), dates, off_tic="BIL").rename("s_trend_tlt")


# ---------- GEM dual momentum ----------

def s_gem(dates):
    spy = load_etf("SPY").reindex(dates).ffill()
    efa = load_etf("EFA").reindex(dates).ffill()
    bil = load_etf("BIL").reindex(dates).ffill()
    ief = load_etf("IEF").reindex(dates).ffill()
    mom_spy = spy.pct_change(252)
    mom_efa = efa.pct_change(252)
    mom_bil = bil.pct_change(252)
    risk_on = _rebal_to_monthly(mom_spy > mom_bil)
    spy_wins = _rebal_to_monthly(mom_spy > mom_efa)
    sig_spy = (risk_on & spy_wins).shift(1).fillna(False)
    sig_efa = (risk_on & ~spy_wins).shift(1).fillna(False)
    sig_ief = (~risk_on).shift(1).fillna(False)
    r = (sig_spy.astype(float) * spy.pct_change().fillna(0) +
         sig_efa.astype(float) * efa.pct_change().fillna(0) +
         sig_ief.astype(float) * ief.pct_change().fillna(0))
    mat = pd.DataFrame({"spy": sig_spy, "efa": sig_efa, "ief": sig_ief}).astype(int)
    changes = mat.diff().abs().sum(axis=1).fillna(0)
    return (r - changes * (TC_BPS / 1e4)).rename("s_gem")


# ---------- calendar ----------

def s_halloween(dates):
    sig = pd.Series([d.month in [11, 12, 1, 2, 3, 4] for d in dates], index=dates)
    return binary_long("SPY", sig, dates).rename("s_halloween")


def s_toy_iwm(dates):
    fwd, _ = _rank_within_month(dates)
    is_jan = np.array([d.month == 1 for d in dates])
    sig = pd.Series(is_jan & (fwd <= 5), index=dates)
    return binary_long("IWM", sig, dates).rename("s_toy")


def s_gold_crisis(dates):
    gld = load_etf("GLD").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    raw = (gld.pct_change(63) > 0) & (spy.pct_change(63) < 0)
    sig = _rebal_to_monthly(raw)
    return binary_long("GLD", sig, dates).rename("s_gold_crisis")


# ---------- NEW: Volatility risk premium ----------

def s_vrp_svxy(dates):
    """Long SVXY when VIX is below its 63d median (calm regime).
    Cash otherwise. Monthly signal evaluation. SVXY inception 2011-10.
    TC 15 bps on regime changes."""
    svxy = load_etf("SVXY")
    if svxy is None:
        return pd.Series(0.0, index=dates).rename("s_vrp_svxy")
    svxy = svxy.reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    vix_med = vix.rolling(63).median()
    raw = (vix < vix_med) & vix.notna()
    sig = _rebal_to_monthly(raw)
    # Only active where SVXY has price data
    live = svxy.notna()
    sig = (sig & live).astype(bool)
    r_svxy = svxy.pct_change().fillna(0)
    sig_sh = sig.shift(1).fillna(False)
    r = pd.Series(np.where(sig_sh, r_svxy, 0), index=dates)
    changes = sig_sh.astype(int).diff().abs().fillna(0)
    tc = changes * (TC_BPS / 1e4)
    return (r - tc).rename("s_vrp_svxy")


# ---------- NEW: Weekly cross-sectional mean reversion ----------

SECTOR_TICS = ["XLK", "XLF", "XLE", "XLB", "XLI", "XLY", "XLP", "XLU", "XLV", "IYR"]


def s_csmr_weekly(dates):
    """Every Monday, rank 10 sector ETFs by prior week's return. Long
    the 2 worst, short the 2 best, hold 1 week, 5 bps/leg TC."""
    prices = {t: load_etf(t).reindex(dates).ffill() for t in SECTOR_TICS}
    R = pd.DataFrame({t: prices[t].pct_change().fillna(0) for t in SECTOR_TICS})
    # Weekly returns (last 5 trading days)
    weekly = R.rolling(5).sum().shift(1)

    # Friday-to-Monday flag: first day of each ISO week
    is_first_of_week = pd.Series(False, index=dates)
    is_first_of_week.iloc[0] = True
    for i in range(1, len(dates)):
        if dates[i].weekday() < dates[i - 1].weekday():
            is_first_of_week.iloc[i] = True

    ranks = weekly.rank(axis=1, method="first")
    n = len(SECTOR_TICS)
    long_mask = ranks <= 2
    short_mask = ranks >= n - 1
    w_daily = long_mask.astype(float) * 0.5 - short_mask.astype(float) * 0.5
    # Freeze weights to week boundaries (rebal Monday only)
    w = w_daily.where(is_first_of_week, np.nan).ffill().fillna(0)
    # Mask out pre-history (need all 10 live)
    first_live = max(prices[t].dropna().index[0] for t in SECTOR_TICS)
    mask = pd.Series(dates >= first_live, index=dates)
    w = w.where(mask, 0)

    ret = (w * R).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    tc = turn * (TC_BPS_ETF / 1e4)
    return (ret - tc).rename("s_csmr_w")


# ---------- NEW: ML regime classifier ----------

def build_features(dates):
    """Daily feature matrix (features known at date t, predict next
    month's SPY direction)."""
    feats = {}
    spy = load_etf("SPY").reindex(dates).ffill()
    feats["spy_ret_20"] = spy.pct_change(20)
    feats["spy_ret_63"] = spy.pct_change(63)
    feats["spy_ret_252"] = spy.pct_change(252)
    feats["spy_vs_sma200"] = (spy / spy.rolling(200).mean()) - 1

    vix = load_fred("VIXCLS").reindex(dates).ffill()
    feats["vix"] = vix
    feats["vix_chg_20"] = vix.diff(20)
    feats["vix_level_hi"] = (vix > 25).astype(float)

    t10y2y = load_fred("T10Y2Y").reindex(dates).ffill()
    feats["t10y2y"] = t10y2y
    feats["t10y2y_chg"] = t10y2y.diff(63)

    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    feats["hy_spread"] = hy
    feats["hy_chg_20"] = hy.diff(20)

    dxy = load_fred("DTWEXBGS").reindex(dates).ffill()
    feats["dxy_ret_63"] = dxy.pct_change(63)

    tlt = load_etf("TLT").reindex(dates).ffill()
    feats["tlt_ret_63"] = tlt.pct_change(63)

    return pd.DataFrame(feats).ffill()


def s_ml_regime(dates, min_train=252 * 5):
    """Monthly XGBoost classifier for SPY next-21d direction.
    - Rolling 5y train, retrain monthly.
    - Predict on month-end; if p(up) > 0.55 long SPY, else if p(up) < 0.45
      long IEF, else cash.
    - Signal executed next day (1-bar lag)."""
    from xgboost import XGBClassifier

    X = build_features(dates)
    spy = load_etf("SPY").reindex(dates).ffill()
    fwd_ret = spy.pct_change(21).shift(-21)  # forward 21d return
    y = (fwd_ret > 0).astype(int)

    valid = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid]
    y = y.loc[valid]

    first = monthly_first_flag(pd.Index(dates))
    # Find month-start dates
    month_starts = [d for d in dates if first.loc[d]]

    signal = pd.Series(0, index=dates)  # 0=cash, 1=SPY, -1=IEF
    last_pred_day = None

    for ms in month_starts:
        # Train on data available up to ms - 1 day
        train_end = ms - pd.Timedelta(days=1)
        train_X = X.loc[:train_end]
        train_y = y.loc[:train_end]
        # Need enough data with target (target shifted -21, so exclude last 21)
        train_X = train_X.iloc[:-21] if len(train_X) > 21 else train_X
        train_y = train_y.iloc[:-21] if len(train_y) > 21 else train_y
        if len(train_X) < min_train:
            continue
        # Keep only last 5y
        train_X = train_X.iloc[-252 * 5:]
        train_y = train_y.iloc[-252 * 5:]

        try:
            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42, verbosity=0,
                eval_metric="logloss",
            )
            model.fit(train_X.values, train_y.values)
            if ms not in X.index:
                continue
            feat_row = X.loc[ms].values.reshape(1, -1)
            p_up = model.predict_proba(feat_row)[0, 1]
            if p_up > 0.55:
                reg = 1
            elif p_up < 0.45:
                reg = -1
            else:
                reg = 0
            # Apply regime until next month boundary
            ms_idx = dates.get_loc(ms)
            # find next month-start
            next_ms = None
            for m2 in month_starts:
                if m2 > ms:
                    next_ms = m2
                    break
            end_idx = dates.get_loc(next_ms) if next_ms is not None else len(dates)
            signal.iloc[ms_idx:end_idx] = reg
        except Exception:
            continue

    # Translate signal to return
    signal_sh = signal.shift(1).fillna(0)
    r_spy = spy.pct_change().fillna(0)
    ief = load_etf("IEF").reindex(dates).ffill()
    r_ief = ief.pct_change().fillna(0)

    r = pd.Series(0.0, index=dates)
    r.loc[signal_sh == 1] = r_spy.loc[signal_sh == 1]
    r.loc[signal_sh == -1] = r_ief.loc[signal_sh == -1]
    changes = signal_sh.diff().abs().fillna(0)
    # Up to 2 legs per regime change
    tc = (changes > 0).astype(float) * (TC_BPS / 1e4) * 2
    return (r - tc).rename("s_ml_regime")


# ---------- ensemble ----------

SLEEVES = [
    s_trend_spy, s_trend_qqq, s_trend_eem, s_trend_efa, s_trend_vnq,
    s_trend_gld, s_trend_tlt,
    s_gem,
    s_halloween, s_toy_iwm, s_gold_crisis,
    s_vrp_svxy,
    s_csmr_weekly,
    s_ml_regime,
]


def build(dates):
    out = {}
    for fn in SLEEVES:
        print(f"  Building {fn.__name__}...", flush=True)
        s = fn(dates)
        out[s.name] = s
    return pd.DataFrame(out).reindex(dates).fillna(0)


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()} ({len(dates)/252:.1f}y)")
    print(f"NOVA3 — {len(SLEEVES)} sleeves, equal-weight, no vol scaling\n")

    df = build(dates)
    print("\nPer-sleeve stats (full history):")
    for c in df.columns:
        s = stats(df[c], c)
        nz = df[c][df[c] != 0]
        start = nz.index[0].date() if len(nz) else None
        print(f"  {c:18s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  live≥{start}")

    valid = (df != 0).sum(axis=1) >= 5
    corr = df[valid].corr()
    tri = corr.values[np.triu_indices_from(corr, k=1)]
    print(f"\nMean |pairwise corr| = {np.mean(np.abs(tri)):.3f}   "
          f"Median = {np.median(np.abs(tri)):.3f}   Max = {np.max(np.abs(tri)):.2f}")

    live = (df != 0).cummax().astype(float)
    w = live.div(live.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    port = (w * df).sum(axis=1)
    nz = (df != 0).any(axis=1)
    port = port[nz]

    s = stats(port, "NOVA3 raw")
    print(f"\n  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    native_vol = port.std() * np.sqrt(252)
    for tgt_str, tgt in [("10%", 0.10), ("20%", 0.20)]:
        lev = tgt / native_vol
        r = port * lev
        s = stats(r, f"static lev {lev:.2f}x → {tgt_str}")
        print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    # Post-2012 window (SVXY + ML both live)
    P = pd.Timestamp("2013-01-01")
    post = port.loc[P:]
    s = stats(post, "NOVA3 post-2013")
    print(f"\n  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
    IS = pd.Timestamp("2018-01-01")
    lev_port = port * (0.10 / native_vol)
    for p, lbl in [(lev_port.loc[:IS], "IS ≤2018"), (lev_port.loc[IS:], "OOS >2018"),
                   (lev_port.loc[P:IS], "post-2013 IS"), (lev_port.loc[IS:], "post-2013 OOS")]:
        s = stats(p, lbl)
        print(f"  {lbl:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")


if __name__ == "__main__":
    main()
