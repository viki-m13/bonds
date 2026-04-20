"""NOVA4 — tight winners-only ensemble + aggressive regime-filtered VRP.

After NOVA2/3, winners are the trend-following sleeves (Faber TTM).
Drop the weak/negative ones (CSMR, TOY, ML-regime). Replace with a
multi-regime-filtered short-vol sleeve and a pairs / macro overlay.

No vol scaling. No leverage targeting. Binary long/flat/cash only.
Equal-weight live sleeves. Monthly weight rebalance.

Sleeves (kept):
  trend_spy, trend_qqq, trend_eem, trend_efa, trend_gld, trend_tlt
  gem, halloween, gold_crisis

New / redesigned:
  vrp_hard       — long SVXY only when VIX<20 AND VIX<VIX_MA10 AND
                   SPY>SMA50 (3-way regime filter). Cash otherwise.
  vrp_soft       — long SVXY only when VIX<30 (simpler baseline).
  trend_ensemble — majority-vote trend: long SPY only when ≥4 of 7
                   assets above their 10m SMA. Reduces whipsaw.
  lq_credit      — long LQD when HY spread tightening (BAMLH0A0HYM2
                   below 63d MA).
  curve_steepen  — long IEF short SHY when 10y-2y spread > 0 and
                   widening (non-inverted curve). Proxy via TLT/SHY.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, load_fred, stats


TC_BPS = 15.0


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


# ---------- trend ----------

def ttm_signal(tic, dates, months=10):
    p = load_etf(tic).reindex(dates).ffill()
    sma = p.rolling(months * 21).mean()
    return _rebal_to_monthly(p > sma)


def s_trend_spy(dates):
    return binary_long("SPY", ttm_signal("SPY", dates), dates, off_tic="IEF").rename("s_trend_spy")


def s_trend_qqq(dates):
    return binary_long("QQQ", ttm_signal("QQQ", dates), dates, off_tic="IEF").rename("s_trend_qqq")


def s_trend_eem(dates):
    return binary_long("EEM", ttm_signal("EEM", dates), dates, off_tic="IEF").rename("s_trend_eem")


def s_trend_efa(dates):
    return binary_long("EFA", ttm_signal("EFA", dates), dates, off_tic="IEF").rename("s_trend_efa")


def s_trend_gld(dates):
    return binary_long("GLD", ttm_signal("GLD", dates), dates, off_tic="BIL").rename("s_trend_gld")


def s_trend_tlt(dates):
    return binary_long("TLT", ttm_signal("TLT", dates), dates, off_tic="BIL").rename("s_trend_tlt")


def s_trend_majority(dates):
    """Long SPY if >=4 of 7 trend-assets are above their 10m SMA;
    otherwise long IEF. Breadth-filtered trend."""
    tics = ["SPY", "QQQ", "EEM", "EFA", "VNQ", "GLD", "TLT"]
    votes = []
    for t in tics:
        p = load_etf(t).reindex(dates).ffill()
        sma = p.rolling(210).mean()
        votes.append((p > sma).astype(int))
    V = pd.DataFrame({t: v for t, v in zip(tics, votes)}).fillna(0)
    raw = V.sum(axis=1) >= 4
    sig = _rebal_to_monthly(raw)
    return binary_long("SPY", sig, dates, off_tic="IEF").rename("s_trend_maj")


# ---------- GEM ----------

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


def s_gold_crisis(dates):
    gld = load_etf("GLD").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    raw = (gld.pct_change(63) > 0) & (spy.pct_change(63) < 0)
    sig = _rebal_to_monthly(raw)
    return binary_long("GLD", sig, dates).rename("s_gold_crisis")


# ---------- Volatility risk premium, hard regime ----------

def s_vrp_hard(dates):
    """Long SVXY when ALL of: VIX<20, VIX<VIX_MA10, SPY>SMA50.
    Cash otherwise. Signal evaluated daily but locked to weekly
    rebalance (Monday) to reduce whipsaw."""
    svxy = load_etf("SVXY")
    if svxy is None:
        return pd.Series(0.0, index=dates).rename("s_vrp_hard")
    svxy = svxy.reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    vix_ma = vix.rolling(10).mean()
    spy = load_etf("SPY").reindex(dates).ffill()
    spy_sma = spy.rolling(50).mean()

    raw = (vix < 20) & (vix < vix_ma) & (spy > spy_sma) & svxy.notna()

    # Freeze to weekly (Monday)
    is_first_of_week = pd.Series(False, index=dates)
    is_first_of_week.iloc[0] = True
    for i in range(1, len(dates)):
        if dates[i].weekday() < dates[i - 1].weekday():
            is_first_of_week.iloc[i] = True
    sig = raw.where(is_first_of_week, np.nan).ffill().fillna(False).astype(bool)

    sig_sh = sig.shift(1).fillna(False)
    r_svxy = svxy.pct_change().fillna(0)
    r = pd.Series(np.where(sig_sh, r_svxy, 0), index=dates)
    changes = sig_sh.astype(int).diff().abs().fillna(0)
    return (r - changes * (TC_BPS / 1e4)).rename("s_vrp_hard")


def s_vrp_trend(dates):
    """Long SVXY when SVXY > its own 63d SMA AND VIX < 25.
    Trend-follow SVXY + regime filter. Monthly signal."""
    svxy = load_etf("SVXY")
    if svxy is None:
        return pd.Series(0.0, index=dates).rename("s_vrp_trend")
    svxy = svxy.reindex(dates).ffill()
    sma = svxy.rolling(63).mean()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    raw = (svxy > sma) & (vix < 25) & svxy.notna()
    sig = _rebal_to_monthly(raw)
    sig_sh = sig.shift(1).fillna(False)
    r_svxy = svxy.pct_change().fillna(0)
    r = pd.Series(np.where(sig_sh, r_svxy, 0), index=dates)
    changes = sig_sh.astype(int).diff().abs().fillna(0)
    return (r - changes * (TC_BPS / 1e4)).rename("s_vrp_trend")


# ---------- credit / curve ----------

def s_lqd_carry(dates):
    """Long LQD when HY spread (BAMLH0A0HYM2) < its 63d MA
    (credit tightening regime). Cash otherwise. Monthly."""
    lqd = load_etf("LQD").reindex(dates).ffill()
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    raw = hy < hy.rolling(63).mean()
    sig = _rebal_to_monthly(raw)
    return binary_long("LQD", sig, dates).rename("s_lqd_carry")


def s_curve(dates):
    """Long TLT/IEF when 10y-2y > 0 (non-inverted) and rising.
    Flipped when inverted: stay in SHY (duration hedge)."""
    curve = load_fred("T10Y2Y").reindex(dates).ffill()
    raw = (curve > 0) & (curve > curve.rolling(63).mean())
    sig = _rebal_to_monthly(raw)
    return binary_long("TLT", sig, dates, off_tic="SHY").rename("s_curve")


# ---------- ensemble ----------

SLEEVES = [
    s_trend_spy, s_trend_qqq, s_trend_eem, s_trend_efa,
    s_trend_gld, s_trend_tlt, s_trend_majority,
    s_gem,
    s_halloween, s_gold_crisis,
    s_vrp_hard, s_vrp_trend,
    s_lqd_carry, s_curve,
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
    print(f"NOVA4 — {len(SLEEVES)} sleeves, equal-weight, no vol scaling\n")

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

    # Equal weight
    live = (df != 0).cummax().astype(float)
    w_eq = live.div(live.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    port = (w_eq * df).sum(axis=1)
    nz = (df != 0).any(axis=1)
    port = port[nz]

    s = stats(port, "NOVA4 equal-wt")
    print(f"\n  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    # Static inverse-vol weights: compute once on first 3 years' vol, freeze forever
    early = df.loc[:dates[len(dates)//4]].dropna()  # first 25% of history
    static_vols = early.std() * np.sqrt(252)
    static_vols = static_vols.replace(0, np.nan).fillna(static_vols.median())
    inv_w = (1.0 / static_vols)
    inv_w = inv_w / inv_w.sum()
    # Apply as constant across all dates, but only to live sleeves
    w_static = pd.DataFrame(np.tile(inv_w.values, (len(dates), 1)),
                            index=dates, columns=df.columns)
    w_static = w_static * live
    w_static = w_static.div(w_static.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    port_static = (w_static * df).sum(axis=1)[nz]
    s = stats(port_static, "NOVA4 static inv-vol")
    print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    native_vol = port.std() * np.sqrt(252)
    for tgt_str, tgt in [("10%", 0.10), ("20%", 0.20)]:
        lev = tgt / native_vol
        r = port * lev
        s = stats(r, f"eq lev {lev:.2f}x → {tgt_str}")
        print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    native_vol_s = port_static.std() * np.sqrt(252)
    for tgt_str, tgt in [("10%", 0.10), ("20%", 0.20)]:
        lev = tgt / native_vol_s
        r = port_static * lev
        s = stats(r, f"stat lev {lev:.2f}x → {tgt_str}")
        print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    IS = pd.Timestamp("2018-01-01")
    for port_name, p_series in [("EqW", port), ("Static", port_static)]:
        for p, lbl in [(p_series.loc[:IS], f"{port_name} IS"), (p_series.loc[IS:], f"{port_name} OOS")]:
            s = stats(p, lbl)
            print(f"  {lbl:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")


if __name__ == "__main__":
    main()
