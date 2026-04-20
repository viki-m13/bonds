"""NOVA2 — ensemble of binary-position sleeves, NO vol scaling.

Design premise: user asked for a 2.0-Sharpe strategy without any vol
scaling. Prior iterations failed (CSMR decimated by daily TC; factor-
momentum timing doesn't work). This version focuses on binary
regime/calendar sleeves that are documented to work historically and
combines them at equal weight with monthly rebal.

Realistic expectation: ~1.0-1.3 unlevered Sharpe. SR 2+ without any
vol/leverage machinery on daily ETF data is almost certainly
unreachable; shipped HYDRA itself needs dynamic vol scaling to clear
1.5. NOVA2 is a clean no-scaling comparison.

Sleeves (all binary long/flat, signal locked at month-start, 1-day
lag before execution, 15 bps TC per regime change):

  s_gem            — Antonacci GEM dual-momentum (SPY/EFA/IEF)
  s_trend_spy      — SPY when above 10m SMA, else IEF (Faber TTM)
  s_trend_qqq      — same for QQQ
  s_trend_eem      — same for EEM
  s_trend_efa      — same for EFA
  s_trend_vnq      — same for VNQ
  s_trend_gld      — GLD when 10m SMA positive, else BIL
  s_trend_tlt      — TLT when 10m SMA positive, else BIL
  s_halloween      — SPY Nov-Apr only
  s_toy_iwm        — IWM first 5 days of January
  s_turn_of_month  — SPY last 3 + first 3 business days
  s_santa          — SPY last 5 of Dec + first 2 of Jan
  s_mid_month      — SPY days 8-12
  s_gold_crisis    — GLD when SPY 3m < 0 and GLD 3m > 0
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
    """Long tic when signal on; else off_tic (or 0). Signal is shifted
    by 1 bar before execution."""
    L = load_etf(tic).reindex(dates).ffill().pct_change().fillna(0)
    sig = signal.reindex(dates).fillna(False).astype(bool).shift(1).fillna(False)
    if off_tic is None:
        r = pd.Series(np.where(sig, L, 0), index=dates)
    else:
        B = load_etf(off_tic).reindex(dates).ffill().pct_change().fillna(0)
        r = pd.Series(np.where(sig, L, B), index=dates)
    changes = sig.astype(int).diff().abs().fillna(0)
    # two-leg switch if off_tic present
    tc = changes * (tc_bps / 1e4) * (2 if off_tic else 1)
    return r - tc


# ---------- Trend-following (Faber 10m SMA) sleeves ----------

def ttm_signal(tic, dates, months=10):
    """True when price > 10-month (~210-day) simple moving average."""
    p = load_etf(tic).reindex(dates).ffill()
    sma = p.rolling(months * 21).mean()
    raw = p > sma
    return _rebal_to_monthly(raw)


def s_trend_spy(dates):
    sig = ttm_signal("SPY", dates)
    return binary_long("SPY", sig, dates, off_tic="IEF").rename("s_trend_spy")


def s_trend_qqq(dates):
    sig = ttm_signal("QQQ", dates)
    return binary_long("QQQ", sig, dates, off_tic="IEF").rename("s_trend_qqq")


def s_trend_eem(dates):
    sig = ttm_signal("EEM", dates)
    return binary_long("EEM", sig, dates, off_tic="IEF").rename("s_trend_eem")


def s_trend_efa(dates):
    sig = ttm_signal("EFA", dates)
    return binary_long("EFA", sig, dates, off_tic="IEF").rename("s_trend_efa")


def s_trend_vnq(dates):
    sig = ttm_signal("VNQ", dates)
    return binary_long("VNQ", sig, dates, off_tic="IEF").rename("s_trend_vnq")


def s_trend_gld(dates):
    sig = ttm_signal("GLD", dates)
    return binary_long("GLD", sig, dates, off_tic="BIL").rename("s_trend_gld")


def s_trend_tlt(dates):
    sig = ttm_signal("TLT", dates)
    return binary_long("TLT", sig, dates, off_tic="BIL").rename("s_trend_tlt")


# ---------- GEM dual momentum ----------

def s_gem(dates):
    """Absolute & relative momentum across SPY/EFA vs IEF, 12m look-back,
    monthly signal, 1-bar lag."""
    spy = load_etf("SPY").reindex(dates).ffill()
    efa = load_etf("EFA").reindex(dates).ffill()
    bil = load_etf("BIL").reindex(dates).ffill()
    ief = load_etf("IEF").reindex(dates).ffill()

    mom_spy = spy.pct_change(252)
    mom_efa = efa.pct_change(252)
    mom_bil = bil.pct_change(252)
    risk_on_raw = mom_spy > mom_bil
    spy_wins_raw = mom_spy > mom_efa
    risk_on = _rebal_to_monthly(risk_on_raw)
    spy_wins = _rebal_to_monthly(spy_wins_raw)

    # Construct three mutually-exclusive signals
    sig_spy = (risk_on & spy_wins).astype(bool)
    sig_efa = (risk_on & ~spy_wins).astype(bool)
    sig_ief = (~risk_on).astype(bool)

    # Shift by 1 bar
    sig_spy = sig_spy.shift(1).fillna(False)
    sig_efa = sig_efa.shift(1).fillna(False)
    sig_ief = sig_ief.shift(1).fillna(False)

    r_spy = spy.pct_change().fillna(0)
    r_efa = efa.pct_change().fillna(0)
    r_ief = ief.pct_change().fillna(0)

    r = (sig_spy.astype(float) * r_spy +
         sig_efa.astype(float) * r_efa +
         sig_ief.astype(float) * r_ief)

    mat = pd.DataFrame({"spy": sig_spy, "efa": sig_efa, "ief": sig_ief}).astype(int)
    changes = mat.diff().abs().sum(axis=1).fillna(0)
    tc = changes * (TC_BPS / 1e4)
    return (r - tc).rename("s_gem")


# ---------- Calendar sleeves ----------

def s_halloween(dates):
    sig = pd.Series([d.month in [11, 12, 1, 2, 3, 4] for d in dates], index=dates)
    return binary_long("SPY", sig, dates).rename("s_halloween")


def s_toy_iwm(dates):
    fwd, _ = _rank_within_month(dates)
    is_jan = np.array([d.month == 1 for d in dates])
    sig = pd.Series(is_jan & (fwd <= 5), index=dates)
    return binary_long("IWM", sig, dates).rename("s_toy")


def s_turn_of_month(dates):
    fwd, rev = _rank_within_month(dates)
    sig = pd.Series((fwd <= 3) | (rev <= 3), index=dates)
    return binary_long("SPY", sig, dates).rename("s_tom")


def s_santa(dates):
    d = pd.DatetimeIndex(dates)
    fwd, rev = _rank_within_month(dates)
    in_dec_end = np.array([(dd.month == 12) for dd in d]) & (rev <= 5)
    in_jan_start = np.array([(dd.month == 1) for dd in d]) & (fwd <= 2)
    sig = pd.Series(in_dec_end | in_jan_start, index=dates)
    return binary_long("SPY", sig, dates).rename("s_santa")


def s_mid_month(dates):
    fwd, _ = _rank_within_month(dates)
    sig = pd.Series((fwd >= 8) & (fwd <= 12), index=dates)
    return binary_long("SPY", sig, dates).rename("s_midmonth")


# ---------- Crisis gold ----------

def s_gold_crisis(dates):
    """Long GLD when SPY 3m negative AND GLD 3m positive (flight-to-safety
    regime). Cash otherwise."""
    gld = load_etf("GLD").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    raw = (gld.pct_change(63) > 0) & (spy.pct_change(63) < 0)
    sig = _rebal_to_monthly(raw)
    return binary_long("GLD", sig, dates).rename("s_gold_crisis")


# ---------- ensemble ----------

SLEEVES = [
    s_trend_spy, s_trend_qqq, s_trend_eem, s_trend_efa, s_trend_vnq,
    s_trend_gld, s_trend_tlt,
    s_gem,
    s_halloween, s_toy_iwm,
    s_gold_crisis,
]


def build(dates):
    out = {fn(dates).name: fn(dates) for fn in SLEEVES}
    return pd.DataFrame(out).reindex(dates).fillna(0)


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()} ({len(dates)/252:.1f}y)")
    print(f"NOVA2 v3 — {len(SLEEVES)} sleeves, equal-weight, no vol scaling\n")

    df = build(dates)
    print("Per-sleeve stats (full history):")
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

    print()
    for r, lbl in [(port, "NOVA2 raw")]:
        s = stats(r, lbl)
        print(f"  {lbl:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    native_vol = port.std() * np.sqrt(252)
    lev_10 = 0.10 / native_vol
    lev_20 = 0.20 / native_vol
    for lev, tgt in [(lev_10, "10%"), (lev_20, "20%")]:
        r = port * lev
        s = stats(r, f"static lev {lev:.2f}x → {tgt}")
        print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    IS = pd.Timestamp("2018-01-01")
    lev_port = port * lev_10
    for p, lbl in [(lev_port.loc[:IS], "IS ≤2018"), (lev_port.loc[IS:], "OOS >2018")]:
        s = stats(p, lbl)
        print(f"  {lbl:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")


if __name__ == "__main__":
    main()
