"""NOVA5 — Lookback-diversified trend ensemble, binary, no vol scaling.

Observation from NOVA2/3/4: single-SMA trend sleeves all hit SR ~0.5-0.7
but are 25% correlated to each other, ceiling ensemble SR ~0.8.
Lookback diversification (short trend, medium trend, long trend) is a
well-documented way to reduce inter-sleeve correlation while keeping
average SR.

Design:
  - 7 assets × 3 lookbacks (~3m/6m/12m SMA) = 21 trend sleeves
  - Plus GEM, halloween, gold-crisis, LQD-carry, curve
  - Equal-weight live sleeves, monthly rebal, binary long/flat
  - No vol scaling, no leverage targeting

Hypothesis: 26 sleeves with avg SR ~0.5 and correlation maybe 0.15
gives ensemble SR = 0.5 * sqrt(26) / sqrt(1 + 25*0.15) = 2.55/2.18 = 1.17.
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


def ttm_signal(tic, dates, days):
    p = load_etf(tic).reindex(dates).ffill()
    sma = p.rolling(days).mean()
    return _rebal_to_monthly(p > sma)


# Assets and their off-asset
TREND_ASSETS = [
    ("SPY", "IEF"),
    ("QQQ", "IEF"),
    ("EEM", "IEF"),
    ("EFA", "IEF"),
    ("VNQ", "IEF"),
    ("GLD", "BIL"),
    ("TLT", "BIL"),
]
LOOKBACKS = [63, 126, 252]  # ~3m, 6m, 12m in trading days


def make_trend_sleeve(tic, off, lb):
    def _fn(dates):
        sig = ttm_signal(tic, dates, lb)
        return binary_long(tic, sig, dates, off_tic=off).rename(f"trend_{tic}_{lb}")
    _fn.__name__ = f"trend_{tic}_{lb}"
    return _fn


TREND_SLEEVES = [make_trend_sleeve(t, o, lb) for t, o in TREND_ASSETS for lb in LOOKBACKS]


# ---------- non-trend sleeves ----------

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


def s_halloween(dates):
    sig = pd.Series([d.month in [11, 12, 1, 2, 3, 4] for d in dates], index=dates)
    return binary_long("SPY", sig, dates).rename("s_halloween")


def s_gold_crisis(dates):
    gld = load_etf("GLD").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    raw = (gld.pct_change(63) > 0) & (spy.pct_change(63) < 0)
    sig = _rebal_to_monthly(raw)
    return binary_long("GLD", sig, dates).rename("s_gold_crisis")


def s_lqd_carry(dates):
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    raw = hy < hy.rolling(63).mean()
    sig = _rebal_to_monthly(raw)
    return binary_long("LQD", sig, dates).rename("s_lqd_carry")


def s_curve(dates):
    curve = load_fred("T10Y2Y").reindex(dates).ffill()
    raw = (curve > 0) & (curve > curve.rolling(63).mean())
    sig = _rebal_to_monthly(raw)
    return binary_long("TLT", sig, dates, off_tic="SHY").rename("s_curve")


SLEEVES = TREND_SLEEVES + [s_gem, s_halloween, s_gold_crisis, s_lqd_carry, s_curve]


def build(dates):
    out = {}
    for fn in SLEEVES:
        s = fn(dates)
        out[s.name] = s
    return pd.DataFrame(out).reindex(dates).fillna(0)


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()} ({len(dates)/252:.1f}y)")
    print(f"NOVA5 — {len(SLEEVES)} sleeves, equal-weight, no vol scaling\n")

    df = build(dates)
    print("Per-sleeve stats (full history):")
    for c in df.columns:
        s = stats(df[c], c)
        print(f"  {c:22s} SR={s['sharpe']:>5.2f}  Vol={s['vol']:>5.2f}%  "
              f"MDD={s['mdd']:>7.2f}%")

    valid = (df != 0).sum(axis=1) >= 10
    corr = df[valid].corr()
    tri = corr.values[np.triu_indices_from(corr, k=1)]
    print(f"\nMean |pairwise corr| = {np.mean(np.abs(tri)):.3f}   "
          f"Median = {np.median(np.abs(tri)):.3f}   Max = {np.max(np.abs(tri)):.2f}")

    live = (df != 0).cummax().astype(float)
    w = live.div(live.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    port = (w * df).sum(axis=1)
    nz = (df != 0).any(axis=1)
    port = port[nz]

    s = stats(port, "NOVA5 equal-wt")
    print(f"\n  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    native_vol = port.std() * np.sqrt(252)
    for tgt_str, tgt in [("10%", 0.10), ("20%", 0.20)]:
        lev = tgt / native_vol
        r = port * lev
        s = stats(r, f"static {lev:.2f}x → {tgt_str}")
        print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    IS = pd.Timestamp("2018-01-01")
    lev_port = port * (0.10 / native_vol)
    for p, lbl in [(lev_port.loc[:IS], "IS"), (lev_port.loc[IS:], "OOS")]:
        s = stats(p, lbl)
        print(f"  {lbl:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")

    # --- CEILING CHECK: in-sample Markowitz-optimal static weights ---
    # Use ONLY pre-2018 data to fit weights, apply them full-sample.
    # Constrained: weights >= 0 (long-only), sum = 1. Plus tangency (max SR).
    print("\n=== Ceiling check: MVO-optimal static weights (IS-fit, OOS-tested) ===")
    df_is = df.loc[:IS]
    live_is = (df_is != 0).cummax().iloc[-1] > 0
    live_cols = df_is.columns[live_is]
    R = df_is[live_cols].fillna(0)
    mu = R.mean() * 252
    cov = R.cov() * 252
    # Add small shrinkage to cov for stability
    cov_reg = cov + np.eye(len(cov)) * 1e-4

    # Unconstrained tangency weights (may be negative or huge). Normalise.
    inv_cov = np.linalg.pinv(cov_reg.values)
    w_tan = inv_cov @ mu.values
    w_tan = pd.Series(w_tan, index=live_cols)
    # Cap gross: renormalise so sum(|w|) = 1
    w_tan_n = w_tan / np.abs(w_tan).sum()

    # Apply to full-sample returns
    full_R = df[live_cols].fillna(0)
    port_mvo = (full_R * w_tan_n).sum(axis=1)[nz]
    s = stats(port_mvo, "MVO tangency (IS fit)")
    print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
    # IS vs OOS of MVO
    for p, lbl in [(port_mvo.loc[:IS], "MVO IS"), (port_mvo.loc[IS:], "MVO OOS")]:
        s = stats(p, lbl)
        print(f"  {lbl:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")

    # Risk-parity (inverse-vol) static weights from IS
    vols_is = R.std() * np.sqrt(252)
    inv_vol = 1.0 / vols_is.replace(0, np.nan).fillna(vols_is.median())
    w_rp = inv_vol / inv_vol.sum()
    port_rp = (full_R * w_rp).sum(axis=1)[nz]
    s = stats(port_rp, "Risk-parity static (IS vols)")
    print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
    for p, lbl in [(port_rp.loc[:IS], "RP IS"), (port_rp.loc[IS:], "RP OOS")]:
        s = stats(p, lbl)
        print(f"  {lbl:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")


if __name__ == "__main__":
    main()
