"""NOVA9 — final synthesis attempt, combining best learnings.

After NOVA1-8, honest ceiling without any vol scaling is ~0.8-1.0 SR
on our 21y universe. Leverage and ML both introduced far more risk
than return. This final version wraps the best pieces:

  - 26 NOVA5 trend sleeves (7 assets × 3 lookbacks) + 5 non-trend
    (GEM, halloween, gold-crisis, LQD carry, curve)
  - HRP (hierarchical risk parity) static weights, FIT ON PRE-2015 DATA
    ONLY, frozen forever. This is an SR-weighted allocation that
    respects correlation clustering WITHOUT peeking at OOS vol.
  - Circuit breaker: if VIX > 35 at month-start, override every sleeve
    to cash for that month (binary, not scaling).
  - A single short-equity sleeve (long SH when SPY below 200d AND VIX
    rising) to add a negatively-correlated risk-off component.

Strict walk-forward: HRP weights use ONLY data before 2015-01-01.
Test IS = pre-2018, OOS = 2018+."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from hydra_core import load_etf, load_fred, stats


TC_BPS = 15.0


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


TREND_ASSETS = [
    ("SPY", "IEF"), ("QQQ", "IEF"), ("EEM", "IEF"), ("EFA", "IEF"),
    ("VNQ", "IEF"), ("GLD", "BIL"), ("TLT", "BIL"),
]
LOOKBACKS = [63, 126, 252]


def make_trend(tic, off, lb):
    name = f"trend_{tic}_{lb}"
    def _fn(dates):
        return binary_long(tic, ttm_signal(tic, dates, lb), dates, off_tic=off).rename(name)
    _fn.__name__ = name
    return _fn


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


def s_short_hedge(dates):
    """Long SH when SPY below 200d SMA AND VIX > 63d median.
    Short-equity hedge sleeve to add negatively-correlated component."""
    spy = load_etf("SPY").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    raw = (spy < spy.rolling(200).mean()) & (vix > vix.rolling(63).median())
    sig = _rebal_to_monthly(raw)
    return binary_long("SH", sig, dates).rename("s_short_hedge")


SLEEVES = [make_trend(t, o, lb) for t, o in TREND_ASSETS for lb in LOOKBACKS] + [
    s_gem, s_halloween, s_gold_crisis, s_lqd_carry, s_curve, s_short_hedge,
]


def hrp_weights(cov):
    """Hierarchical risk parity weights from De Prado (2016).
    Returns a pd.Series of weights summing to 1."""
    corr = cov.corr() if hasattr(cov, "corr") else None
    if corr is None:
        # cov is DataFrame of returns, compute corr from cov matrix
        std = np.sqrt(np.diag(cov.values))
        corr = pd.DataFrame(cov.values / np.outer(std, std),
                            index=cov.index, columns=cov.columns)
    dist = np.sqrt(((1 - corr) / 2).clip(lower=0))
    # Collapse symmetric to condensed distance
    cond = squareform(dist.values, checks=False)
    link = linkage(cond, method="single")
    # Get the reordered leaves
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(link)
    ordered = corr.columns[order].tolist()

    # Recursive bisection
    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while any(len(c) > 1 for c in clusters):
        new = []
        for c in clusters:
            if len(c) <= 1:
                new.append(c)
                continue
            split = len(c) // 2
            left = c[:split]
            right = c[split:]
            # Variance of each subcluster using inverse-variance portfolio
            def sub_var(sub):
                C = cov.loc[sub, sub].values
                diag = np.diag(C)
                iv = 1.0 / diag
                iv = iv / iv.sum()
                return iv @ C @ iv
            vl = sub_var(left)
            vr = sub_var(right)
            alpha = 1 - vl / (vl + vr)
            for tic in left:
                w[tic] *= alpha
            for tic in right:
                w[tic] *= 1 - alpha
            new.append(left)
            new.append(right)
        clusters = new

    w = w / w.sum()
    return w


def build(dates):
    out = {}
    for fn in SLEEVES:
        s = fn(dates)
        out[s.name] = s
    return pd.DataFrame(out).reindex(dates).fillna(0)


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()}")
    print(f"NOVA9 — HRP-weighted ensemble, {len(SLEEVES)} sleeves, circuit breaker\n")

    df = build(dates)

    # HRP weights fit on pre-2015 only (strict OOS beyond)
    CUT = pd.Timestamp("2015-01-01")
    R_fit = df.loc[:CUT].dropna(how="all")
    # Only use sleeves that are live in the fit window
    live_cols = [c for c in df.columns if (R_fit[c] != 0).sum() > 200]
    R_fit = R_fit[live_cols].fillna(0)
    cov = R_fit.cov() * 252
    # Add shrinkage
    cov += np.eye(len(cov)) * 1e-6
    w_hrp = hrp_weights(cov)
    print("HRP weights (top 10):")
    for k, v in w_hrp.sort_values(ascending=False).head(10).items():
        print(f"  {k:25s} {v:.4f}")

    # Apply weights to full-sample
    full_R = df[live_cols].fillna(0)
    live_mask = (df[live_cols] != 0).cummax()
    # Renormalise per-row over live sleeves only
    w_series = pd.DataFrame(np.tile(w_hrp.reindex(live_cols).fillna(0).values, (len(dates), 1)),
                            index=dates, columns=live_cols)
    w_series = w_series * live_mask
    row_sums = w_series.sum(axis=1).replace(0, np.nan)
    w_series = w_series.div(row_sums, axis=0).fillna(0)
    port = (full_R * w_series).sum(axis=1)

    # Circuit breaker
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    first = monthly_first_flag(pd.Index(dates))
    breaker = pd.Series(False, index=dates)
    cur = False
    for i, d in enumerate(dates):
        if first.iloc[i] and not pd.isna(vix.iloc[i]):
            cur = vix.iloc[i] > 35
        breaker.iloc[i] = cur
    # On breaker days, override to cash (0 return)
    port_cb = port.where(~breaker, 0)

    nz = (df != 0).any(axis=1)
    port = port[nz]
    port_cb = port_cb[nz]

    for r, lbl in [(port, "NOVA9 HRP"), (port_cb, "NOVA9 HRP + circuit")]:
        s = stats(r, lbl)
        print(f"\n  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
        IS = pd.Timestamp("2018-01-01")
        # strict OOS is post-2018 (HRP fit pre-2015)
        for p, tag in [(r.loc[:CUT], "pre-2015 (HRP-fit)"),
                       (r.loc[CUT:IS], "2015-2018 (buffer)"),
                       (r.loc[IS:], ">2018 strict OOS")]:
            ss = stats(p, tag)
            print(f"    {tag:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Leverage (static) scaling
    for lbl, port_use in [("HRP", port), ("HRP+CB", port_cb)]:
        nv = port_use.std() * np.sqrt(252)
        for tgt_str, tgt in [("10%", 0.10), ("20%", 0.20)]:
            lev = tgt / nv
            s = stats(port_use * lev, f"{lbl} {lev:.2f}x → {tgt_str}")
            print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
                  f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")


if __name__ == "__main__":
    main()
