"""APEX v17 novel sleeves.

  SL_FOMC       — Pre-FOMC announcement drift (Lucca-Moench 2015 JoF).
                  Documented +49bps in SPY over 24h before FOMC. Hold UPRO
                  for ~2 trading days around the meeting.

  SL_PCA_FACTOR — Decompose LETF returns into PCA factors. Trade factor
                  momentum separately. Factors are orthogonal by construction.

  SL_BUY_FEAR   — Buy extreme fear: VIX > 35 AND SPY 20d < -10%. Ultimate
                  contrarian. Historically strongest 30-day forward returns.

  SL_VOLvsVOL   — Vol-of-vol regime: when VIX 5d realized vol > 1.5 × trailing
                  60d, volatility is unstable → go defensive. When vol-of-vol
                  is low, lean risk-on.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import util

ROOT = Path("/home/user/bonds")
FRED = ROOT / "data/fred"
ETF = ROOT / "data/etfs"

# FOMC announcement dates (end-of-meeting 2PM ET release)
FOMC_DATES = [
    "2005-02-02","2005-03-22","2005-05-03","2005-06-30","2005-08-09","2005-09-20","2005-11-01","2005-12-13",
    "2006-01-31","2006-03-28","2006-05-10","2006-06-29","2006-08-08","2006-09-20","2006-10-25","2006-12-12",
    "2007-01-31","2007-03-21","2007-05-09","2007-06-28","2007-08-07","2007-09-18","2007-10-31","2007-12-11",
    "2008-01-30","2008-03-18","2008-04-30","2008-06-25","2008-08-05","2008-09-16","2008-10-29","2008-12-16",
    "2009-01-28","2009-03-18","2009-04-29","2009-06-24","2009-08-12","2009-09-23","2009-11-04","2009-12-16",
    "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10","2010-09-21","2010-11-03","2010-12-14",
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09","2011-09-21","2011-11-02","2011-12-13",
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-07-31","2012-09-13","2012-10-24","2012-12-12",
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31","2013-09-18","2013-10-30","2013-12-18",
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30","2014-09-17","2014-10-29","2014-12-17",
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-15","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
    "2026-01-28","2026-03-18","2026-04-29"
]


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def _weights_to_ret(W, cp):
    w = W.fillna(0.0)
    rets = cp.pct_change()
    r = (w.shift(1).fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def _scale_to_vol(W, cp, target_vol=0.15):
    r = _weights_to_ret(W, cp)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


# ==========================================================================
# FOMC PRE-ANNOUNCEMENT DRIFT (Lucca-Moench 2015 JoF)
# ==========================================================================

def sleeve_fomc(cp: pd.DataFrame, target_vol: float = 0.18) -> pd.DataFrame:
    """Long UPRO/TQQQ over the 2 trading days ending on each FOMC announcement.

    Published effect: SPY gains avg 49 bps in 24h before FOMC 2PM statement.
    Via UPRO (3x), that's ~1.4% per meeting. 8 meetings/yr → +12%/yr pure alpha.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "UPRO" not in cp.columns:
        return W

    fomc_set = set(pd.Timestamp(d) for d in FOMC_DATES)

    # Hold position starting 2 days BEFORE FOMC through the day of FOMC
    # So hold = True on FOMC_day - 2, -1, and 0 (exit at close of FOMC day)
    hold = pd.Series(False, index=idx)
    for d in idx:
        # Check if d is within 2 days before OR on an FOMC day
        for offset in [0, 1, 2]:
            check_date = d + pd.Timedelta(days=offset)
            # Find nearest trading day match
            if check_date in fomc_set:
                hold[d] = True
                break

    # Convert hold to position (trade the next day, held until exit)
    on = hold.astype(float).shift(1).fillna(0.0)

    W["UPRO"] = on * 0.5
    if "TQQQ" in cp.columns:
        W["TQQQ"] = on * 0.3

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# PCA FACTOR ROTATION
# ==========================================================================

def sleeve_pca(cp: pd.DataFrame, target_vol: float = 0.15,
               n_components: int = 3, lookback: int = 252) -> pd.DataFrame:
    """PCA on LETF returns; trade factor momentum.

    Factor 1 ≈ market beta
    Factor 2 ≈ bonds vs equities split
    Factor 3 ≈ commodity/crypto-like
    """
    universe = [a for a in ["UPRO", "TQQQ", "TECL", "SOXL", "FAS", "EDC",
                             "TMF", "UBT", "UGL", "UCO", "DRN"] if a in cp.columns]
    if len(universe) < 6:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    p = cp[universe]
    rets = p.pct_change().fillna(0)

    # Rolling PCA: estimate factor weights on IS window, apply forward
    # For simplicity: fit ONCE on IS (2005-2018), freeze
    from sklearn.decomposition import PCA
    is_rets = rets.loc["2005-01-01":"2018-12-31"].dropna()
    if len(is_rets) < 500:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    pca = PCA(n_components=n_components)
    pca.fit(is_rets.values)
    comps = pca.components_   # shape (k, n)

    # Project all daily returns onto factors
    factor_rets = pd.DataFrame(
        rets.values @ comps.T,
        index=rets.index,
        columns=[f"F{i+1}" for i in range(n_components)]
    )

    # Each factor: trade its momentum
    # Factor momentum: cumulate factor return over 63d, if >0 go long "factor loading"
    f_cumret = (1 + factor_rets).rolling(63).apply(lambda x: x.prod() - 1, raw=True)
    # Signal: +1 if factor up, -1 if down (but we can't short — so 1/0 gate)
    f_signal = (f_cumret > 0).astype(float).shift(1).fillna(0.0)

    # Translate factor signal to LETF weights
    # When factor F_i is on, weight each LETF by its absolute loading on F_i (positive parts)
    # Long only: take max(0, loading)
    W_list = []
    for i in range(n_components):
        loading = comps[i]   # shape (n,)
        # Long-only: positive loadings
        pos_loading = np.maximum(loading, 0)
        if pos_loading.sum() == 0:
            continue
        pos_loading = pos_loading / pos_loading.sum()
        # Allocate
        W_i = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
        for j, u in enumerate(universe):
            W_i[u] = f_signal[f"F{i+1}"] * pos_loading[j] / n_components
        W_list.append(W_i)

    W = sum(W_list) if W_list else pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# BUY THE FEAR — extreme contrarian
# ==========================================================================

def sleeve_buy_fear(cp: pd.DataFrame, target_vol: float = 0.18) -> pd.DataFrame:
    """Buy UPRO when VIX > 35 AND SPY 20d < -8%. Hold 10 days.

    Contrarian play on extreme fear. Historically very strong OOS:
    2008-10 (post-Lehman), 2020-03, 2022-06, 2022-10.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "UPRO" not in cp.columns:
        return W

    spy = cp["SPY"]
    vix = _fred("VIXCLS", idx)

    trig = ((vix > 35) & (spy.pct_change(20) < -0.08)).astype(float)
    # Hold for 10 days
    held = trig.rolling(10, min_periods=1).sum().clip(upper=1.0)

    W["UPRO"] = held.shift(1).fillna(0.0) * 0.60
    if "TQQQ" in cp.columns:
        W["TQQQ"] = held.shift(1).fillna(0.0) * 0.30

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# VOLATILITY OF VOLATILITY regime
# ==========================================================================

def sleeve_vol_of_vol(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """When VIX 5d realized vol > 1.5× 60d VIX realized vol → regime unstable.
    Hold UGL (gold) during unstable periods. Hold SSO during stable low-vol.
    """
    idx = cp.index
    vix = _fred("VIXCLS", idx)
    if vix.isna().all():
        return pd.DataFrame(0.0, index=idx, columns=cp.columns)

    vix_r = vix.pct_change()
    vv_5d = vix_r.rolling(5).std()
    vv_60d = vix_r.rolling(60).std()
    ratio = vv_5d / vv_60d.replace(0, np.nan)

    unstable = (ratio > 1.5).astype(float)
    stable_low = ((ratio < 0.8) & (vix < 18)).astype(float)

    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "UGL" in cp.columns:
        W["UGL"] = unstable.shift(1).fillna(0.0) * 0.5
    if "SSO" in cp.columns:
        W["SSO"] = stable_low.shift(1).fillna(0.0) * 0.5

    return _scale_to_vol(W, cp, target_vol=target_vol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()

    sleeves = {
        "FOMC":       sleeve_fomc(cp),
        "PCA":        sleeve_pca(cp),
        "BUY_FEAR":   sleeve_buy_fear(cp),
        "VOL_OF_VOL": sleeve_vol_of_vol(cp),
    }
    print(f"{'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sleeves.items():
        r = _weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")
