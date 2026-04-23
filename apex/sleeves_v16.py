"""APEX v16 novel sleeves.

  SL_CRASH_CONTRARIAN — Buy UPRO/TQQQ after severe 5-10d drops (>2σ)
    when NOT in confirmed bear (SPY still above 300d MA). Historically
    3-5d recoveries after extreme drops are strong. V-shaped recovery trade.

  SL_CROSS_DECORR — When 20d correlation among LETF universe breaks down
    (determinant of 63d correlation matrix drops), rotate defensively.

  SL_REAL_YIELD — Long UGL when DGS10 - DGS10 shift(60) < 0 (real yields
    falling) AND CPI YoY elevated. Long TMF when DGS10 falling AND CPI low.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import util

ROOT = Path("/home/user/bonds")
FRED = ROOT / "data/fred"
ETF = ROOT / "data/etfs"


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


def sleeve_crash_contrarian(cp: pd.DataFrame, target_vol: float = 0.18) -> pd.DataFrame:
    """Buy equity LETF after severe 5d drop when underlying still >300d MA.

    Rule:
      - SPY 5d return < -6% AND SPY > 300d MA (not in deep bear)
      - VIX > 25 (fear spike)
      - Buy UPRO for 5-day hold
      - Exit early if SPY drops another 3% (add fuel-not-trap logic)
    Historically: crashes in bull markets (2018 Q4, 2020 Mar, 2022 summer)
    recovered within 5 days on average.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if "UPRO" not in cp.columns:
        return W

    spy = cp["SPY"]
    r5 = spy.pct_change(5)
    ma300 = spy.rolling(300).mean()
    vix = _fred("VIXCLS", idx)

    # Trigger: SPY 5d <-6% AND SPY>300MA AND VIX>25
    trig = ((r5 < -0.06) & (spy > ma300) & (vix > 25)).astype(float)
    # Hold 5 days
    held = trig.rolling(5, min_periods=1).sum().clip(upper=1.0)
    # Cut if SPY continues dropping
    spy_r1 = spy.pct_change(1)
    continuing_drop = (spy_r1 < -0.02).astype(float)
    # When continuing_drop fires, zero out held position for next day
    held_safe = held * (1 - continuing_drop.shift(1).fillna(0))
    held_safe = held_safe.clip(lower=0, upper=1.0)

    W["UPRO"] = held_safe.shift(1).fillna(0.0) * 0.50
    # Also TQQQ
    if "TQQQ" in cp.columns:
        W["TQQQ"] = held_safe.shift(1).fillna(0.0) * 0.30

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_real_yield(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """Macro sleeve:
    - Long UGL when real rates falling AND CPI elevated (stagflation gold)
    - Long TMF when real rates falling AND CPI benign (disinflation bond rally)
    - Long UPRO when real rates stable AND curve positive (goldilocks)
    - Cash otherwise
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    dgs10 = _fred("DGS10", idx)
    cpi = _fred("CPIAUCSL", idx)
    curve = _fred("T10Y2Y", idx)

    if dgs10.isna().all() or cpi.isna().all():
        return W

    # Real yield proxy: 10Y - CPI YoY
    cpi_yoy = cpi.pct_change(252) * 100
    real_yield = dgs10 - cpi_yoy
    real_yield_falling = real_yield.diff(60) < -0.30  # dropping 30bp in 60d

    # Regimes
    cpi_hot = cpi_yoy > 4.0
    cpi_benign = cpi_yoy < 3.0
    curve_positive = curve > 0
    curve_ok = curve > -0.3

    # 3 regimes:
    stagflation_like = real_yield_falling & cpi_hot
    disinflation_like = real_yield_falling & cpi_benign
    goldilocks = curve_positive & ~real_yield_falling & ~cpi_hot

    if "UGL" in cp.columns:
        W["UGL"] = stagflation_like.astype(float).shift(1).fillna(0) * 0.5
    if "TMF" in cp.columns:
        W["TMF"] = disinflation_like.astype(float).shift(1).fillna(0) * 0.5
    if "UPRO" in cp.columns:
        W["UPRO"] = goldilocks.astype(float).shift(1).fillna(0) * 0.5

    return _scale_to_vol(W, cp, target_vol=target_vol)


def sleeve_cross_decorr(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """When cross-LETF 63d correlation matrix determinant drops (decorrelation),
    market is transitioning — rotate to top-momentum single best LETF.
    When high correlation (risk-on regime), use equal-weight across top-3.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","TMF","UGL","UCO"]
                if a in cp.columns]
    if len(universe) < 4:
        return W
    p = cp[universe]

    # 63d rolling correlation determinant
    r_daily = p.pct_change()
    det_series = pd.Series(np.nan, index=idx)
    for i in range(63, len(idx), 5):   # compute every 5 days (efficiency)
        try:
            corr = r_daily.iloc[i-63:i].corr().values
            det_series.iloc[i] = np.linalg.det(corr)
        except Exception:
            pass
    det_series = det_series.ffill()

    # Low determinant = high decorrelation = regime change
    det_low = (det_series < det_series.rolling(252, min_periods=60).quantile(0.20))

    # 126d momentum, rank
    mom126 = p.pct_change(126)
    rnk = mom126.rank(axis=1, ascending=False, method="first")
    # When det_low: top-1 concentrated. Otherwise top-3 equal.
    sel_top1 = (rnk <= 1)
    sel_top3 = (rnk <= 3)

    # Weekly rebal
    is_rebal = pd.Series(idx.weekday, index=idx) == 4

    sel_effective = pd.DataFrame(0.0, index=idx, columns=universe)
    for u in universe:
        sel_effective[u] = (
            det_low.astype(float) * sel_top1[u].astype(float)
            + (1 - det_low.astype(float)) * sel_top3[u].astype(float) / 3.0
        )

    sel_eff_wk = sel_effective.where(is_rebal, axis=0).ffill().fillna(0.0)

    # Market filter
    spy = cp["SPY"]
    spy_ok = (spy > spy.rolling(200).mean()).astype(float)

    for u in universe:
        W[u] = sel_eff_wk[u] * spy_ok

    return _scale_to_vol(W, cp, target_vol=target_vol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()

    sleeves = {
        "CRASH_CONTRARIAN": sleeve_crash_contrarian(cp),
        "REAL_YIELD": sleeve_real_yield(cp),
        "CROSS_DECORR": sleeve_cross_decorr(cp),
    }
    print(f"{'Sleeve':20s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sleeves.items():
        r = _weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:20s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")
