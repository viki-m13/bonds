"""APEX v19 sleeves — additional price-action & technical.

  SL_GAP_FADE      — Fade overnight gaps (close-to-open) in opposite direction.
  SL_PIVOT_TREND   — Trade when price breaks weekly pivot levels.
  SL_RANGE_BREAK   — Bollinger-band breakout (20d mean ± 2σ).
  SL_VOLZ_MOM      — Volatility-z-score momentum: vol rank * momentum rank.
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


# ==========================================================================
# RANGE BREAKOUT — Bollinger Band upper breakout
# ==========================================================================

def sleeve_range_break(cp: pd.DataFrame, target_vol: float = 0.15,
                        lookback: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Buy LETF when its underlying breaks ABOVE its 20d + 2σ upper band
    (true breakout). Hold 10 days.
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    pairs = {"SPY": "UPRO", "QQQ": "TQQQ", "TLT": "TMF", "GLD": "UGL"}

    for under, letf in pairs.items():
        if under not in cp.columns or letf not in cp.columns:
            continue
        p = cp[under]
        ma = p.rolling(lookback).mean()
        sd = p.rolling(lookback).std()
        upper = ma + k * sd
        # Breakout signal
        broke = (p > upper).astype(float)
        held = broke.rolling(10, min_periods=1).sum().clip(upper=1.0)
        # Market trend filter
        ma200 = p.rolling(200).mean()
        trend_ok = (p > ma200).astype(float)
        W[letf] = (held * trend_ok).shift(1).fillna(0.0) * 0.25

    s = W.sum(axis=1).clip(upper=1.0)
    scale = (s / W.sum(axis=1).replace(0, np.nan)).fillna(1.0).clip(upper=1.0)
    W = W.mul(scale, axis=0)
    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# VOL-Z MOMENTUM — momentum weighted by vol stability
# ==========================================================================

def sleeve_volz_mom(cp: pd.DataFrame, target_vol: float = 0.15,
                     k_top: int = 3, rebal: int = 21) -> pd.DataFrame:
    """Score = 126d-momentum rank × (1 - 60d-vol rank). High-mom + LOW-vol wins.
    Favors stable uptrends, avoids volatile reversals.
    """
    universe = [a for a in ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","YINN",
                             "TMF","UBT","UGL","UCO","DRN"] if a in cp.columns]
    p = cp[universe]
    mom126 = p.shift(21).pct_change(105)   # skip 21
    vol60 = p.pct_change().rolling(60).std()
    mom_rank = mom126.rank(axis=1, pct=True)
    lowvol_rank = 1 - vol60.rank(axis=1, pct=True)
    score = mom_rank * lowvol_rank
    # Top-K
    rnk = score.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= k_top) & (mom126 > 0)

    # Monthly rebal
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % rebal == 0
    sel_m = sel.where(is_rebal).ffill().fillna(False)

    # Market filter
    spy_ok = (cp["SPY"] > cp["SPY"].rolling(200).mean()).astype(float)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for u in universe:
        W[u] = (sel_m[u].astype(float) / k_top * spy_ok)

    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# ABSOLUTE-DRAWDOWN TRIGGER — long when at max drawdown (contrarian)
# ==========================================================================

def sleeve_dd_contrarian(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """When an LETF is at >30% drawdown AND market has recovered 5% in 10d,
    buy it (mean-reversion on deep drawdowns).
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    spy = cp["SPY"]
    spy_recovery = (spy.pct_change(10) > 0.05).astype(float)

    # Market not in crash mode
    vix = _fred("VIXCLS", idx)
    vix_ok = (vix < 40).astype(float).fillna(1.0)

    for tic in ["UPRO", "TQQQ", "TECL", "SOXL"]:
        if tic not in cp.columns:
            continue
        p = cp[tic]
        hwm = p.rolling(252, min_periods=30).max()
        dd = p / hwm - 1
        deep_dd = (dd < -0.30).astype(float)
        # Signal: was in >30% drawdown AND market recovering AND vix OK
        sig = deep_dd * spy_recovery * vix_ok
        held = sig.rolling(15, min_periods=1).sum().clip(upper=1.0)
        W[tic] = held.shift(1).fillna(0.0) * 0.25

    s = W.sum(axis=1).clip(upper=1.0)
    scale = (s / W.sum(axis=1).replace(0, np.nan)).fillna(1.0).clip(upper=1.0)
    W = W.mul(scale, axis=0)
    return _scale_to_vol(W, cp, target_vol=target_vol)


# ==========================================================================
# VOL-OF-VOL EXPANSION — when VIX starts expanding rapidly, defensive
# ==========================================================================

def sleeve_vix_expand(cp: pd.DataFrame, target_vol: float = 0.15) -> pd.DataFrame:
    """When VIX expanding rapidly (5d change > 30%), LONG UGL/TMF.
    When VIX contracting rapidly (5d change < -20%), LONG UPRO (volatility crush).
    """
    idx = cp.index
    W = pd.DataFrame(0.0, index=idx, columns=cp.columns)

    vix = _fred("VIXCLS", idx)
    vix_5d_chg = vix.pct_change(5)

    expanding = (vix_5d_chg > 0.30).astype(float).shift(1).fillna(0)
    contracting = (vix_5d_chg < -0.20).astype(float).shift(1).fillna(0)

    if "UGL" in cp.columns:
        W["UGL"] = expanding * 0.4
    if "TMF" in cp.columns:
        W["TMF"] = expanding * 0.3
    if "UPRO" in cp.columns:
        W["UPRO"] = contracting * 0.6

    return _scale_to_vol(W, cp, target_vol=target_vol)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()

    sleeves = {
        "RANGE_BREAK":    sleeve_range_break(cp),
        "VOLZ_MOM":       sleeve_volz_mom(cp),
        "DD_CONTRARIAN":  sleeve_dd_contrarian(cp),
        "VIX_EXPAND":     sleeve_vix_expand(cp),
    }
    print(f"{'Sleeve':18s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sleeves.items():
        r = _weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:18s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")
