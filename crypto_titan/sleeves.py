"""CRYPTO-TITAN sleeves — BTC/ETH-centric vol-managed trend.

Four long-only sleeves, each a variant of "vol-managed + trend-gated +
catastrophic-DD-protected". They share the same underlying idea but differ
in speed (fast/slow) and asset (BTC/ETH/alts), which gives useful
decorrelation without introducing new directional risks.

  1. BTC_VM        — vol-managed BTC, fast trend (21d mom, 100MA)
  2. ETH_VM        — vol-managed ETH, fast trend
  3. BTC_SLOW      — slow trend (126d mom, 200MA) — captures multi-quarter regimes
  4. ALT_DIVERS    — gated alt basket (breadth > 60%, per-coin cap 10%)

A CONSENSUS overlay is computed in strategy.py that scales the total long
exposure up when 3+ signals agree, down when only 1 does.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from util import DPY, eligibility


def _vol_managed_core(cp: pd.DataFrame, coin: str,
                       vol_target: float = 0.22,
                       ma_len: int = 100, mom_len: int = 63,
                       dd_cut: float = -0.28) -> pd.Series:
    if coin not in cp.columns:
        return pd.Series(0.0, index=cp.index)
    s = cp[coin]
    rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(DPY)
    size = (vol_target / rv.replace(0, np.nan)).clip(lower=0.0, upper=1.5)
    ma = s.rolling(ma_len, min_periods=ma_len // 2).mean()
    trend = ((s > ma) & (s.pct_change(mom_len) > 0.0)).astype(float)
    hwm = s.rolling(90, min_periods=30).max()
    dd = s / hwm - 1
    alive = (dd > dd_cut).astype(float)
    return (size * trend * alive).fillna(0.0).shift(1).fillna(0.0)


def sleeve_btc_vm(cp):
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W["BTC"] = _vol_managed_core(cp, "BTC", vol_target=0.22,
                                  ma_len=100, mom_len=63, dd_cut=-0.28)
    return W


def sleeve_eth_vm(cp):
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "ETH" in cp.columns:
        W["ETH"] = _vol_managed_core(cp, "ETH", vol_target=0.22,
                                      ma_len=100, mom_len=63, dd_cut=-0.28)
    return W


def sleeve_btc_slow(cp):
    """Slow trend: 200MA + 126d mom > 20%. Wider stop (DD -35%).
    Decorrelates with BTC_VM because it trades less."""
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    s = cp["BTC"]
    rv = s.pct_change().rolling(60, min_periods=30).std() * np.sqrt(DPY)
    size = (0.20 / rv.replace(0, np.nan)).clip(lower=0.2, upper=1.2)
    ma200 = s.rolling(200, min_periods=100).mean()
    trend = ((s > ma200) & (s.pct_change(126) > 0.20)).astype(float)
    hwm = s.rolling(180, min_periods=60).max()
    dd = s / hwm - 1
    alive = (dd > -0.35).astype(float)
    W["BTC"] = (size * trend * alive).shift(1).fillna(0.0)
    return W


def sleeve_alt_divers(cp):
    """Breadth-gated top-3 alt basket, per-coin cap 10%, basket cap 30%."""
    alt_cols = [c for c in cp.columns if c not in ["BTC", "ETH"]]
    if not alt_cols:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    cp_alt = cp[alt_cols]
    elig = eligibility(cp_alt, 180, catastrophe_dd=-0.25, dd_window=60)
    ma100 = cp_alt.rolling(100, min_periods=50).mean()
    mom63 = cp_alt.pct_change(63)
    trending = ((cp_alt > ma100) & (mom63 > 0.15)).astype(float) * elig
    breadth = trending.sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
    breadth_ok = (breadth > 0.60).astype(float).fillna(0.0)

    score = mom63.where(trending.astype(bool))
    ranks = score.rank(axis=1, ascending=False, method="first")
    pick = (ranks <= 3.0).astype(float)
    rv = cp_alt.pct_change().rolling(60, min_periods=20).std() * np.sqrt(DPY)
    inv = (1.0 / rv.replace(0, np.nan)).where(pick.astype(bool))
    W_alt = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W_alt = W_alt.clip(upper=0.10)
    s_sum = W_alt.sum(axis=1).replace(0, np.nan)
    scale = (0.30 / s_sum).clip(upper=1.0).fillna(0.0)
    W_alt = W_alt.mul(scale, axis=0).mul(breadth_ok, axis=0)

    btc = cp["BTC"]
    btc_gate = ((btc > btc.rolling(150, min_periods=75).mean()) &
                (btc.pct_change(63) > 0.0)).astype(float)
    W_alt = W_alt.mul(btc_gate, axis=0)

    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for c in W_alt.columns:
        W[c] = W_alt[c]
    return W.shift(1).fillna(0.0)


BUILDERS = {
    "BTC_VM":     sleeve_btc_vm,
    "ETH_VM":     sleeve_eth_vm,
    "BTC_SLOW":   sleeve_btc_slow,
    "ALT_DIVERS": sleeve_alt_divers,
}


def build_all(cp):
    out = {}
    for name, fn in BUILDERS.items():
        W = fn(cp)
        out[name] = W.fillna(0.0).reindex(columns=cp.columns, fill_value=0.0)
    return out


def consensus_signal(cp: pd.DataFrame, sleeves: dict) -> pd.Series:
    """Sleeve-agreement ratio in [0,1]: what fraction of sleeves are long
    TODAY. Used as a multiplicative scale on total exposure (high conviction
    when multiple independent signals concur)."""
    votes = []
    for name, W in sleeves.items():
        votes.append((W.sum(axis=1) > 1e-4).astype(float))
    vote_df = pd.concat(votes, axis=1)
    ratio = vote_df.mean(axis=1).fillna(0.0)
    return ratio
