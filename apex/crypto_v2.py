"""Enhanced crypto sleeve — weighted multi-crypto with dynamic BTC regime.

Key insights:
  1. BTC is the dominant crypto asset (highest liquidity, longest history).
    Weight it more heavily than equal-split.
  2. BTC momentum is the MASTER regime signal for all crypto.
  3. Dynamic weighting: when BTC in strong uptrend, lean heavier.
  4. Add ADA for 4-coin basket when available.

Weights: BTC 50%, ETH 30%, SOL 15%, ADA 5% (when each exists)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import util

ETF = Path("/home/user/bonds/data/etfs")
FRED = Path("/home/user/bonds/data/fred")


def _etf_close(t, idx):
    fp = ETF / f"{t}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Close"].astype(float).reindex(idx).ffill()


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def weighted_crypto_returns(idx: pd.DatetimeIndex, target_vol: float = 0.25,
                              weights: dict = None) -> pd.Series:
    """Weighted multi-crypto with BTC-heavy allocation.

    When BTC momentum is very strong (>20% 63d), lean aggressive.
    When BTC in bear (63d<0), full cash.
    """
    if weights is None:
        weights = {"BTC_USD": 0.50, "ETH_USD": 0.30, "SOL_USD": 0.15, "ADA_USD": 0.05}

    spy = _etf_close("SPY", idx)
    vix = _fred("VIXCLS", idx)
    btc = _etf_close("BTC_USD", idx)

    # Macro gates
    spy_ok = (spy > spy.rolling(200).mean()).astype(float)
    vix_ok = (vix < 30).astype(float).fillna(1.0)
    gate = spy_ok * vix_ok

    # BTC regime (master for all crypto)
    btc_mom63 = btc.pct_change(63)
    btc_on = (btc > btc.rolling(200).mean()).astype(float) * (btc_mom63 > 0).astype(float)

    total_r = pd.Series(0.0, index=idx)
    sum_weights = 0.0
    for coin, w in weights.items():
        p = _etf_close(coin, idx)
        if p.isna().all():
            continue
        # Use coin-specific signal
        coin_mom63 = p.pct_change(63)
        coin_on = (coin_mom63 > 0).astype(float)
        # Combined: coin ON AND BTC ON AND macro gates
        signal = coin_on * btc_on * gate
        signal = signal.shift(1).fillna(0.0)

        r = p.pct_change().fillna(0.0)
        r_weighted = signal * r * w
        # TC drag per coin
        pos_change = signal.diff().abs().fillna(signal.abs())
        r_weighted = r_weighted - pos_change * 0.002 * w
        total_r = total_r + r_weighted
        sum_weights += w

    if sum_weights > 0:
        total_r = total_r / sum_weights

    # Vol scale (down only)
    rv = total_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return total_r * m


def btc_regime_strength(idx: pd.DatetimeIndex) -> pd.Series:
    """0-1 strength of BTC regime. Higher = more aggressive crypto allocation."""
    btc = _etf_close("BTC_USD", idx)
    spy = _etf_close("SPY", idx)
    vix = _fred("VIXCLS", idx)

    btc_mom63 = btc.pct_change(63)
    btc_mom21 = btc.pct_change(21)
    btc_above_200 = (btc > btc.rolling(200).mean()).astype(float)
    spy_above_200 = (spy > spy.rolling(200).mean()).astype(float)
    vix_low = (vix < 25).astype(float).fillna(1.0)

    # Composite strength score 0-1
    strength = pd.Series(0.0, index=idx)
    strength = strength + 0.25 * (btc_mom63 > 0.10).astype(float)
    strength = strength + 0.25 * (btc_mom21 > 0.05).astype(float)
    strength = strength + 0.20 * btc_above_200
    strength = strength + 0.15 * spy_above_200
    strength = strength + 0.15 * vix_low
    return strength


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()

    # Original BTC-only
    import crypto_sleeve as CS
    r_btc_only = CS.crypto_sleeve_returns(cp.index, target_vol=0.20)

    # Multi equal (v15)
    import sleeves_v15 as SV15
    r_multi_eq = SV15.multi_crypto_returns(cp.index, target_vol=0.20)

    # Multi weighted (new)
    r_multi_wt = weighted_crypto_returns(cp.index, target_vol=0.20)

    # BTC-heavy
    r_btc_heavy = weighted_crypto_returns(cp.index, target_vol=0.20,
                                            weights={"BTC_USD": 0.70, "ETH_USD": 0.20, "SOL_USD": 0.10})

    rets = {
        "BTC_only":     r_btc_only,
        "Multi_EQ":     r_multi_eq,
        "Multi_wt":     r_multi_wt,
        "BTC_heavy":    r_btc_heavy,
    }
    print(f"{'Strategy':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'OOS_CAGR':>8}  {'2022':>7}")
    for name, r in rets.items():
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{om.get('cagr',0)*100:>7.1f}%  {m22.get('sharpe',0):>7.2f}")

    r_multi_wt.to_frame("crypto_v2").to_csv("/home/user/bonds/data/apex/crypto_v2_returns.csv")
    r_btc_heavy.to_frame("btc_heavy").to_csv("/home/user/bonds/data/apex/btc_heavy_returns.csv")
