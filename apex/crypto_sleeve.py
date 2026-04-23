"""Crypto sleeve that actually trades BTC (via BTC_USD price directly).

Since BTC isn't in the LETF universe, we make the sleeve emit returns directly
rather than weights. This means it's an "external" sleeve — we add its
returns to the portfolio at a fixed weight (rather than passing weights to
a LETF trader).

Trade rule (Phoenix-style):
  Long BTC when 63d mom > 0 AND SPY > 200d MA AND VIX < 30.
  Else flat (0% return, 0% vol).
  Target sleeve vol 25% ann (scaled down).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import util

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"


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


def crypto_sleeve_returns(cp_index: pd.DatetimeIndex,
                           target_vol: float = 0.25) -> pd.Series:
    """Return the daily return series of the crypto sleeve (not weights)."""
    btc = _etf_close("BTC_USD", cp_index)
    spy = _etf_close("SPY", cp_index)
    vix = _fred("VIXCLS", cp_index)

    # Signal: 63d mom > 0 AND SPY > 200MA AND VIX < 30
    btc_mom63 = btc.pct_change(63)
    spy_ok = (spy > spy.rolling(200).mean()).astype(float)
    vix_ok = (vix < 30).astype(float).fillna(1.0)
    on = (btc_mom63 > 0).astype(float) * spy_ok * vix_ok
    on = on.shift(1).fillna(0)  # execute at next day

    btc_ret = btc.pct_change().fillna(0)
    r = on * btc_ret
    # TC drag 20 bps per trade
    pos_change = on.diff().abs().fillna(on.abs())
    r = r - pos_change * 0.002

    # Scale to target vol (down only)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return r * m


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/user/bonds/apex')
    op, cp = util.load_prices()
    r = crypto_sleeve_returns(cp.index)
    util.summarize(r, "CRYPTO FULL")
    util.summarize(util.regime_slice(r, "2015-01-01", "2018-12-31"), "IS 2015-18")
    util.summarize(util.regime_slice(r, "2019-01-02", "2027-12-31"), "OOS 19+")
    util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "2022")
    r.to_frame("crypto").to_csv("/home/user/bonds/data/apex/crypto_sleeve_returns.csv")
