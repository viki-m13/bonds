"""HYDRA — 10-sleeve diversified ensemble targeting SR 3+ via sleeve uncorrelation.

Theoretical basis: if N sleeves each have SR = s and pairwise correlation = rho,
ensemble SR = s * N / sqrt(N + N*(N-1)*rho).
  - N=10, s=1.0, rho=0.0 → SR = 3.16
  - N=10, s=1.0, rho=0.2 → SR = 1.91
  - N=10, s=1.0, rho=0.1 → SR = 2.40
So we need (a) solid SR per sleeve, and (b) very low pairwise correlation.

Each sleeve:
  - Produces a daily return series (long, short, or long-short)
  - Is independently vol-targeted to a common vol (set VOL_TARGET)
  - No leverage at sleeve level beyond vol-target
  - 15 bps transaction cost applied per unit of turnover
  - 1-bar signal lag (signal uses t-1 close, executes at t)

Portfolio: equal-vol contribution (risk parity on sleeves), capped portfolio
gross leverage by GROSS_CAP. Final book vol-targeted to PORT_VOL.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

VOL_TARGET = 0.10        # per-sleeve target annualised vol
VOL_LOOKBACK = 63        # realized-vol window (~3 months)
PORT_VOL = 0.10          # final portfolio target vol
GROSS_CAP = 4.0          # cap on gross leverage (sum of abs weights)
TC_BPS = 15.0            # transaction cost per unit turnover


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def vol_target(ret, target=VOL_TARGET, window=VOL_LOOKBACK, cap=1.5):
    """Scale a daily-return stream to hit target annualised vol.
    Vol is estimated with a rolling lookback, lagged 1 day to avoid look-ahead.
    Uses the MAX of rolling vol and a floor = 0.5*target to prevent sparse-
    signal sleeves from spiking when the signal flips on.
    Scaling factor capped at `cap` to prevent blowups after quiet periods."""
    vol = ret.rolling(window).std().shift(1) * np.sqrt(252)
    vol_floor = target * 0.5
    vol = vol.clip(lower=vol_floor)
    scale = (target / vol).clip(upper=cap).fillna(0)
    return ret * scale, scale


def apply_tc(weight_ts, raw_ret, ret_index):
    """Apply transaction cost based on weight turnover, aligned to returns."""
    tc = weight_ts.diff().abs().sum(axis=1).fillna(0) * (TC_BPS / 1e4)
    return raw_ret - tc.reindex(ret_index).fillna(0)


def stats(r, label=""):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"label": label, "n": len(r), "sharpe": 0, "ret": 0, "vol": 0, "mdd": 0, "navx": 1.0}
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return {"label": label, "n": len(r),
            "sharpe": round(sr, 3),
            "ret": round(ar * 100, 2),
            "vol": round(av * 100, 2),
            "mdd": round(mdd * 100, 2),
            "navx": round(float(cum.iloc[-1]), 2)}
