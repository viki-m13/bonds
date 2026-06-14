"""Loader for the hourly crypto panel (data/crypto_hourly/*.csv, binance.us).

Wide hourly frames (open/high/low/close/volume) on a common UTC hourly index,
plus a liquidity/eligibility mask. All downstream signals are causal: a signal
at bar t uses information through the CLOSE of bar t and is traded at the next
bar (shift(1)) — the harness enforces the lag.
"""
import glob
import os

import numpy as np
import pandas as pd

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "crypto_hourly")
ANN = 24 * 365  # hourly bars per year


def load():
    fr = {k: {} for k in ("open", "high", "low", "close", "volume")}
    for f in sorted(glob.glob(os.path.join(DIR, "*.csv"))):
        t = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        d["ts"] = pd.to_datetime(d["ts"], unit="ms")
        d = d[~d["ts"].duplicated()].set_index("ts").sort_index()
        for k in fr:
            fr[k][t] = pd.to_numeric(d[k], errors="coerce")
    P = {k: pd.DataFrame(v).sort_index() for k, v in fr.items()}
    # align to a regular hourly grid (union), forward-fill small gaps in close
    idx = P["close"].index
    for k in P:
        P[k] = P[k].reindex(idx)
    R = P["close"].pct_change()
    R[R.abs() > 1.0] = np.nan          # drop impossible hourly ticks
    P["ret"] = R
    return P


def eligibility(P, min_dollar_vol=2e6, vol_window=24 * 7):
    """Coin tradable at bar t if 7-day avg hourly dollar volume > threshold and
    price history exists."""
    dv = (P["close"] * P["volume"]).rolling(vol_window).mean()
    return P["close"].notna() & P["ret"].notna() & (dv > min_dollar_vol)


if __name__ == "__main__":
    P = load()
    el = eligibility(P)
    C = P["close"]
    print("coins", C.shape[1], "bars", C.shape[0],
          "span", C.index[0], "->", C.index[-1])
    print("eligible coins/bar median", int(el.sum(axis=1).median()))
    print("coins:", ", ".join(C.columns))
