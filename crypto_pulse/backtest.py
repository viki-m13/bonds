"""Vectorized hourly cross-sectional/time-series backtest engine for crypto.

A signal is a wide frame (bars x coins) known at the CLOSE of bar t. The engine
builds dollar-neutral (or long/flat) weights, lags them one bar (trade at t+1),
applies per-bar returns, and charges turnover-based costs. Sharpe is annualized
with sqrt(24*365). Everything reported gross AND net, IS AND OOS.
"""
import numpy as np
import pandas as pd

import data_hourly as dh

ANN = dh.ANN


def sharpe(p, ann=ANN):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ann) if len(p) > 100 and p.std() > 0 else np.nan


def weights_xs(sig, elig, vol, gross=1.0):
    """Cross-sectional dollar-neutral inverse-vol weights from a signal."""
    s = sig.where(elig)
    z = s.sub(s.mean(axis=1), axis=0)
    w = (z / vol).replace([np.inf, -np.inf], np.nan)
    w = w.div(w.abs().sum(axis=1), axis=0) * gross
    return w


def run(w, ret, cost_bps=5.0, split="2024-06-01", label="", vol_target=None,
        ann=ANN, quiet=False):
    """w: target weights known at bar t; realized at t+1. Returns the net pnl
    series and prints IS/OOS/full Sharpe."""
    wl = w.shift(1)
    turn = (wl - wl.shift(1)).abs().sum(axis=1)
    pnl = (wl * ret).sum(axis=1) - turn * cost_bps / 1e4
    if vol_target is not None:
        scale = (vol_target / (pnl.rolling(24 * 14).std() * np.sqrt(ann))
                 ).shift(1).clip(0, 4)
        pnl = pnl * scale
        turn = turn * scale
    idx = pnl.index
    sp = pd.Timestamp(split)
    warm = idx[24 * 30]
    full = sharpe(pnl[idx >= warm])
    is_ = sharpe(pnl[(idx >= warm) & (idx < sp)])
    oos = sharpe(pnl[idx >= sp])
    ann_ret = pnl[idx >= warm].mean() * ann
    if not quiet:
        print(f"{label:30s} FULL Sh={full:+.2f} IS={is_:+.2f} OOS={oos:+.2f} "
              f"| ann={ann_ret:+.0%} turn/bar={turn.mean():.3f} "
              f"cost={cost_bps:.0f}bps")
    return pnl


def realized_vol(ret, window=24 * 3):
    return ret.rolling(window).std()
