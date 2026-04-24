"""Chronos-T5 foundation-model forecasting sleeve.

Uses Amazon's Chronos-T5 pretrained time-series transformer to forecast
21-day forward return for each LETF's underlying. Signal: long top-3 by
predicted forward return.

Chronos is pretrained on billions of synthetic + real time series,
so it doesn't need additional training on our data — zero-shot forecasting.

If HuggingFace download fails, fallback to no-op sleeve.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import util

UNIVERSE = ["UPRO", "TQQQ", "TECL", "SOXL", "FAS", "EDC", "YINN",
            "TMF", "UBT", "UGL", "UCO", "DRN"]


def _weights_to_ret(W, cp):
    w = W.fillna(0.0)
    rets = cp.pct_change()
    r = (w.shift(1).fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def chronos_forecast_returns(cp: pd.DataFrame, forecast_horizon: int = 21,
                              context_length: int = 252,
                              rebal_every: int = 21,
                              model_name: str = "amazon/chronos-t5-small"):
    """Use Chronos to forecast next-21d return for each LETF.

    Returns wide DataFrame of predictions (Date × Ticker).
    """
    try:
        from chronos import ChronosPipeline
        import torch
    except ImportError as e:
        print(f"  Chronos not available: {e}")
        return None

    print(f"  Loading Chronos model: {model_name}...")
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"  Failed to load: {e}")
        return None

    # For each rebal date, forecast N-day ahead for each ticker
    idx = cp.index
    rebal_dates_mask = pd.Series(range(len(idx)), index=idx) % rebal_every == 0
    rebal_dates = idx[rebal_dates_mask & (idx >= "2010-01-01")]

    preds = pd.DataFrame(np.nan, index=idx, columns=UNIVERSE)

    for d_idx, d in enumerate(rebal_dates):
        if d_idx % 30 == 0:
            print(f"  Chronos forecasting {d.date()} ({d_idx+1}/{len(rebal_dates)})...")
        # For each ticker
        for tic in UNIVERSE:
            if tic not in cp.columns:
                continue
            p = cp[tic]
            # Use prior 252 days as context
            end_loc = p.index.get_loc(d)
            if end_loc < context_length:
                continue
            ctx = p.iloc[end_loc - context_length:end_loc].values
            if np.any(np.isnan(ctx)) or np.any(ctx <= 0):
                continue
            try:
                # Chronos wants a torch tensor
                ctx_tensor = torch.tensor(ctx, dtype=torch.float32)
                forecast = pipeline.predict(
                    inputs=ctx_tensor,
                    prediction_length=forecast_horizon,
                    num_samples=20,
                )
                # forecast shape: [batch, num_samples, prediction_length]
                # Median of the final step
                if forecast.ndim == 3:
                    median_forecast = forecast[0].median(dim=0).values.numpy()
                else:
                    median_forecast = forecast.median(dim=0).values.numpy()
                last_price = ctx[-1]
                predicted_final = median_forecast[-1]
                fwd_ret = (predicted_final - last_price) / last_price
                preds.loc[d, tic] = fwd_ret
            except Exception as e:
                pass

    return preds


def sleeve_chronos(cp: pd.DataFrame, target_vol: float = 0.18,
                   k_top: int = 3, rebal_every: int = 21) -> pd.DataFrame:
    """Chronos-predicted top-K sleeve."""
    preds = chronos_forecast_returns(cp, rebal_every=rebal_every)
    if preds is None or preds.isna().all().all():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)

    preds = preds.ffill()
    rnk = preds.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= k_top) & (preds > 0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for tic in UNIVERSE:
        if tic in cp.columns and tic in sel.columns:
            W[tic] = sel[tic].astype(float) / k_top

    w = W.fillna(0.0)
    r = (w.shift(1).fillna(0.0) * cp.pct_change().fillna(0.0)).sum(axis=1)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/bonds/apex")
    op, cp = util.load_prices()
    # Only test on recent data for speed
    cp_sample = cp.loc["2014-01-01":]
    print("Testing Chronos sleeve on 2014+ data...")
    W = sleeve_chronos(cp_sample)
    if W.sum().sum() > 0:
        r = _weights_to_ret(W, cp_sample)
        util.summarize(r, "CHRONOS FULL")
        util.summarize(util.regime_slice(r, "2019-01-02", "2027-12-31"), "OOS 19+")
        r.to_frame("chronos").to_csv("/home/user/bonds/data/apex/chronos_returns.csv")
        W.to_csv("/home/user/bonds/data/apex/chronos_weights.csv")
