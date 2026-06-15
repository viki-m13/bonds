"""HuggingFace Chronos-Bolt downside-risk signal for entry selection.

State-of-the-art time-series foundation model: Amazon Chronos-Bolt
(`amazon/chronos-bolt-small`, T5-based, pretrained on a large corpus, hosted on
the HuggingFace Hub). Unlike a point forecaster, Chronos-Bolt emits *quantile*
forecasts for every future step, i.e. a full predictive distribution.

That distribution is exactly what the underwater-avoidance objective needs.
An earlier experiment in this repo (`dca/research/results_chronos.md`) used only
the median 42d forecast to rank *expected return* and it failed — median
extrapolation is noise for large-cap direction. Here we use the model
differently and on a different objective: from the quantile forecast over the
next `H` steps we read off, per future step t, the predicted probability that
the price sits BELOW today's price P0 (interpolating where P0 falls among the
forecast quantiles). Averaged over the horizon this is the model's

    predicted underwater fraction  =  mean_t  P( price_t < P0 ).

We pick the names with the LOWEST predicted underwater fraction. This is the
model forecasting the objective itself, not a return it cannot forecast.

Causality: the context for a signal date d is closes strictly through d's
close; selection executes at the next open (handled by the evaluator). Scores
are cached so the (slow) model pass runs once.
"""
import os
import time

import numpy as np
import pandas as pd

from data import load_panel, eligibility
from objective import signal_positions, TRADING_DAYS_MONTH

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "amazon/chronos-bolt-small"
CONTEXT = 256          # trailing trading days fed to the model (~1y)
PRED_LEN = 63          # forecast horizon used for the safety score (~3 months)
QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

PU_PATH = os.path.join(_HERE, "chronos_pu.parquet")     # predicted underwater frac
QM_PATH = os.path.join(_HERE, "chronos_qm.parquet")     # q10 terminal margin


def _pu_qm(q, p0, levels):
    """Vectorized predicted underwater fraction + q10 terminal margin.

    q: (N, S, Q) forecast quantiles (ascending in Q); p0: (N,) entry ref;
    levels: (Q,) quantile levels. Returns (pu, qm) each (N,).
    pu[n] = mean over the S steps of the interpolated CDF level at which p0
    falls among that step's quantiles = predicted mean P(price < p0)."""
    N, S, Q = q.shape
    p0e = p0[:, None, None]
    cnt = (q < p0e).sum(axis=2)                    # (N,S), 0..Q
    lo = np.clip(cnt - 1, 0, Q - 1)
    hi = np.clip(cnt, 0, Q - 1)
    lo_val = np.take_along_axis(q, lo[..., None], axis=2)[..., 0]
    hi_val = np.take_along_axis(q, hi[..., None], axis=2)[..., 0]
    denom = hi_val - lo_val
    frac = np.where(denom > 0, (p0e[..., 0] - lo_val) / np.where(denom > 0,
                                                                 denom, 1.0), 0.0)
    interp = levels[lo] + frac * (levels[hi] - levels[lo])
    interp = np.where(cnt == 0, 0.05, interp)      # p0 below all quantiles
    interp = np.where(cnt == Q, 0.95, interp)      # p0 above all quantiles
    pu = interp.mean(axis=1)
    qm = q[:, -1, 0] / p0 - 1.0
    return pu, qm


def _pipeline():
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(os.cpu_count() or 4)
    return BaseChronosPipeline.from_pretrained(MODEL, device_map="cpu",
                                               dtype=torch.float32)


def generate(start="2010-01-01", end=None, every=TRADING_DAYS_MONTH,
             force=False, batch=512):
    """Run Chronos over monthly signal dates; cache two (days x tickers)
    frames: predicted underwater fraction and q10 terminal margin."""
    if not force and os.path.exists(PU_PATH) and os.path.exists(QM_PATH):
        return {"pu": pd.read_parquet(PU_PATH), "qm": pd.read_parquet(QM_PATH)}

    import torch
    p = load_panel()
    close = p["close"]
    elig = eligibility(min_history=CONTEXT).to_numpy(bool)
    cl = close.to_numpy(float)
    idx, cols = close.index, close.columns
    sig = signal_positions(idx, every, 0, start, end)

    pu = np.full(cl.shape, np.nan)      # predicted underwater fraction
    qm = np.full(cl.shape, np.nan)      # q10 terminal forecast / P0 - 1
    levels = np.array(QLEVELS)

    pipe = _pipeline()
    t_start = time.time()
    for n, sp in enumerate(sig):
        cand = np.where(elig[sp])[0]
        # need a clean, gap-free trailing window
        ctxs, keep = [], []
        for t in cand:
            w = cl[sp - CONTEXT + 1:sp + 1, t]
            if np.isnan(w).any():
                continue
            ctxs.append(torch.tensor(w, dtype=torch.float32))
            keep.append(t)
        if not ctxs:
            continue
        # batched inference
        q_all = []
        for i in range(0, len(ctxs), batch):
            q, _ = pipe.predict_quantiles(ctxs[i:i + batch],
                                          prediction_length=PRED_LEN,
                                          quantile_levels=QLEVELS)
            q_all.append(q.numpy())
        q = np.concatenate(q_all, axis=0)         # (n, PRED_LEN, 9)
        p0 = cl[sp, keep]                          # today's close per kept name
        pu_b, qm_b = _pu_qm(q, p0, levels)
        keep = np.array(keep)
        pu[sp, keep] = pu_b
        qm[sp, keep] = qm_b
        if (n + 1) % 20 == 0 or n + 1 == len(sig):
            el = time.time() - t_start
            print(f"  {n + 1:>3}/{len(sig)} dates  {idx[sp].date()}  "
                  f"{len(keep)} names  {el:5.0f}s elapsed", flush=True)

    pu_df = pd.DataFrame(pu, index=idx, columns=cols)
    qm_df = pd.DataFrame(qm, index=idx, columns=cols)
    pu_df.to_parquet(PU_PATH)
    qm_df.to_parquet(QM_PATH)
    return {"pu": pu_df, "qm": qm_df}


def safety_score() -> np.ndarray:
    """Higher = safer entry. = negative predicted underwater fraction."""
    pu = pd.read_parquet(PU_PATH)
    return (-pu).to_numpy(float)


def q10_margin_score() -> np.ndarray:
    return pd.read_parquet(QM_PATH).to_numpy(float)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = generate(start=a.start, end=a.end, force=a.force)
    print("cached:", PU_PATH)
    print("predicted underwater frac — describe:\n",
          out["pu"].stack().describe().round(4).to_string())
