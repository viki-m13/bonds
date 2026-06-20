"""Chronos-bolt re-ranking experiment for the biweekly DCA picker.

Question (scoped): does a pretrained time-series foundation model add
stock-selection value over plain momentum, holding the candidate set fixed?

Design
------
* Signal dates: every 21 trading days from 2016-01 to end of panel.
* Candidate set at each signal date: top 30 members by 189d-return-skip-21d
  momentum (close.shift(21)/close.shift(189) - 1), members only, >=252d
  history (mirrors the engine's eligibility mask).
* Chronos input: last 512 daily LOG PRICES (gaps ffilled, leading NaNs
  dropped, >=256 points required).  Log prices rather than log returns:
  daily returns are near-white-noise and a pretrained forecaster collapses
  to a ~zero-drift forecast, making the score pure noise; log prices keep
  the trend/level structure the model can actually extrapolate, and
  chronos-bolt's internal affine (instance) normalization removes the level
  so series of different price scales are comparable.  The cumulative
  return score is then just a difference of log levels.
* Forecast: amazon/chronos-bolt-small, horizon 42 trading days, batched
  per signal date (30 series / call).
  score = q50_forecast[h=42] - last observed log price
        = median forecasted cumulative 42d log return.
* Control: same dates, same 30 candidates, score = the momentum value
  itself.  Both score matrices are forward-filled (limit 20 days) so every
  window's buy grid picks from the most recent monthly scoring -- ffill of
  past information is causal, and the treatment is identical in both arms.

Causality: all inputs strictly trailing (close through day d).  Caveat:
chronos-bolt was pretrained on public data whose corpus may overlap our
2016-2026 evaluation period -- a positive result would still need
out-of-corpus confirmation.

Cache: research/chronos_scores.parquet (long format: date, ticker, mom,
chronos score), appended every CHECKPOINT_EVERY dates so partial progress
survives interruption.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import data as data_mod  # noqa: E402

CACHE_PQ = os.path.join(_HERE, "chronos_scores.parquet")

MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT = 512          # daily log prices fed to the model
MIN_CONTEXT = 256      # skip candidates with shorter usable history
HORIZON = 42           # forecast horizon, trading days
SIG_EVERY = 21         # signal-date spacing, trading days
START = "2016-01-01"
TOP_N = 30             # candidate-set size (by momentum)
MOM_LB, MOM_SKIP = 189, 21
CHECKPOINT_EVERY = 10  # signal dates between parquet checkpoints


def signal_positions(index):
    p0 = int(index.searchsorted(pd.Timestamp(START)))
    return np.arange(p0, len(index), SIG_EVERY)


def eligibility(P):
    """Engine-mirroring eligibility: member & >=252d history & close ok."""
    c = P["close"]
    enough = c.notna().rolling(252).count() >= 252
    return P["member"] & enough & c.notna()


def build_cache(force=False, verbose=True):
    """Run all chronos forecasts; return long DataFrame
    (date, ticker, mom, chronos)."""
    P = data_mod.build_panel()
    c = P["close"]
    idx, cols = c.index, c.columns
    logp = np.log(c.to_numpy(float))
    mom = (c.shift(MOM_SKIP) / c.shift(MOM_LB) - 1.0).to_numpy(float)
    elig = eligibility(P).to_numpy(bool)

    pos_all = signal_positions(idx)

    done = set()
    parts = []
    if not force and os.path.exists(CACHE_PQ):
        prev = pd.read_parquet(CACHE_PQ)
        done = set(pd.to_datetime(prev["date"]).unique())
        parts.append(prev)
        if verbose:
            print(f"cache: {len(done)} signal dates already done", flush=True)
    todo = [p for p in pos_all if idx[p] not in done]
    if not todo:
        return pd.concat(parts, ignore_index=True)

    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(4)
    pipe = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float32)

    t0 = time.time()
    rows = []
    n_fc = 0
    for i, p in enumerate(todo):
        m = mom[p].copy()
        m[~elig[p]] = np.nan
        ok = ~np.isnan(m)
        if ok.sum() < TOP_N:
            continue
        cand = np.argsort(-np.where(ok, m, -np.inf))[:TOP_N]

        ctx, kept = [], []
        for t in cand:
            s = pd.Series(logp[max(0, p - CONTEXT + 1):p + 1, t]).ffill()
            s = s.dropna()
            if len(s) < MIN_CONTEXT or not np.isfinite(s.iloc[-1]):
                continue
            ctx.append(torch.tensor(s.to_numpy(np.float32)))
            kept.append(t)
        if not kept:
            continue
        q, _ = pipe.predict_quantiles(
            ctx, prediction_length=HORIZON, quantile_levels=[0.5])
        # q: (batch, horizon, 1) -> median forecast path
        q50_end = q[:, -1, 0].numpy()
        n_fc += len(kept)
        for j, t in enumerate(kept):
            last = ctx[j][-1].item()
            rows.append({"date": idx[p], "ticker": cols[t],
                         "mom": float(m[t]),
                         "chronos": float(q50_end[j] - last)})
        if verbose and (i % 10 == 0 or i == len(todo) - 1):
            el = time.time() - t0
            print(f"{i + 1}/{len(todo)} dates ({idx[p].date()}), "
                  f"{n_fc} forecasts, {el:.0f}s "
                  f"({el / (i + 1):.1f}s/date)", flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(todo) - 1:
            out = pd.concat(parts + [pd.DataFrame(rows)], ignore_index=True)
            out.to_parquet(CACHE_PQ)
    out = pd.concat(parts + [pd.DataFrame(rows)], ignore_index=True)
    out = (out.drop_duplicates(["date", "ticker"])
              .sort_values(["date", "ticker"]).reset_index(drop=True))
    out.to_parquet(CACHE_PQ)
    if verbose:
        print(f"cache complete: {out['date'].nunique()} dates, "
              f"{len(out)} forecasts, {time.time() - t0:.0f}s", flush=True)
    return out


def to_wide(long_df, col, index, columns, ffill_limit=SIG_EVERY - 1):
    """Sparse long scores -> dates x tickers, ffilled to cover buy grids."""
    w = long_df.pivot(index="date", columns="ticker", values=col)
    w = w.reindex(index=index, columns=columns)
    return w.ffill(limit=ffill_limit)


if __name__ == "__main__":
    import protocol

    long_df = build_cache(force="--force" in sys.argv)
    P = protocol.get_shared()["panels"]
    idx, cols = P["close"].index, P["close"].columns

    S_chronos = to_wide(long_df, "chronos", idx, cols)
    S_control = to_wide(long_df, "mom", idx, cols)

    for k in (1, 2, 3):
        protocol.evaluate_signal(S_chronos, f"chronos_rerank_k{k}",
                                 k=k, every=SIG_EVERY)
        protocol.evaluate_signal(S_control, f"chronos_control_mom_k{k}",
                                 k=k, every=SIG_EVERY)
