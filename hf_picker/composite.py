"""Cross-sectional rank-composite of safety predictors.

Combine predictors the leakage-free way: at each date, rank every name within
that date (members only, NaN-safe), then average the ranks. No coefficients are
fit on outcomes, so there is nothing to overfit to the evaluation grid — the
only choice is *which* predictors to average, made on the IC evidence
(low_vol + Chronos q10 downside margin, the two statistically-significant,
weakly-correlated signals)."""
import numpy as np
import pandas as pd

import baselines
import chronos_signal as cs


def _row_rank(M: np.ndarray) -> np.ndarray:
    """Per-row (per-date) rank in [0,1], NaNs preserved. Higher value ->
    higher rank."""
    out = np.full(M.shape, np.nan)
    order = np.argsort(np.where(np.isnan(M), -np.inf, M), axis=1)
    for i in range(M.shape[0]):
        row = M[i]
        valid = ~np.isnan(row)
        n = valid.sum()
        if n == 0:
            continue
        ranks = pd.Series(row[valid]).rank().to_numpy()
        out[i, valid] = (ranks - 1) / max(n - 1, 1)
    return out


def rank_composite(mats) -> np.ndarray:
    """Average per-date ranks of several score matrices. A name is scored only
    where ALL inputs are present (so every arm sees the same support)."""
    ranks = [_row_rank(M) for M in mats]
    stacked = np.stack(ranks, axis=0)
    present = ~np.isnan(stacked).any(axis=0)
    avg = np.nanmean(stacked, axis=0)
    avg[~present] = np.nan
    return avg


def build_named():
    """The composites we evaluate."""
    low_vol = baselines.build("low_vol")
    qm = cs.q10_margin_score()
    safety = cs.safety_score()
    return {
        "composite_lv_qm": rank_composite([low_vol, qm]),
        "composite_lv_qm_safe": rank_composite([low_vol, qm, safety]),
    }
