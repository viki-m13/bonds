"""PatchTST cross-sectional ranker for the biweekly-DCA harness.

A genuinely different model from the CNN / LightGBM / Chronos already on
record: PatchTST (Nie et al. 2023, "A Time Series is Worth 64 Words") is the
current SOTA patching-transformer for time series, used here via HuggingFace
`PatchTSTForRegression`. We train it FROM SCRATCH, walk-forward, so there is no
pretraining-corpus overlap with the evaluation period (the caveat that weakened
the Chronos test).

Framing is identical to strategy_cnn so the comparison is apples-to-apples:
  * inputs: the same causal (3 x 126) channel windows (~1y of vol-normalised
    return / excess-return / volume-z, subsampled every 2nd day);
  * target: the within-date rank-percentile of the forward 63-day return
    (continuous in [0,1]) -> a pure cross-sectional ranking objective (MSE);
  * walk-forward refit, labels fully closed before the fit date, scored
    strictly after. PatchTST does its own per-sample instance normalisation
    (scaling="std"), which is causal.

Run:  python strategy_patchtst.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as data_mod          # noqa: E402
import strategy_cnn as sc        # reuse the exact causal feature pipeline  # noqa: E402

# ---- hyper-parameters (transformer is slower -> biennial refit) -------------
REFIT_EVERY = 504    # refit every ~2 trading years (9 refits 2009->present)
EPOCHS = 5
BATCH = 256
LR = 5e-4
PATCH_LEN = 16
PATCH_STRIDE = 8
D_MODEL = 64
N_LAYERS = 3
N_HEADS = 4
FFN = 128
SEED = 7


def _model():
    from transformers import PatchTSTConfig, PatchTSTForRegression
    cfg = PatchTSTConfig(
        num_input_channels=3, context_length=sc.WINDOW,
        patch_length=PATCH_LEN, patch_stride=PATCH_STRIDE,
        prediction_length=1, num_targets=1,
        d_model=D_MODEL, num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS, ffn_dim=FFN,
        dropout=0.2, scaling="std", loss="mse")
    return PatchTSTForRegression(cfg)


def _rank_pct_targets(fwd_row, elig_row):
    """Within-date rank-percentile (in [0,1]) of forward return, eligible only."""
    m = elig_row & np.isfinite(fwd_row)
    out = np.full_like(fwd_row, np.nan, dtype=np.float32)
    if m.sum() > 1:
        v = fwd_row[m]
        r = v.argsort().argsort().astype(np.float32) / (len(v) - 1)
        out[m] = r
    return out


def build_scores(P: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if P is None:
        P = data_mod.build_panel()
    close = P["close"]
    idx, cols = close.index, close.columns
    T, N = close.shape

    chan = sc._channels(P)                          # (C, T, N)
    elig = sc._eligible(P)                           # (T, N)
    closev = close.to_numpy(float)

    fwd = np.full((T, N), np.nan)
    fwd[:T - sc.HORIZON] = closev[sc.HORIZON:] / closev[:T - sc.HORIZON] - 1.0
    target = np.vstack([_rank_pct_targets(fwd[i], elig[i]) for i in range(T)])

    first_fit = idx.searchsorted(pd.Timestamp(sc.FIRST_FIT))
    train_grid = np.arange(sc.SPAN, T - sc.HORIZON, sc.TRAIN_STRIDE)
    score_grid = np.arange(max(sc.SPAN, first_fit), T, sc.SCORE_STRIDE)
    scores = np.full((T, N), np.nan, np.float32)
    fit_rows = list(range(first_fit, T, REFIT_EVERY))

    for fi, fit in enumerate(fit_rows):
        tr_rows = train_grid[train_grid + sc.HORIZON <= fit]
        Rr, Rc = [], []
        for r in tr_rows:
            c = np.nonzero(elig[r] & np.isfinite(target[r]))[0]
            if len(c):
                Rr.append(np.full(len(c), r)); Rc.append(c)
        rows = np.concatenate(Rr); colz = np.concatenate(Rc)
        X = np.transpose(sc._windows(chan, rows, colz), (0, 2, 1))   # (n,W,C)
        y = target[rows, colz][:, None]

        model = _model()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
        n = len(yt); model.train()
        g = torch.Generator().manual_seed(SEED + fi)
        for ep in range(EPOCHS):
            perm = torch.randperm(n, generator=g)
            for b in range(0, n, BATCH):
                ix = perm[b:b + BATCH]
                opt.zero_grad()
                out = model(past_values=Xt[ix], target_values=yt[ix])
                out.loss.backward()
                opt.step()

        hi = fit_rows[fi + 1] if fi + 1 < len(fit_rows) else T
        model.eval()
        sgrid = score_grid[(score_grid >= fit) & (score_grid < hi)]
        with torch.no_grad():
            for r in sgrid:
                c = np.nonzero(elig[r])[0]
                if not len(c):
                    continue
                Xr = np.transpose(sc._windows(chan, np.full(len(c), r), c),
                                  (0, 2, 1))
                pred = model(past_values=torch.from_numpy(Xr)
                             ).regression_outputs.numpy().ravel()
                scores[r, c] = pred
        if verbose:
            print(f"  refit {idx[fit].date()}  train={n:>7}  "
                  f"oos->{idx[min(hi, T-1)].date()}", flush=True)

    df = pd.DataFrame(scores, index=idx, columns=cols)
    return df.ffill(limit=sc.SCORE_STRIDE * 2)


if __name__ == "__main__":
    import protocol

    print("building PatchTST walk-forward scores...", flush=True)
    P = data_mod.build_panel()
    S = build_scores(P)
    out = os.path.join(HERE, "research", "patchtst_scores.parquet")
    S.to_parquet(out)
    print("scored days with >=1 pick:",
          int((S.notna().sum(axis=1) > 0).sum()), "/", len(S), flush=True)
    for k in (2, 3):
        protocol.evaluate_signal(S, f"patchtst_k{k}", k=k)
