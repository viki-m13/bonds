"""CNN stock-selection signal for the biweekly DCA harness.

Identity: a 1-D convolutional network reads each stock's recent price/volume
"shape" and predicts whether it will out-perform the cross-section over the
next ~month. The per-date, per-ticker out-performance logits become the
`scores` matrix the DCA engine consumes (top-k at next open).

Why a CNN: momentum signals hard-code a few formation windows (the SUMMIT
12-1 / 9-1 horizons). A small Conv1d stack instead *learns* which local
return/volume patterns precede relative strength, sharing weights across the
whole lookback so it can pick up acceleration, base-building, volume thrust,
etc. without us naming them.

Causality contract (see RESEARCH_PROTOCOL.md, NON-NEGOTIABLE):
  * A feature window at date d uses closes/volumes through the CLOSE of d only.
    All normalisation is trailing (rolling std / z-score) or self-contained
    within the window -- no full-sample or centred statistics.
  * The label is the forward H-day return, used ONLY in training.
  * The net is fit WALK-FORWARD: at each annual refit date T we train on
    samples whose label window has fully closed on or before T, then predict
    strictly after T. No sample ever sees its own future, and no later year's
    data leaks into an earlier prediction.

Run:  python strategy_cnn.py            # train, score, evaluate, save card
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as data_mod  # noqa: E402

# ---- hyper-parameters -------------------------------------------------------
# The lookback must span ~1 year so the net can see the 6-12m momentum that the
# baseline signals exploit; we subsample every 2nd day to keep the sequence
# short (WINDOW points x SUBSAMPLE spacing = ~252 calendar trading days).
WINDOW = 126         # points fed to the CNN ...
SUBSAMPLE = 2        # ... spaced 2 trading days apart -> ~252d of history
HORIZON = 63         # forward days the label looks ahead (~1 quarter)
VOL_WIN = 63         # trailing window for return vol-normalisation
TRAIN_STRIDE = 21    # sample one training row per ticker per ~month
SCORE_STRIDE = 5     # emit predictions every 5 trading days (ffilled to daily)
REFIT_EVERY = 252    # walk-forward: refit once a trading year
FIRST_FIT = "2009-01-01"   # first refit (>=4y of training history before it)
EPOCHS = 8
BATCH = 512
LR = 1e-3
SEED = 7
SPAN = (WINDOW - 1) * SUBSAMPLE   # left history a window anchor needs


def _channels(P: dict):
    """Three causal (T x N) feature channels, all trailing-normalised.

    c0 vol-normalised log return   (own momentum, scale-free)
    c1 vol-normalised excess return vs the equal-weight market  (relative str.)
    c2 trailing z-scored log volume                              (participation)
    """
    close = P["close"].astype(float)
    vol = P["volume"].astype(float)
    logret = np.log(close).diff()
    sig = logret.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std()
    mkt = logret.mean(axis=1)                     # equal-weight market return
    zret = (logret / sig)
    zrel = ((logret.sub(mkt, axis=0)) / sig)
    logv = np.log(vol.clip(lower=1.0))
    vz = (logv - logv.rolling(VOL_WIN, min_periods=VOL_WIN // 2).mean()) / \
        logv.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std()
    chans = [zret, zrel, vz]
    arrs = [c.to_numpy(np.float32) for c in chans]
    # clip fat tails so a single bad tick can't dominate a conv filter
    arrs = [np.clip(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), -8, 8)
            for a in arrs]
    return np.stack(arrs, axis=0)                 # (C, T, N)


def _windows(chan: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    """Gather (n, C, WINDOW) feature tensors for the given (row, col) anchors.

    Anchor row i means the window spans closes [i-WINDOW+1 .. i] inclusive, so
    every value is known at the close of day i."""
    C = chan.shape[0]
    offs = np.arange(-SPAN, 1, SUBSAMPLE)          # (WINDOW,) subsampled span
    ridx = rows[:, None] + offs[None, :]          # (n, WINDOW)
    out = chan[:, ridx, cols[:, None]]            # (C, n, WINDOW)
    return np.transpose(out, (1, 0, 2)).copy()    # (n, C, WINDOW)


def _eligible(P: dict):
    """Boolean (T x N): index member, >=1y of history, valid close today."""
    close = P["close"]
    enough = close.notna().rolling(252).count() >= 252
    return (P["member"].to_numpy(bool) & enough.to_numpy(bool)
            & close.notna().to_numpy(bool))


def _build_model():
    import torch
    import torch.nn as nn

    class StockCNN(nn.Module):
        """Conv stack -> concat(avg-pool, max-pool) -> MLP head.

        The avg-pool branch is essential: momentum is the *integral* of returns
        over the window, which max-pool alone cannot represent (an earlier
        max-only version scored ~0 IC because it could not sum returns). Avg-pool
        over the vol-normalised return channels gives the net a risk-adjusted
        cumulative-return feature to build on; max-pool keeps the shift-tolerant
        pattern detector."""
        def __init__(self, c_in=3):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(c_in, 16, 5, padding=2), nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            )
            self.amax = nn.AdaptiveMaxPool1d(1)
            self.aavg = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(64, 16), nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            h = self.conv(x)
            z = torch.cat([self.amax(h).squeeze(-1),
                           self.aavg(h).squeeze(-1)], dim=1)
            return self.head(z).squeeze(-1)

    return StockCNN()


def build_scores(P: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    """Walk-forward CNN out-performance logits, aligned (dates x tickers)."""
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if P is None:
        P = data_mod.build_panel()
    close = P["close"]
    idx, cols = close.index, close.columns
    T, N = close.shape

    chan = _channels(P)                                   # (C, T, N)
    elig = _eligible(P)                                   # (T, N)
    closev = close.to_numpy(float)

    # forward H-day return and its within-date cross-sectional median; label is
    # 1 if a name beats the eligible median over the next HORIZON days.
    fwd = np.full((T, N), np.nan)
    fwd[:T - HORIZON] = closev[HORIZON:] / closev[:T - HORIZON] - 1.0
    fwd_masked = np.where(elig, fwd, np.nan)
    med = np.nanmedian(fwd_masked, axis=1)                # (T,)
    label = (fwd > med[:, None]).astype(np.float32)

    first_fit = idx.searchsorted(pd.Timestamp(FIRST_FIT))
    # rows usable as anchors: enough left history for a window, enough right
    # room for a label (the latter only matters for *training* anchors).
    train_grid = np.arange(SPAN, T - HORIZON, TRAIN_STRIDE)
    score_grid = np.arange(max(SPAN, first_fit), T, SCORE_STRIDE)

    scores = np.full((T, N), np.nan, np.float32)
    fit_rows = list(range(first_fit, T, REFIT_EVERY))
    device = "cpu"

    for fi, fit in enumerate(fit_rows):
        # --- assemble training set: labels fully observed by the fit date ---
        tr_rows = train_grid[train_grid + HORIZON <= fit]
        Xr, Xc = [], []
        for r in tr_rows:
            c = np.nonzero(elig[r] & np.isfinite(fwd[r]))[0]
            if len(c):
                Xr.append(np.full(len(c), r))
                Xc.append(c)
        rows = np.concatenate(Xr)
        colz = np.concatenate(Xc)
        X = _windows(chan, rows, colz)
        y = label[rows, colz]

        model = _build_model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        n = len(yt)
        model.train()
        g = torch.Generator().manual_seed(SEED + fi)
        for ep in range(EPOCHS):
            perm = torch.randperm(n, generator=g)
            for b in range(0, n, BATCH):
                ix = perm[b:b + BATCH]
                opt.zero_grad()
                out = model(Xt[ix])
                loss = lossf(out, yt[ix])
                loss.backward()
                opt.step()

        # --- predict for this refit's out-of-sample slice [fit, next_fit) ---
        hi = fit_rows[fi + 1] if fi + 1 < len(fit_rows) else T
        model.eval()
        sgrid = score_grid[(score_grid >= fit) & (score_grid < hi)]
        with torch.no_grad():
            for r in sgrid:
                c = np.nonzero(elig[r])[0]
                if not len(c):
                    continue
                Xr = _windows(chan, np.full(len(c), r), c)
                logit = model(torch.from_numpy(Xr)).numpy()
                scores[r, c] = logit
        if verbose:
            print(f"  refit {idx[fit].date()}  train={n:>7}  "
                  f"oos={idx[fit].date()}->{idx[min(hi, T-1)].date()}")

    df = pd.DataFrame(scores, index=idx, columns=cols)
    # signal dates fall between score-grid days; carry the last known logit
    # forward (still strictly causal -- only past predictions are reused).
    return df.ffill(limit=SCORE_STRIDE * 2)


def current_picks(k: int = 3):
    """Live helper: today's CNN picks (to execute at the next open)."""
    P = data_mod.build_panel()
    s = build_scores(P, verbose=False)
    elig = pd.DataFrame(_eligible(P), index=P["close"].index,
                        columns=P["close"].columns)
    row = s.iloc[-1].where(elig.iloc[-1]).dropna()
    return row.sort_values(ascending=False).head(k)


if __name__ == "__main__":
    import protocol

    print("building CNN walk-forward scores...")
    P = data_mod.build_panel()
    S = build_scores(P)
    out = os.path.join(HERE, "research", "cnn_scores.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    S.to_parquet(out)
    print("scored days with >=1 pick:",
          int((S.notna().sum(axis=1) > 0).sum()), "/", len(S))

    for k in (2, 3):
        protocol.evaluate_signal(S, f"cnn_k{k}", k=k)
