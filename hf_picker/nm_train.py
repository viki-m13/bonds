"""Walk-forward training of the L2GMOM network-momentum model on the
underwater-avoidance objective, plus the no-graph linear ablation.

Label: realised underwater fraction over the next H trading days, standardised
cross-sectionally within each date (the model learns the ordering selection
uses). Lower predicted = safer; the cached score is the negative, so "higher =
safer" like every other arm.

Leakage control (mirrors the repo's ML harness): expanding-window refits at a
set of cutoffs. A model used to predict dates after cutoff C is trained ONLY on
signal dates d with d + H <= C, so every training label is fully realised
before any predicted date. Within the training set the last 20% of dates are a
validation tail for early stopping.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from data import load_panel
import nm_features
from nm_model import L2GMOM, LinearMom
from objective import Arrays, signal_positions
from evaluate import realized_panel

_HERE = os.path.dirname(os.path.abspath(__file__))
HORIZON = 126
EVERY = 21                       # monthly signal/eval cadence
CUTOFFS = ["2011-01-01", "2014-01-01", "2017-01-01", "2020-01-01", "2023-01-01"]


def label_matrix(arr, horizon, start, end):
    """uw_frac[day, ticker] at monthly dates (NaN elsewhere)."""
    rp = realized_panel(arr, horizon, start, end, every=EVERY)
    T, N = arr.close.shape
    lab = np.full((T, N), np.nan, np.float32)
    lab[rp["sig_pos"].to_numpy(), rp["ticker"].to_numpy()] = rp["uw_frac"].to_numpy()
    return lab


def _samples(feat, lab, positions):
    """Per-date (U, z-target, ticker_idx) for graph training."""
    out = []
    for p in positions:
        mask = (~np.isnan(feat[:, p, :]).any(axis=0)) & (~np.isnan(lab[p]))
        idx = np.where(mask)[0]
        if len(idx) < 30:
            continue
        U = torch.tensor(feat[:, p, idx].T)             # (n, F)
        y = lab[p, idx]
        z = (y - y.mean()) / (y.std() + 1e-8)
        out.append((U, torch.tensor(z, dtype=torch.float32), p, idx))
    return out


def _fit(model, train, val, epochs=40, lr=0.01, patience=6):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(train))
        for i in order:
            U, z, _, _ = train[i]
            opt.zero_grad()
            pred = model(U)
            loss = ((pred - z) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([((model(U) - z) ** 2).mean().item()
                          for U, z, _, _ in val]) if val else 0.0
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in
                                         model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best


def train_arm(graph: bool, feat, lab, idx, n_features, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T, N = lab.shape
    score = np.full((T, N), np.nan, np.float32)
    cutoffs = [pd.Timestamp(c) for c in CUTOFFS]
    eval_pos = signal_positions(idx, EVERY, 0, CUTOFFS[0], None)
    for ci, cut in enumerate(cutoffs):
        nxt = cutoffs[ci + 1] if ci + 1 < len(cutoffs) else idx[-1] + pd.Timedelta(days=1)
        # training dates: monthly, labels realised before cut
        train_end = cut - pd.Timedelta(days=int(HORIZON * 1.5))
        tr_pos = signal_positions(idx, EVERY, 0, "2005-01-01", train_end)
        samp = _samples(feat, lab, tr_pos)
        if len(samp) < 20:
            continue
        k = max(1, int(len(samp) * 0.8))
        train, val = samp[:k], samp[k:]
        model = L2GMOM(n_features, graph=True) if graph else LinearMom(n_features)
        vloss = _fit(model, train, val)
        # predict eval dates in (cut, nxt]
        ev = eval_pos[(idx[eval_pos] > cut) & (idx[eval_pos] <= nxt)]
        model.eval()
        with torch.no_grad():
            for p in ev:
                mask = ~np.isnan(feat[:, p, :]).any(axis=0)
                jj = np.where(mask)[0]
                if len(jj) < 30:
                    continue
                U = torch.tensor(feat[:, p, jj].T)
                score[p, jj] = -model(U).numpy()         # higher = safer
        print(f"  {'graph' if graph else 'linear'} cutoff {cut.date()}: "
              f"train {len(train)} val {len(val)} vloss {vloss:.4f} "
              f"predicted {len(ev)} dates", flush=True)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["graph", "linear", "both"], default="both")
    a = ap.parse_args()
    arr = Arrays()
    feat, idx, cols, names = nm_features.build_features()
    lab = label_matrix(arr, HORIZON, "2005-01-01", None)
    nf = feat.shape[0]
    if a.arm in ("graph", "both"):
        s = train_arm(True, feat, lab, idx, nf)
        pd.DataFrame(s, index=idx, columns=cols).to_parquet(
            os.path.join(_HERE, "l2gmom_score.parquet"))
        print("saved l2gmom_score.parquet")
    if a.arm in ("linear", "both"):
        s = train_arm(False, feat, lab, idx, nf)
        pd.DataFrame(s, index=idx, columns=cols).to_parquet(
            os.path.join(_HERE, "linmom_score.parquet"))
        print("saved linmom_score.parquet")


if __name__ == "__main__":
    main()
