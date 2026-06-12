"""Leakage audit: verify a signal builder is causal.

Method (truncation test): pick random audit dates T. Rebuild the signal with
all panel data hard-truncated at T (rows after T removed). The signal row at
T must be identical to the row at T from the full-sample build. Any
difference ⇒ the builder uses post-T information (lookahead) or full-sample
fitting. This catches centered windows, global z-scores, future shifts, and
ML fit-on-all-data bugs.

Usage: the builder must be a function f(panels: dict) -> DataFrame, where
panels has keys open/high/low/close/volume/member.
"""
import numpy as np
import pandas as pd

import data as data_mod


def audit_builder(builder, n_dates: int = 6, seed: int = 3,
                  tol: float = 1e-9, verbose: bool = True) -> bool:
    P = data_mod.build_panel()
    full = builder(P)
    idx = full.index
    rng = np.random.default_rng(seed)
    # audit dates spread over the sample, away from the very start
    cand = idx[int(len(idx) * 0.25):-1]
    dates = sorted(rng.choice(len(cand), size=n_dates, replace=False))
    ok = True
    for di in dates:
        T = cand[di]
        Pt = {k: v.loc[:T] for k, v in P.items()}
        part = builder(Pt)
        a = full.loc[T].astype(float)
        b = part.loc[T].reindex(a.index).astype(float)
        both = a.notna() & b.notna()
        diff = (a[both] - b[both]).abs().max() if both.any() else 0.0
        nan_mismatch = int((a.notna() != b.notna()).sum())
        good = (diff is np.nan or diff <= tol) and nan_mismatch == 0
        ok &= bool(good)
        if verbose:
            print(f"  audit {T.date()}: max|Δ|={diff:.2e} "
                  f"nan-mismatch={nan_mismatch} -> {'OK' if good else 'LEAK'}")
    return ok
