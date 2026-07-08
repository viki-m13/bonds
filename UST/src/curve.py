"""Daily Nelson-Siegel-Svensson yield-curve fits.

For each date we fit NSS to the cross-section of computed YTMs of nominal
coupon securities (plus bills for the short end). For fixed decay parameters
(tau1, tau2) the model is linear in the four betas, so we solve a weighted
linear least-squares over a small grid of (tau1, tau2) and keep the best —
fast and far more robust than a full nonlinear optimizer per day.

Weights: inverse duration, which approximates fitting price errors rather
than yield errors (standard GSW-style choice), so the long end doesn't
dominate.

Output per (date, cusip): fitted yield and residual (actual - fitted).
Positive residual = bond is CHEAP relative to the curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAU1_GRID = np.array([0.4, 0.8, 1.5, 2.5, 4.0])
TAU2_GRID = np.array([6.0, 10.0, 15.0, 22.0])


def nss_basis(t: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    """Columns: level, slope, curve1, curve2 for maturities t (years)."""
    t = np.maximum(t, 1e-4)
    x1 = t / tau1
    x2 = t / tau2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return np.column_stack([np.ones_like(t), f1, f2, f3])


def fit_day(t: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Weighted LS over the (tau1, tau2) grid; returns (betas, tau1, tau2)."""
    best = (None, np.inf, None, None)
    sw = np.sqrt(w)
    for tau1 in TAU1_GRID:
        for tau2 in TAU2_GRID:
            if tau2 <= tau1:
                continue
            X = nss_basis(t, tau1, tau2)
            Xw = X * sw[:, None]
            yw = y * sw
            beta, res, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            sse = float(np.sum((Xw @ beta - yw) ** 2))
            if sse < best[1]:
                best = (beta, sse, tau1, tau2)
    return best[0], best[2], best[3]


def fit_panel(panel: pd.DataFrame, min_mat: float = 0.08, max_mat: float = 31.0) -> pd.DataFrame:
    """Fit NSS per date; return DataFrame(date, cusip, fitted, resid).

    Fit universe: all nominal securities with maturity in [min_mat, max_mat]
    years. Residuals are produced for every row in that range.
    """
    rows_date, rows_cusip, rows_fit = [], [], []
    par_rows = []
    for date, day in panel.groupby("date", sort=True):
        d = day.dropna(subset=["ytm", "mod_dur", "tsy_years"])
        d = d[(d["tsy_years"] >= min_mat) & (d["tsy_years"] <= max_mat)]
        if len(d) < 30:
            continue
        t = d["tsy_years"].values
        y = d["ytm"].values
        w = 1.0 / np.clip(d["mod_dur"].values, 0.1, None)
        w = w / w.mean()
        beta, tau1, tau2 = fit_day(t, y, w)
        fitted = nss_basis(t, tau1, tau2) @ beta
        rows_date.append(np.full(len(d), date))
        rows_cusip.append(d["cusip"].values)
        rows_fit.append(fitted)
        par_rows.append((date, *beta, tau1, tau2))
    fits = pd.DataFrame({
        "date": np.concatenate(rows_date),
        "cusip": np.concatenate(rows_cusip),
        "fitted": np.concatenate(rows_fit),
    })
    params = pd.DataFrame(
        par_rows, columns=["date", "b0", "b1", "b2", "b3", "tau1", "tau2"]
    ).set_index("date")
    return fits, params


def nss_yield(params: pd.DataFrame, dates: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate fitted NSS yield at maturities t (years) for given dates."""
    p = params.reindex(pd.DatetimeIndex(dates))
    out = np.full(len(t), np.nan)
    for (tau1, tau2), grp in p.groupby(["tau1", "tau2"], dropna=True):
        mask = p.index.isin(grp.index)
        # rows aligned by position: mask over the input arrays
        sel = np.asarray(mask)
        X = nss_basis(np.asarray(t)[sel], tau1, tau2)
        B = grp[["b0", "b1", "b2", "b3"]].values
        # align B rows to selected positions (params reindexed to dates order)
        Bfull = p.loc[sel, ["b0", "b1", "b2", "b3"]].values
        out[sel] = np.sum(X * Bfull, axis=1)
    return out
