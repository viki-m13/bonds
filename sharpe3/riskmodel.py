"""Statistical risk model + factor-neutral portfolio construction.

Every portfolio so far has been a naive decile long-short, which carries large
uncontrolled exposures to market/sector/size/vol factors. Real stat-arb desks
neutralize those: the same signal, hedged, can have materially higher Sharpe
because the factor noise is removed from the P&L.

Model: rolling PCA on the trailing 252d correlation matrix of member returns
(top K eigenvectors = statistical factors). Refit monthly, applied forward
only. Signal is then residualized against the loadings, and the portfolio is
built to be (approximately) factor-neutral and dollar-neutral.

Causality: factors at date d are fit on returns through d only; the resulting
neutralized signal is used for execution at d+1's open (bt.run handles lag).
"""
import numpy as np
import pandas as pd


def rolling_factor_loadings(returns, member, n_factors=10, window=252, refit=21,
                            min_names=100):
    """Yield (date_index_position, tickers, loadings) at each refit date.

    loadings: (n_names x n_factors) principal-component loadings estimated on
    the trailing `window` days of standardized returns through that date.
    """
    idx = returns.index
    out = []
    for i in range(window, len(idx), refit):
        sl = returns.iloc[i - window:i]
        cols = member.iloc[i - 1]
        cols = cols[cols].index
        sub = sl[cols].dropna(axis=1, thresh=int(window * 0.9))
        if sub.shape[1] < min_names:
            continue
        sub = sub.fillna(0.0)
        z = (sub - sub.mean()) / sub.std().replace(0, np.nan)
        z = z.dropna(axis=1)
        if z.shape[1] < min_names:
            continue
        # PCA via SVD on the standardized panel
        u, s, vt = np.linalg.svd(z.values, full_matrices=False)
        load = vt[:n_factors].T                     # names x factors
        out.append((i, list(z.columns), load))
    return out


def neutralize(signal, returns, member, n_factors=10, window=252, refit=21):
    """Cross-sectionally residualize `signal` against the rolling PCA factors.

    Returns a signal frame of the same shape whose rows are orthogonal to the
    estimated factor loadings (i.e. a factor-neutral alpha).
    """
    fits = rolling_factor_loadings(returns, member, n_factors, window, refit)
    out = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)
    sigv = signal.values
    colpos = {c: j for j, c in enumerate(signal.columns)}
    for k, (i0, cols, load) in enumerate(fits):
        i1 = fits[k + 1][0] if k + 1 < len(fits) else len(signal.index)
        j = np.array([colpos[c] for c in cols])
        # projection matrix onto the orthogonal complement of the loadings
        q, _ = np.linalg.qr(load)                   # names x factors, orthonormal
        for i in range(i0, i1):
            s = sigv[i, j]
            ok = ~np.isnan(s)
            if ok.sum() < 30:
                continue
            sv = s[ok]
            qq = q[ok]
            # re-orthonormalize the sub-block, then remove factor component
            qr, _ = np.linalg.qr(qq)
            resid = sv - qr @ (qr.T @ sv)
            row = np.full(len(j), np.nan)
            row[ok] = resid
            out.iloc[i, j] = row
    return out


def optimized_weights(signal, returns, member, n_factors=10, window=252,
                      refit=21, gross=2.0, max_w=0.03):
    """Factor-neutral, dollar-neutral weights proportional to the neutralized
    signal, with a per-name cap and inverse-vol scaling."""
    alpha = neutralize(signal, returns, member, n_factors, window, refit)
    vol = returns.rolling(60).std().shift(1)
    w = (alpha / vol.replace(0, np.nan)).where(member)
    w = w.sub(w.mean(axis=1), axis=0)                        # dollar-neutral
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0) * gross
    w = w.clip(-max_w * gross, max_w * gross)
    w = w.sub(w.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0) * gross
    return w.fillna(0.0)
