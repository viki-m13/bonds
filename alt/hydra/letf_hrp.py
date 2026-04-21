"""Priority 2d — Hierarchical risk parity (Lopez de Prado 2016).

HRP vs naive inv-vol vs min-var with shrinkage.

HRP builds a diversified portfolio by:
 1. Cluster assets via hierarchical clustering on correlation distance
 2. Within each cluster, allocate by inverse variance
 3. Between clusters, recursively bisect

LdP shows HRP generally delivers HIGHER OOS Sharpe than min-var (which
is theoretically optimal IS but wildly overfits covariance estimates)
and lower drawdowns than naive inv-vol.

Pre-registered: no sweep. HRP on core6 and clean4, 126-day lookback,
monthly rebal. Compare against the invvol clean4 lb=21 baseline.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from letf_engine import (common_window_returns, run_backtest, summarise,
                         w_fixed)
from letf_crypto_universe import load_with_crypto
from letf_universe import LETF_LONG_2011


OUT = Path("/home/user/bonds/data/results")


def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def get_cluster_var(cov, c_items):
    cov_ = cov.loc[c_items, c_items]
    ivp = 1.0 / np.diag(cov_.values)
    w = ivp / ivp.sum()
    return (w @ cov_.values @ w)


def recursive_bisection(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while c_items:
        new = []
        for items in c_items:
            if len(items) <= 1: continue
            half = len(items) // 2
            left = items[:half]
            right = items[half:]
            v_l = get_cluster_var(cov, left)
            v_r = get_cluster_var(cov, right)
            alpha = 1 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= (1 - alpha)
            new.append(left); new.append(right)
        c_items = new
    return w


def hrp_weights(rets_sub):
    cov = rets_sub.cov()
    corr = rets_sub.corr()
    # Distance from correlation
    dist = np.sqrt(0.5 * (1 - corr))
    dist_sq = squareform(dist.values, checks=False)
    link = linkage(dist_sq, method="single")
    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()
    w = recursive_bisection(cov, sort_ix)
    return w.reindex(corr.index).fillna(0)


def hrp_fn(tickers, lookback=126):
    def fn(d, hist):
        if len(hist) < lookback + 10: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] < 2: return None
        w = hrp_weights(r)
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w.values
        return out
    return fn


def min_var_shrink_fn(tickers, lookback=126, shrink=0.5):
    """Min-variance with Ledoit-Wolf-like shrinkage toward diagonal."""
    def fn(d, hist):
        if len(hist) < lookback + 10: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] < 2: return None
        cov = r.cov().values
        diag = np.diag(np.diag(cov))
        cov_s = (1 - shrink) * cov + shrink * diag
        try:
            inv = np.linalg.inv(cov_s)
        except np.linalg.LinAlgError:
            return None
        ones = np.ones(len(cov_s))
        w = inv @ ones
        w = w / w.sum()
        w = np.clip(w, 0, None)  # long-only
        w = w / w.sum() if w.sum() > 0 else np.zeros_like(w)
        out = pd.Series(0.0, index=hist.columns)
        out.loc[r.columns] = w
        return out
    return fn


def invvol_fn(tickers, lookback=63):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        inv = 1 / r.std().replace(0, np.nan).fillna(0)
        w = inv / inv.sum()
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


def main():
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]

    strategies = [
        ("HRP core6 lb=126", hrp_fn(core6, 126)),
        ("HRP clean4 lb=126", hrp_fn(clean4, 126)),
        ("HRP all17 lb=126", hrp_fn(LETF_LONG_2011, 126)),
        ("MV-shrink core6 lb=126 s=0.5", min_var_shrink_fn(core6, 126, 0.5)),
        ("MV-shrink clean4 lb=126 s=0.5", min_var_shrink_fn(clean4, 126, 0.5)),
        ("MV-shrink all17 lb=126 s=0.5", min_var_shrink_fn(LETF_LONG_2011, 126, 0.5)),
        ("IV core6 lb=63", invvol_fn(core6, 63)),
        ("IV clean4 lb=21", invvol_fn(clean4, 21)),
        ("IV all17 lb=63", invvol_fn(LETF_LONG_2011, 63)),
    ]

    rows = []
    for name, fn in strategies:
        r, _ = run_backtest(rets, fn, rebal_days=21, exec_lag=1)
        s = summarise(r, name)
        rows.append(s)
        print(f"  {name}: CAGR={s['cagr']}% SR={s['sharpe']} MDD={s['mdd']}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_hrp.csv", index=False)
    print()
    print(df.sort_values("sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
