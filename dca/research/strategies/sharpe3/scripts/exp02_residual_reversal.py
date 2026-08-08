"""Invention track A: RESIDUAL reversal.

Idea: raw short-term reversal is contaminated by factor moves (market, sector
momentum) that do NOT revert. Strip the systematic part with a rolling PCA
factor model estimated PIT, revert only the idiosyncratic residual, skip the
most recent day (bid-ask bounce we can't capture), weight proportionally to
the z-score, rebalance weekly. DEV only.

Variants swept here (all logged): horizon 3/5/10d, skip 0/1d, nfac 0/5/10,
weighting decile vs proportional.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
DEV_END = pd.Timestamp("2014-12-31")
idx = R.index
wk = [d for d in G.week_ends(idx) if pd.Timestamp("1995-01-01") < d <= DEV_END]
E_wk = G.elig_on(wk, ELIG)
Rv = R.astype(np.float32)

def residual_signal(horizon, skip, nfac, lookback=126):
    """For each weekly date: PCA on trailing window of eligible names,
    residual cumulative return over [t-skip-horizon, t-skip]. Returns dict date->Series."""
    out = {}
    for d in wk:
        i = idx.get_loc(d)
        if i < lookback + 30: continue
        e = E_wk.loc[d]; names = e[e].index
        Rw = Rv.iloc[i-lookback:i+1][names]
        good = Rw.columns[Rw.notna().sum() > lookback*0.9]
        Rw = Rw[good].fillna(0.0)
        X = Rw.values
        mu = X.mean(0); Xc = X - mu
        if nfac > 0:
            # PCA via SVD on the window
            try:
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            F = U[:, :nfac] * S[:nfac]          # factor returns
            B = Vt[:nfac].T                      # loadings
            resid = Xc - F @ B.T
        else:
            resid = Xc - Xc.mean(1, keepdims=True)  # just de-mean cross-sectionally
        w = resid[-(horizon+skip):len(resid)-skip if skip else None]
        sig = w.sum(0)
        z = (sig - sig.mean()) / (sig.std() + 1e-12)
        out[d] = pd.Series(-z, index=good)       # long losers
    return out

def to_W(sigdict, prop=True, q=0.1):
    rows = []
    for d, s in sigdict.items():
        s = s.dropna()
        if prop:
            s = s.clip(-3, 3)
            pos = s[s > 0]; neg = s[s < 0]
            w = pd.concat([0.5*pos/pos.sum(), 0.5*neg/(-neg).sum()]) if len(pos) and len(neg) else pd.Series(dtype=float)
        else:
            w = pd.Series(G.normalize_ls(s, topq=q, botq=q))
        w.name = d
        rows.append(w)
    W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
    return W

results = []
for horizon, skip, nfac, prop in [(5,1,10,True), (5,1,5,True), (5,0,10,True), (3,1,10,True),
                                   (10,1,10,True), (5,1,0,True), (5,1,10,False)]:
    sig = residual_signal(horizon, skip, nfac)
    W = to_W(sig, prop=prop)
    net, gross, tno = G.run(W, R)
    net, gross, tno = net[:DEV_END], gross[:DEV_END], tno[:DEV_END]
    rep = G.report(net, gross, tno, f"resrev h{horizon} skip{skip} f{nfac} {'prop' if prop else 'dec'}")
    print(G.fmt(rep), flush=True)
    results.append(rep)
pd.DataFrame(results).to_csv("/tmp/sharpe3_work/exp02.csv", index=False)
print(f"exp02 done t={time.time()-t0:.0f}s", flush=True)
