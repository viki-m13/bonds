"""Invention track C: the ensemble. Recompute the best sleeve configurations
(chosen from battery1/exp02/exp03 DEV results), save each sleeve's daily net
returns, then combine:
  - equal-risk weighting across sleeves (risk parity on trailing 63d vol, PIT)
  - portfolio vol targeting (scale gross to hit 10% ann vol, PIT trailing 21d)
  - regime tilt: reversal sleeves scaled up when trailing 21d market vol is high
Sharpe of the combined book is the headline. DEV only.

NOTE: sleeve list is EDITED BY HAND after earlier experiments; every tried
combination goes to the worklog.
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
HERE = "/tmp/sharpe3_work"

# ---------------- sleeve definitions (filled from earlier results) ----------------
def sleeve_returns(name):
    p = f"{HERE}/sleeve_{name}.pkl"
    if os.path.exists(p):
        return pd.read_pickle(p)
    W = BUILDERS[name]()
    net, gross, tno = G.run(W, R)
    net = net[:DEV_END]
    net.to_pickle(p)
    print(f"built sleeve {name}: {G.fmt(G.report(net, gross[:DEV_END], tno[:DEV_END], name))}", flush=True)
    return net

def _prop_weights(sigdict):
    rows = []
    for d, s in sigdict.items():
        s = s.dropna().clip(-3, 3)
        pos = s[s > 0]; neg = s[s < 0]
        if len(pos) == 0 or len(neg) == 0: continue
        w = pd.concat([0.5*pos/pos.sum(), 0.5*neg/(-neg).sum()]); w.name = d
        rows.append(w)
    return pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)

def build_resrev():
    Rv = R.astype(np.float32)
    out = {}
    for d in wk:
        i = idx.get_loc(d)
        if i < 160: continue
        e = E_wk.loc[d]; names = e[e].index
        Rw = Rv.iloc[i-126:i+1][names]
        good = Rw.columns[Rw.notna().sum() > 113]
        Rw = Rw[good].fillna(0.0)
        Xc = Rw.values - Rw.values.mean(0)
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError: continue
        F = U[:, :10]*S[:10]; B = Vt[:10].T
        resid = Xc - F @ B.T
        sig = resid[-6:-1].sum(0)          # 5d, skip most recent day
        z = (sig - sig.mean())/(sig.std()+1e-12)
        out[d] = pd.Series(-z, index=good)
    return _prop_weights(out)

BUILDERS = {"resrev": build_resrev}
# more sleeves appended by later edits

if __name__ == "__main__":
    parts = {nm: sleeve_returns(nm) for nm in BUILDERS}
    A = pd.DataFrame(parts).dropna(how="all")
    print("sleeve corr:\n", A.corr().round(2), flush=True)
    # risk parity, PIT
    volp = A.rolling(63, min_periods=40).std().shift(1)
    Wrp = (1/volp); Wrp = Wrp.div(Wrp.sum(axis=1), axis=0)
    comb = (A * Wrp).sum(axis=1).dropna()
    # vol target 10% using trailing 21d of the combined book
    lev = (0.10/np.sqrt(252)) / comb.rolling(21, min_periods=15).std().shift(1)
    lev = lev.clip(0, 5.0)
    combT = (comb * lev).dropna()
    for nm, r_ in [("ensemble raw", comb), ("ensemble voltarget", combT)]:
        print(f"{nm:24} Sharpe {G.sharpe(r_):.2f}", flush=True)
    print(f"exp04 done t={time.time()-t0:.0f}s", flush=True)
