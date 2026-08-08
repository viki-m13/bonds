"""Invention track D: volume-conditioned reversal/continuation (LMSW logic).

A big weekly move on QUIET volume = liquidity pressure -> expect reversal.
A big weekly move on LOUD volume = information -> expect drift, not reversal.
Sleeves:
  A: residual reversal restricted to quiet-volume movers (the clean half)
  B: continuation on loud-volume movers (the other half, opposite sign)
  C: A+B combined book
Also: reversal signal interacted continuously: sig = -resid5 * (1 - vz)
DEV only, weekly, proportional weights, gross 1.
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
volz_m = (DV.rolling(5).mean() / DV.rolling(63, min_periods=40).mean())

def signals(d):
    i = idx.get_loc(d)
    if i < 160: return None
    e = E_wk.loc[d]; names = e[e].index
    Rw = Rv.iloc[i-126:i+1][names]
    good = Rw.columns[Rw.notna().sum() > 113]
    Rw = Rw[good].fillna(0.0)
    Xc = Rw.values - Rw.values.mean(0)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    F = U[:, :10]*S[:10]; B = Vt[:10].T
    resid = Xc - F @ B.T
    sig5 = pd.Series(resid[-6:-1].sum(0), index=good)
    z5 = (sig5 - sig5.mean())/(sig5.std()+1e-12)
    vz = volz_m.loc[:d].iloc[-1][good]
    vrank = vz.rank(pct=True)
    return z5, vrank

def prop(s):
    s = s.dropna().clip(-3, 3)
    pos = s[s > 0]; neg = s[s < 0]
    if len(pos) == 0 or len(neg) == 0: return None
    return pd.concat([0.5*pos/pos.sum(), 0.5*neg/(-neg).sum()])

rowsA, rowsB, rowsC, rowsI = [], [], [], []
for d in wk:
    out = signals(d)
    if out is None: continue
    z5, vrank = out
    quiet = vrank < 0.5; loud = vrank > 0.8
    a = prop(-z5[quiet])
    b = prop(z5[loud])
    ii = prop(-z5 * (1 - vrank))
    if a is not None: a.name = d; rowsA.append(a)
    if b is not None: b.name = d; rowsB.append(b)
    if a is not None or b is not None:
        c = pd.concat([x*0.5 for x in (a, b) if x is not None]); c = c.groupby(c.index).sum(); c.name = d
        rowsC.append(c)
    if ii is not None: ii.name = d; rowsI.append(ii)

for nm, rows in [("A quiet-vol reversal", rowsA), ("B loud-vol continuation", rowsB),
                 ("C both halves", rowsC), ("I interacted -z*(1-vrank)", rowsI)]:
    W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
    net, gross, tno = G.run(W, R)
    rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], nm)
    print(G.fmt(rep), flush=True)
print(f"exp05 done t={time.time()-t0:.0f}s", flush=True)
