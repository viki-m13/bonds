"""Invention track E: keep the gross alpha, kill the turnover.

Machinery:
- Factor model refreshed MONTHLY (PCA loadings on trailing 126d, eligible names);
  daily residuals via cross-sectional OLS on those loadings (fully PIT).
- Composite residual-reversal z: 0.5*z5 + 0.3*z3 + 0.2*z10, risk-scaled (/vol20).
- DAILY portfolio update with hysteresis: enter |z|>zin, exit |z|<zout,
  partial adjustment speed lam toward equal-weight-per-side targets.
Sweep: (zin, zout, lam). DEV only.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
DEV_END = pd.Timestamp("2014-12-31")
idx = R.index
dstart = idx.get_indexer([pd.Timestamp("1995-01-01")], method="nearest")[0]
mo = [d for d in G.month_ends(idx) if pd.Timestamp("1994-06-30") < d <= DEV_END]
Rf = R.fillna(0.0).astype(np.float32)

# ---------- monthly factor loadings -> daily residuals ----------
print("building daily residuals...", flush=True)
resid = pd.DataFrame(np.nan, index=idx, columns=R.columns, dtype=np.float32)
E_mo = G.elig_on(mo, ELIG)
for mi, d in enumerate(mo):
    i = idx.get_loc(d)
    if i < 160: continue
    e = E_mo.loc[d]; names = [c for c in e[e].index if c in R.columns]
    Rw = R.iloc[i-126:i+1][names]
    good = [c for c in Rw.columns[Rw.notna().sum() > 113]]
    if len(good) < 100: continue
    X = Rw[good].fillna(0.0).values
    Xc = X - X.mean(0)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        continue
    B = Vt[:10].T                                  # loadings (n x 10)
    # apply through next month
    j_end = idx.get_loc(mo[mi+1]) if mi+1 < len(mo) else min(i+23, len(idx)-1)
    Rn = Rf.iloc[i+1:j_end+1][good].values         # days x n
    BtB_inv_Bt = np.linalg.pinv(B.T @ B) @ B.T
    F = Rn @ BtB_inv_Bt.T                          # days x 10
    res = Rn - F @ B.T
    resid.iloc[i+1:j_end+1, [resid.columns.get_loc(c) for c in good]] = res
print(f"residuals ready t={time.time()-t0:.0f}s", flush=True)

vol20 = R.rolling(20, min_periods=10).std()
def zmat(h):
    s = resid.rolling(h).sum() / (vol20 * np.sqrt(h) + 1e-9)
    return s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1) + 1e-12, axis=0)
Z = (0.5*zmat(5) + 0.3*zmat(3) + 0.2*zmat(10))
Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1) + 1e-12, axis=0)
Zv = (-Z).values  # long low residual returns
print(f"Z ready t={time.time()-t0:.0f}s", flush=True)

# eligibility daily
E_d = ELIG.reindex(idx).ffill().fillna(False)[R.columns].values

days = [k for k in range(dstart, len(idx)) if idx[k] <= DEV_END]
def run_patient(zin, zout, lam, nmax=75):
    W = np.zeros(len(R.columns), dtype=np.float64)
    rows = np.zeros((len(days), len(R.columns)), dtype=np.float32)
    for a, k in enumerate(days):
        z = Zv[k].copy()
        z[~E_d[k]] = np.nan
        held = W != 0
        z_h = np.where(np.isnan(z), 0.0, z)
        keep_l = held & (W > 0) & (z_h > zout)
        keep_s = held & (W < 0) & (z_h < -zout)
        cand_l = np.argsort(-np.where(np.isnan(z), -np.inf, z))[:nmax*2]
        cand_s = np.argsort(np.where(np.isnan(z), np.inf, z))[:nmax*2]
        new_l = [c for c in cand_l if z[c] > zin][:nmax]
        new_s = [c for c in cand_s if z[c] < -zin][:nmax]
        tgt = np.zeros_like(W)
        longs = set(np.where(keep_l)[0]) | set(new_l)
        shorts = set(np.where(keep_s)[0]) | set(new_s)
        longs -= shorts
        if len(longs) >= 10 and len(shorts) >= 10:
            tgt[list(longs)] = 0.5/len(longs)
            tgt[list(shorts)] = -0.5/len(shorts)
        W = (1-lam)*W + lam*tgt
        rows[a] = W
    Wdf = pd.DataFrame(rows, index=idx[days], columns=R.columns)
    net, gross, tno = G.run(Wdf, R)
    rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END],
                   f"patient zin{zin} zout{zout} lam{lam}")
    print(G.fmt(rep), flush=True)
    return rep

res = []
for zin, zout, lam in [(1.5, 0.5, 1.0), (1.5, 0.5, 0.5), (1.5, 0.5, 0.25),
                       (2.0, 0.5, 0.5), (2.0, 1.0, 0.5), (1.0, 0.25, 0.25),
                       (1.5, 0.25, 0.33)]:
    res.append(run_patient(zin, zout, lam))
pd.DataFrame(res).to_csv("/tmp/sharpe3_work/exp06.csv", index=False)
print(f"exp06 done t={time.time()-t0:.0f}s", flush=True)
