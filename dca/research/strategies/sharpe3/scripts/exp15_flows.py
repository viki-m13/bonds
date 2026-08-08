"""Modern-regime probes round 2 (DEV2 = 2005-2019, TEST 2020+ locked):
flow and calendar mechanics rather than price reversal.

1. turn-of-month: long the eligible universe (equal-weight) days T-1..T+3 of
   each month, hedged short SPY (does the TOM premium exist cross-sectionally
   beyond the index?). Also high-beta-minus-low-beta TOM spread.
2. rank-crossers: names whose 63d dollar-volume rank crosses INTO the top 500
   this month (fund-inclusion proxy) — long 1 month vs SPY.
3. peer-gap reversion: z of (own 10d return - top-20-corr peer basket 10d
   return), weekly, proportional, patient lam 0.25 — the last untried
   residualization flavor.
NOTE: hedged books here execute at the signal close (lag=1) — an OPTIMISTIC
bound; failures at an optimistic bound are strong failures.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
idx = R.index
A2, B2 = pd.Timestamp("2005-01-01"), pd.Timestamp("2019-12-31")
E_d = ELIG.reindex(idx).ffill().fillna(False)[R.columns].astype(bool)
spy_col = R.columns.get_loc("SPY")
mask2 = (idx >= A2) & (idx <= B2)

# ---------- 1) turn-of-month ----------
month = pd.Series(idx.month, index=idx)
nxt = month.shift(-1).fillna(0)
is_last = (month != nxt)
tom = np.zeros(len(idx), dtype=bool)
li = np.where(is_last.values)[0]
for i0 in li:
    tom[max(i0-1, 0):i0+4] = True
beta = R.rolling(252, min_periods=150).cov(R["SPY"]).div(R["SPY"].rolling(252, min_periods=150).var(), axis=0)
Wv = np.zeros((len(idx), len(R.columns)))
Wb = np.zeros((len(idx), len(R.columns)))
Ev = E_d.values
bv = beta.values
for ti in range(len(idx)):
    if not (mask2[ti] and tom[ti]): continue
    names = np.where(Ev[ti])[0]
    if len(names) < 100: continue
    Wv[ti, names] = 0.5/len(names)
    Wv[ti, spy_col] = -0.5
    b = bv[ti, names]
    fin = np.isfinite(b)
    nb = names[fin]; b = b[fin]
    hi = nb[b >= np.percentile(b, 70)]; lo = nb[b <= np.percentile(b, 30)]
    if len(hi) and len(lo):
        Wb[ti, hi] = 0.5/len(hi); Wb[ti, lo] = -0.5/len(lo)
for nm, Wx in [("TOM univ vs SPY", Wv), ("TOM hi-beta vs lo-beta", Wb)]:
    W = pd.DataFrame(Wx, index=idx, columns=R.columns)
    line = f"{nm:24}"
    for fee in (0, 5, 10):
        net, g_, t_ = G.run(W, R, fee_bps=fee, lag=1)
        line += f"  {fee}bp {G.sharpe(net[A2:B2]):5.2f}"
    print(line + f"  (tno {t_[A2:B2].mean():.3f})", flush=True)

# ---------- 2) rank-crossers ----------
DV63 = DV.rolling(63, min_periods=40).median()
rk = DV63.rank(axis=1, ascending=False)
mo = [d for d in G.month_ends(idx) if A2 < d <= B2]
Wv = np.zeros((len(idx), len(R.columns)))
rkv = rk.values
for mi, d in enumerate(mo[1:], 1):
    ti = idx.get_loc(d)
    tip = idx.get_loc(mo[mi-1])
    cross = np.where((rkv[ti] <= 500) & (rkv[tip] > 550) & np.isfinite(rkv[tip]))[0]
    if len(cross) == 0: continue
    tj = idx.get_loc(mo[mi+1]) if mi+1 < len(mo) else min(ti+22, len(idx)-1)
    for ni in cross:
        Wv[ti:tj, ni] += 1.0
g = np.abs(Wv).sum(axis=1); g[g == 0] = 1.0
Wn = 0.5*Wv/g[:, None]
Wn[:, spy_col] = -0.5*(np.abs(Wn).sum(axis=1) > 1e-9)
W = pd.DataFrame(Wn, index=idx, columns=R.columns)
line = "rank-crossers 1mo       "
for fee in (0, 5, 10):
    net, g_, t_ = G.run(W, R, fee_bps=fee, lag=1)
    line += f"  {fee}bp {G.sharpe(net[A2:B2]):5.2f}"
print(line + f"  (tno {t_[A2:B2].mean():.3f})", flush=True)

# ---------- 3) peer-gap reversion ----------
L = np.log1p(R)
r10 = np.expm1(L.rolling(10).sum())
wk = [d for d in G.week_ends(idx) if A2 < d <= B2]
E_wk = G.elig_on(wk, ELIG)
qtr = [d for d in G.month_ends(idx) if A2 < d <= B2][::3]
peer_cache = {}
def peers(q_):
    if q_ not in peer_cache:
        i = idx.get_loc(q_)
        e = E_d.iloc[i]; names = list(np.where(e.values)[0])
        Rw = R.iloc[i-252:i+1].iloc[:, names]
        goodm = Rw.notna().sum() > 230
        gnames = [names[k] for k in range(len(names)) if goodm.iloc[k]]
        Rw = R.iloc[i-252:i+1].iloc[:, gnames].fillna(0.0)
        C = np.corrcoef(Rw.values.T); np.fill_diagonal(C, -1)
        topk = np.argsort(-C, axis=1)[:, :20]
        peer_cache[q_] = (gnames, topk)
    return peer_cache[q_]
rows = []
r10v = r10.values
for d in wk:
    qs = [q_ for q_ in qtr if q_ <= d]
    if not qs: continue
    gnames, topk = peers(qs[-1])
    ti = idx.get_loc(d)
    r = r10v[ti, gnames]
    r = np.where(np.isfinite(r), r, 0.0)
    gap = r - r[topk].mean(axis=1)
    z = (gap - gap.mean())/(gap.std()+1e-12)
    s = pd.Series(-z, index=[R.columns[k] for k in gnames]).clip(-3, 3)
    pos = s[s > 0]; neg = s[s < 0]
    w = pd.concat([0.5*pos/pos.sum(), 0.5*neg/(-neg).sum()]); w.name = d
    rows.append(w)
Wt = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
# patient partial adjustment on weekly grid
Tv = Wt.values; Wv2 = np.zeros_like(Tv); cur = np.zeros(Tv.shape[1])
for k in range(len(Tv)):
    cur = cur + 0.25*(Tv[k]-cur); Wv2[k] = cur
Wt = pd.DataFrame(Wv2, index=Wt.index, columns=Wt.columns)
line = "peer-gap wk lam.25      "
for fee in (0, 5, 10):
    net, g_, t_ = G.run(Wt, R, fee_bps=fee)
    line += f"  {fee}bp {G.sharpe(net[A2:B2]):5.2f}"
print(line + f"  (tno {t_[A2:B2].mean():.3f}, gross {G.sharpe(g_[A2:B2]):.2f})", flush=True)
print(f"exp15 done t={time.time()-t0:.0f}s", flush=True)
