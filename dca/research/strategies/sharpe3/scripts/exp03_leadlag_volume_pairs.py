"""Invention track B: three more independent families, DEV only.

1. LEAD-LAG (Lo-MacKinlay): mega-cap composite return this week predicts
   smaller liquid names next week. Long laggards whose 'leader' rose.
2. HIGH-VOLUME PREMIUM (Gervais): abnormal-volume stocks earn more next weeks.
3. PAIRS-LITE: within trailing-corr top pairs, trade the spread z-score,
   weekly, dollar-neutral (a light Gatev variant on liquid names).
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
r5 = PX.pct_change(5, fill_method=None)

# ---------- 1) lead-lag ----------
# leader = top-100 by dollar volume (PIT, monthly); follower = the rest of eligibles.
DV63 = DV.rolling(63, min_periods=40).median()
res1 = []
for corr_w in [126]:
    rows = []
    for d in wk:
        i = idx.get_loc(d)
        if i < 200: continue
        e = E_wk.loc[d]; names = e[e].index
        dv = DV63.loc[:d].iloc[-1][names].dropna()
        leaders = dv.sort_values(ascending=False).head(100).index
        followers = dv.index.difference(leaders)
        lead_ret = r5.loc[d, leaders].mean()
        # follower signal: leader move minus own move (catch-up expectation)
        own = r5.loc[d, followers]
        sig = (lead_ret - own).dropna()
        z = (sig - sig.mean())/(sig.std()+1e-12)
        z = z.clip(-3,3)
        pos = z[z>0]; neg = z[z<0]
        if len(pos)==0 or len(neg)==0: continue
        w = pd.concat([0.5*pos/pos.sum(), 0.5*neg/(-neg).sum()]); w.name = d
        rows.append(w)
    W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
    net, gross, tno = G.run(W, R)
    rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], f"leadlag catchup w{corr_w}")
    print(G.fmt(rep), flush=True); res1.append(rep)

# ---------- 2) high-volume premium ----------
volratio = DV.rolling(5).mean() / DV.rolling(63, min_periods=40).mean()
rows = []
for d in wk:
    e = E_wk.loc[d]; names = e[e].index
    s = volratio.loc[:d].iloc[-1][names].dropna()
    z = (s - s.mean())/(s.std()+1e-12)
    w = pd.Series(G.normalize_ls(z, 0.1, 0.1)); w.name = d   # long high-volume, short low
    rows.append(w)
W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
net, gross, tno = G.run(W, R)
rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], "highvolume premium wk")
print(G.fmt(rep), flush=True)

# ---------- 3) pairs-lite ----------
# monthly: find top-300 correlated pairs among top-400 dollar-volume names on
# trailing 252d; during the month, weekly z of 20d spread; trade |z|>2 -> converge.
mo = [d for d in G.month_ends(idx) if pd.Timestamp("1995-01-01") < d <= DEV_END]
pairs_by_month = {}
for d in mo:
    i = idx.get_loc(d)
    if i < 300: continue
    e = G.elig_on([d], ELIG).iloc[0]; names = e[e].index
    dv = DV63.loc[:d].iloc[-1][names].dropna().sort_values(ascending=False)
    top = dv.head(400).index
    Rw = R.iloc[i-252:i+1][top]
    good = Rw.columns[Rw.notna().sum() > 230]
    Rw = Rw[good].fillna(0.0)
    C = np.corrcoef(Rw.values.T)
    n = len(good)
    iu = np.triu_indices(n, 1)
    flat = C[iu]
    kbest = np.argsort(flat)[-300:]
    pairs_by_month[d] = [(good[iu[0][k]], good[iu[1][k]]) for k in kbest]
logpx = np.log(PX)
rows = []
cur_pairs = None
for d in wk:
    pm = [m for m in pairs_by_month if m <= d]
    if not pm: continue
    cur_pairs = pairs_by_month[max(pm)]
    i = idx.get_loc(d)
    win = logpx.iloc[i-20:i+1]
    w = {}
    for a, b in cur_pairs:
        try:
            sp = win[a] - win[b]
        except KeyError: continue
        if sp.isna().any(): continue
        z = (sp.iloc[-1] - sp.mean())/(sp.std()+1e-12)
        if z > 2:   w[a] = w.get(a,0)-1; w[b] = w.get(b,0)+1
        elif z < -2: w[a] = w.get(a,0)+1; w[b] = w.get(b,0)-1
    if not w: continue
    s = pd.Series(w, dtype=float)
    gsum = s.abs().sum()
    s = s/gsum
    s.name = d
    rows.append(s)
W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
net, gross, tno = G.run(W, R)
rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], "pairs-lite z2 wk")
print(G.fmt(rep), flush=True)
print(f"exp03 done t={time.time()-t0:.0f}s", flush=True)
