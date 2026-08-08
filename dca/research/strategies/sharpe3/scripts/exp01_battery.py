"""Battery 1: classic cross-sectional signals, DEV period only (1995-2014).
Weekly rebalance unless noted. Long-short deciles, dollar-neutral, gross 1.
Purpose: map the baseline landscape before inventing anything.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
DEV_END = "2014-12-31"
idx = R.index
wk = [d for d in G.week_ends(idx) if pd.Timestamp("1994-06-30") < d <= pd.Timestamp(DEV_END)]
mo = [d for d in G.month_ends(idx) if pd.Timestamp("1994-06-30") < d <= pd.Timestamp(DEV_END)]
E_wk = G.elig_on(wk, ELIG); E_mo = G.elig_on(mo, ELIG)
print(f"weekly dates {len(wk)}, monthly {len(mo)}, cols {R.shape[1]}  t={time.time()-t0:.0f}s", flush=True)

# ---- signal matrices (computed once, full panel, PIT by construction) ----
r5 = PX.pct_change(5, fill_method=None)
r21 = PX.pct_change(21, fill_method=None)
r252_21 = PX.shift(21).pct_change(231, fill_method=None)      # 12-1 momentum
vol63 = R.rolling(63, min_periods=40).std()
volspike = DV.rolling(5).mean() / DV.rolling(63, min_periods=40).mean()
maxret21 = R.rolling(21).max()
print(f"signals ready t={time.time()-t0:.0f}s", flush=True)

def ls_weights(sigrow, eligrow, invert=False, q=0.1):
    s = sigrow[eligrow[eligrow].index.intersection(sigrow.index)].dropna()
    if invert: s = -s
    return G.normalize_ls(s, topq=q, botq=q)

def run_sig(name, SIG, dates, E, invert=False, q=0.1):
    rows = []
    for d in dates:
        w = ls_weights(SIG.loc[:d].iloc[-1], E.loc[d], invert=invert, q=q)
        rows.append(pd.Series(w, name=d))
    W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
    net, gross, tno = G.run(W, R)
    net, gross, tno = net[:DEV_END], gross[:DEV_END], tno[:DEV_END]
    rep = G.report(net, gross, tno, name)
    print(G.fmt(rep), flush=True)
    return rep

res = []
res.append(run_sig("rev5 weekly (long losers)", r5, wk, E_wk, invert=True))
res.append(run_sig("rev21 monthly (long losers)", r21, mo, E_mo, invert=True))
res.append(run_sig("mom 12-1 monthly", r252_21, mo, E_mo))
res.append(run_sig("lowvol weekly (long calm)", vol63, wk, E_wk, invert=True))
res.append(run_sig("short lottery (maxret21) mo", maxret21, mo, E_mo, invert=True))
res.append(run_sig("volspike5 weekly (long quiet)", volspike, wk, E_wk, invert=True))

# 1-day reversal, daily rebalance (cost stress test)
dl = [d for d in idx if pd.Timestamp("1994-06-30") < d <= pd.Timestamp(DEV_END)]
dl = dl[::1]
E_dl = G.elig_on(dl[::21], ELIG)  # refresh eligibility monthly to save memory
E_dl = E_dl.reindex(pd.DatetimeIndex(dl)).ffill().fillna(False)
rows = []
r1 = R
for d in dl:
    w = ls_weights(r1.loc[d], E_dl.loc[d], invert=True)
    rows.append(pd.Series(w, name=d))
W = pd.DataFrame(rows).reindex(columns=R.columns).fillna(0.0)
net, gross, tno = G.run(W, R)
rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], "rev1 DAILY (long losers)")
print(G.fmt(rep), flush=True)
res.append(rep)

pd.DataFrame(res).to_csv("/tmp/sharpe3_work/battery1.csv", index=False)
print(f"battery1 done t={time.time()-t0:.0f}s", flush=True)
