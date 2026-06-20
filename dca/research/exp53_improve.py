import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P["ym"]=pd.to_datetime(P.ym)
names=[c for c in me.columns if c in set(P.tk)]; P=P[P.tk.isin(set(names))]
def pan(col): return P.pivot_table(index="ym",columns="tk",values=col,aggfunc="sum").reindex(index=me.index,columns=names).fillna(0)
buy,sell,nb,offb,ceob=pan("buy"),pan("sell"),pan("nbuyers"),pan("off_buy"),pan("ceo_buy")
nb3=nb.rolling(3,min_periods=1).sum(); buy3=buy.rolling(3,min_periods=1).sum()
off3=offb.rolling(3,min_periods=1).sum(); ceo3=ceob.rolling(3,min_periods=1).sum()
bigthr=buy3.where(buy3>0).quantile(0.7,axis=1)
mp=me[names]; ret=(mp/mp.shift(1)-1).clip(-0.90,2.0)
px_ok=mp.shift(1)>=3.0

# conviction score (standardized within month): cluster size, $ size, officer/CEO
def z(x):
    m=x.where(x>0); return (m.sub(m.mean(axis=1),axis=0)).div(m.std(axis=1).replace(0,np.nan),axis=0)
score=( z(nb3).fillna(0) + z(buy3).fillna(0) + 0.5*(off3>0).astype(float) + 0.5*(ceo3>0).astype(float) )
anybuy=(buy3>0)&px_ok
score=score.where(anybuy)

start=pd.Timestamp("2011-01-01"); end=pd.Timestamp("2025-12-31")
def stats(r):
    r=r.dropna()
    cagr=(1+r).prod()**(12/len(r))-1; sh=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return cagr,sh,dd
qret=(me["QQQ"]/me["QQQ"].shift(1)-1)
iwm=(me["IWM"]/me["IWM"].shift(1)-1) if "IWM" in me else None

def topN_ret(N):
    rk=score.rank(axis=1,ascending=False)
    sel=(rk<=N)
    w=sel.shift(1).fillna(False).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w*ret).sum(axis=1)

print("Concentration sweep (long-only, equal-wt top-N by insider conviction, monthly rebal):")
print(f"{'topN':>6} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
idx0=me.index[(me.index>=start)&(me.index<=end)]
for N in [10,20,30,50,100,300,99999]:
    r=topN_ret(N).reindex(idx0)
    c,s,d=stats(r); print(f"{N:>6} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
c,s,d=stats(qret.reindex(idx0)); print(f"{'QQQ':>6} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
if iwm is not None:
    c,s,d=stats(iwm.reindex(idx0)); print(f"{'IWM':>6} {c:>7.1%} {s:>7.2f} {d:>7.1%}")

# the edge as a spread: insider top-decile minus small-cap peers (long/short, market neutral-ish)
rk=score.rank(axis=1,ascending=False,pct=True)
longs=(rk<=0.20); shorts=anybuy&(~longs)  # within the buy universe, top vs rest
wl=longs.shift(1).fillna(False).astype(float); wl=wl.div(wl.sum(axis=1).replace(0,np.nan),axis=0)
ws=shorts.shift(1).fillna(False).astype(float); ws=ws.div(ws.sum(axis=1).replace(0,np.nan),axis=0)
ls=( (wl*ret).sum(axis=1) - (ws*ret).sum(axis=1) ).reindex(idx0)
c,s,d=stats(ls); print(f"\nL/S top20% vs rest-of-buyers: CAGR {c:.1%} Sharpe {s:.2f} maxDD {d:.1%}")
print(f"t={time.time()-t0:.0f}s")
