import glob, os, numpy as np, pandas as pd, time
t0=time.time()
# rebuild monthly cache from ALL chunks (resample per-chunk then concat = fast)
mes=[]
for f in sorted(glob.glob("/home/user/bonds/dca/research/data/tiingo/prices/ac_*.parquet")):
    d=pd.read_parquet(f)
    if d.shape[1]==0: continue
    d.index=pd.to_datetime(d.index); mes.append(d.resample("ME").last())
me=pd.concat(mes,axis=1); me=me.loc[:,~me.columns.duplicated()]; me.index=me.index.to_period("M").to_timestamp()
me.to_pickle("/tmp/wave/_tiingo_me.pkl")
print(f"monthly panel {me.shape} t={time.time()-t0:.0f}s",flush=True)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet"); uni["endDate"]=pd.to_datetime(uni.endDate,errors="coerce")
deli=set(uni[(uni.assetType=="Stock")&(uni.endDate<'2025-01-01')].ticker)
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P["ym"]=pd.to_datetime(P.ym)
names=[c for c in me.columns if c in set(P.tk)]
P=P[P.tk.isin(set(names))]                      # FILTER FIRST -> fast pivots
me=me[names]
def pan(col): return P.pivot_table(index="ym",columns="tk",values=col,aggfunc="sum").reindex(index=me.index,columns=names).fillna(0)
buy,sell,nb,offb,ceob=pan("buy"),pan("sell"),pan("nbuyers"),pan("off_buy"),pan("ceo_buy")
b3=(buy-sell).rolling(3,min_periods=1).sum(); buy3=buy.rolling(3,min_periods=1).sum()
nb3=nb.rolling(3,min_periods=1).sum(); off3=offb.rolling(3,min_periods=1).sum(); ceo3=ceob.rolling(3,min_periods=1).sum()
bigthr=buy3.where(buy3>0).quantile(0.7,axis=1)
mef=me.ffill(limit=3); fwd=(mef.shift(-3)/me-1).clip(-0.95,2.0); fok=fwd.where(me>=3.0)
mask=(fok.index>=pd.Timestamp("2011-01-01"))&(fok.index<pd.Timestamp("2025-01-01"))
print(f"insider names priced: {len(names)} (delisted {len([n for n in names if n in deli])}) t={time.time()-t0:.0f}s",flush=True)
def edge(flag, restrict=None):
    f2=fok if restrict is None else fok.where(pd.DataFrame({c:[c in restrict]*len(fok) for c in names},index=fok.index))
    bm=fok.where(flag).mean(axis=1); rm=fok.where(~flag).mean(axis=1)
    nbuy=flag.where(fok.notna()).sum(axis=1); nrest=(~flag).where(fok.notna()).sum(axis=1)
    ok=mask&(nbuy>=3)&(nrest>=10); d=(bm-rm)[ok].dropna()
    return d.mean()*100, d.mean()/(d.std()+1e-9)*np.sqrt(len(d)), nbuy[ok].mean()
SIG={"net-buyer":(b3>0),"officer-buy":(off3>0),"CEO/CFO-buy":(ceo3>0),"cluster>=2":(nb3>=2),
     "cluster>=3":(nb3>=3),"large-$ buy":buy3.gt(bigthr,axis=0),
     "officer+cluster2":((off3>0)&(nb3>=2)),"CEO/CFO+large$":((ceo3>0)&buy3.gt(bigthr,axis=0))}
print(f"\n{'signal':18s} edge%/3m   t    avg#",flush=True)
for nm,flag in SIG.items():
    e,tt,n=edge(flag); print(f"{nm:18s} {e:+6.2f}  {tt:+5.1f}  {n:.0f}",flush=True)
# survivorship check on the best (officer-buy): incl-delisted vs survivors-only
print("\nSurvivorship check (officer-buy):",flush=True)
fo=(off3>0)
e1,t1,_=edge(fo); print(f"  ALL incl delisted: {e1:+.2f}%/3m t={t1:+.1f}",flush=True)
surv=[n for n in names if n not in deli]; fok2=fok[surv]; me2=me[surv]
def edge_sub(flag,cols):
    fk=fok[cols]; fl=flag[cols]
    bm=fk.where(fl).mean(axis=1); rm=fk.where(~fl).mean(axis=1)
    nbuy=fl.where(fk.notna()).sum(axis=1); nrest=(~fl).where(fk.notna()).sum(axis=1)
    ok=mask&(nbuy>=3)&(nrest>=10); d=(bm-rm)[ok].dropna(); return d.mean()*100,d.mean()/(d.std()+1e-9)*np.sqrt(len(d))
e2,t2=edge_sub(fo,surv); print(f"  survivors only:    {e2:+.2f}%/3m t={t2:+.1f}",flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s",flush=True)
