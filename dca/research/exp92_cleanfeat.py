import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=dict(D["FEAT"]),D["fok"],D["liq"],D["me"],D["cols"]
M=me.index
base_names=list(FEAT.keys())
T=pd.read_pickle("/tmp/wave/_techfeat.pkl")
for k,df in T.items(): FEAT[k]=df.reindex(index=M,columns=cols)
full_names=list(FEAT.keys())
Z={nm:FEAT[nm].where(liq).rank(axis=1,pct=True) for nm in full_names}
ret=(me/me.shift(1)-1).clip(-0.9,3.0)
# WINNING target: fwd-3m tercile (top vs bottom)
def build(feat_names):
    recs=[]
    for dt in M[(M>=pd.Timestamp("2011-06-01"))]:
        fv=fok.loc[dt].dropna()
        if len(fv)<60: continue
        q1,q2=fv.quantile(1/3),fv.quantile(2/3)
        y=pd.Series(np.where(fv>=q2,1,np.where(fv<=q1,0,np.nan)),index=fv.index)
        X=np.column_stack([Z[nm].loc[dt].reindex(fv.index).values for nm in feat_names])
        for i,tk in enumerate(fv.index): recs.append((dt,tk,*X[i],y.iloc[i]))
    return pd.DataFrame.from_records(recs,columns=["date","tk"]+feat_names+["y"])
from sklearn.ensemble import HistGradientBoostingClassifier
def walkforward(DF,feat_names):
    preds=[]
    for ytest in range(2015,2026):
        tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")].dropna(subset=["y"]); te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
        if len(te)==0 or len(tr)<5000: continue
        clf=HistGradientBoostingClassifier(max_iter=200,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200,random_state=0)
        clf.fit(tr[feat_names].values,tr["y"].astype(int).values)
        t2=te[["date","tk"]].copy(); t2["p"]=clf.predict_proba(te[feat_names].values)[:,1]; preds.append(t2)
    return pd.concat(preds).pivot_table(index="date",columns="tk",values="p").reindex(M)
DFfull=build(full_names); p(f"built table {DFfull.shape} t={time.time()-t0:.0f}s")
PROBfull=walkforward(DFfull,full_names); p(f"trained 50-feat t={time.time()-t0:.0f}s")
PROBbase=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)   # original 36-feat fwd-3m model
ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qret=qpx.pct_change().reindex(idx)
fwd3=(me.shift(-3)/me-1).clip(-0.9,3.0)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def ic(P):
    fr=P.where(liq); ics=[]
    for dt in idx:
        if dt not in P.index: continue
        d=pd.concat([fr.loc[dt],fwd3.loc[dt]],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: ics.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(ics)
def champ(P,N=12):
    a=P-P.shift(2); el=(liq&(me>=3.0)&(me>ma10)&(mom3>0)&(a>0)).fillna(False).astype(bool)
    sc=P.where(el); rank=sc.rank(axis=1,ascending=False); pos={}; cash=1.0; out=[]
    for k,dt in enumerate(M):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx)
            if cpx/e["peak"]-1<=-0.30 or cpx<ma10.loc[dt].get(tk,np.nan): cash+=e["val"]; pos.pop(tk)
        if dt in P.index:
            rk=rank.loc[dt]; cands=[t for t in rk[rk<=N*4].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan))]
            need=N-len(pos)
            if need>0 and cash>1e-9 and cands:
                sl=cash/need
                for tk in cands[:need]: pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":sl}; cash-=sl
        eq0=cash+sum(e["val"] for e in pos.values())
        if k+1<len(M):
            for tk in pos:
                r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(M): out.append((M[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)
p(f"\nIC(fwd3m): base-36feat {ic(PROBbase):.3f}   +14 tech (50feat) {ic(PROBfull):.3f}")
c,s,d=stats(qret); p(f"\n{'QQQ':34}{c:>7.1%}{s:>7.2f}{d:>8.1%}")
r=champ(PROBbase); c,s,d=stats(r); p(f"{'champion 36-feat (current)':34}{c:>7.1%}{s:>7.2f}{d:>8.1%}")
r=champ(PROBfull); c,s,d=stats(r); p(f"{'champion +14 tech (50-feat)':34}{c:>7.1%}{s:>7.2f}{d:>8.1%}")
rb=champ(PROBfull)
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(rb[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
pd.to_pickle(PROBfull,"/tmp/wave/_mlprob_50feat.pkl")
p(f"\nDONE t={time.time()-t0:.0f}s")
