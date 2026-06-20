import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=dict(D["FEAT"]),D["fok"],D["liq"],D["me"],D["cols"]
M=me.index
T=pd.read_pickle("/tmp/wave/_techfeat.pkl")
nadd=0
for k,df in T.items():
    FEAT[k]=df.reindex(index=M,columns=cols); nadd+=1
p(f"features: {len(FEAT)} (+{nadd} technical/volume) t={time.time()-t0:.0f}s")
fnames=list(FEAT.keys())
Z={nm:FEAT[nm].where(liq).rank(axis=1,pct=True) for nm in fnames}
ret=(me/me.shift(1)-1).clip(-0.9,3.0)
fwd6=(me.shift(-6)/me-1)                 # 6-month forward (catch runners early)
# "banger" target = top decile of fwd-6m each month
recs=[]
for dt in M[(M>=pd.Timestamp("2011-06-01"))]:
    fv=fwd6.where(liq).loc[dt].dropna()
    if len(fv)<60: continue
    y=(fv>=fv.quantile(0.90)).astype(int)
    X=np.column_stack([Z[nm].loc[dt].reindex(fv.index).values for nm in fnames])
    for i,tk in enumerate(fv.index): recs.append((dt,tk,*X[i],int(y.iloc[i])))
DF=pd.DataFrame.from_records(recs,columns=["date","tk"]+fnames+["y"])
p(f"samples {len(DF)} banger-rate {DF.y.mean():.1%} t={time.time()-t0:.0f}s")
from sklearn.ensemble import HistGradientBoostingClassifier
NBAG=6
preds=[]
for ytest in range(2015,2026):
    tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")]; te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
    if len(te)==0 or len(tr)<5000: continue
    pr=np.zeros(len(te))
    for b in range(NBAG):
        sub=tr.sample(frac=0.7,random_state=b)
        clf=HistGradientBoostingClassifier(max_iter=220,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200,random_state=b)
        clf.fit(sub[fnames].values,sub["y"].values)
        pr+=clf.predict_proba(te[fnames].values)[:,1]
    t2=te[["date","tk"]].copy(); t2["p"]=pr/NBAG; preds.append(t2)
    p(f"  bagged {ytest} t={time.time()-t0:.0f}s")
PROB=pd.concat(preds).pivot_table(index="date",columns="tk",values="p").reindex(M)
pd.to_pickle(PROB,"/tmp/wave/_mlprob_banger.pkl")
ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1; accel=PROB-PROB.shift(2)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qret=qpx.pct_change().reindex(idx)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def ic(P,fwd):
    fr=P.where(liq); ics=[]
    for dt in idx:
        if dt not in P.index: continue
        d=pd.concat([fr.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: ics.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(ics)
def champ(P,N=12,gate=True):
    a=P-P.shift(2); el=(liq&(me>=3.0)&(me>ma10)&((mom3>0)&(a>0) if gate else True)).fillna(False).astype(bool)
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
P_old=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
p(f"\nIC(fwd6m): old-single {ic(P_old,fwd6):.3f}  banger-ensemble {ic(PROB,fwd6):.3f}")
c,s,d=stats(qret); p(f"\n{'QQQ':34}{c:>7.1%}{s:>7.2f}{d:>8.1%}")
r=champ(P_old); c,s,d=stats(r); p(f"{'champion (old single ML)':34}{c:>7.1%}{s:>7.2f}{d:>8.1%}")
r=champ(PROB); c,s,d=stats(r); p(f"{'champion (BANGER ensemble+tech)':34}{c:>7.1%}{s:>7.2f}{d:>8.1%}")
rb=champ(PROB)
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(rb[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
# feature importance of the banger model (permutation, last fit on all)
clf=HistGradientBoostingClassifier(max_iter=220,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200,random_state=0)
clf.fit(DF[fnames].values,DF["y"].values)
from sklearn.inspection import permutation_importance
samp=DF.sample(min(40000,len(DF)),random_state=1)
pi=permutation_importance(clf,samp[fnames].values,samp["y"].values,n_repeats=2,random_state=0,n_jobs=2)
imp=sorted(zip(fnames,pi.importances_mean),key=lambda x:-x[1])[:15]
p(f"\nbanger-model top features:")
for nm,v in imp: p(f"  {nm:18} {v:.5f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
