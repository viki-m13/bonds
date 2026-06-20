import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; fnames=list(FEAT.keys())
Z={nm:FEAT[nm].where(liq).rank(axis=1,pct=True) for nm in fnames}
recs=[]
for dt in M[(M>=pd.Timestamp("2011-06-01"))]:
    fv=fok.loc[dt].dropna()
    if len(fv)<60: continue
    q1,q2=fv.quantile(1/3),fv.quantile(2/3)
    y=pd.Series(np.where(fv>=q2,1,np.where(fv<=q1,0,np.nan)),index=fv.index)
    X=np.column_stack([Z[nm].loc[dt].reindex(fv.index).values for nm in fnames])
    for i,tk in enumerate(fv.index): recs.append((dt,tk,*X[i],y.iloc[i]))
DF=pd.DataFrame.from_records(recs,columns=["date","tk"]+fnames+["y"]).fillna(0.5)
p(f"table {DF.shape} t={time.time()-t0:.0f}s")
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
def wf(builder,name):
    preds=[]
    for ytest in range(2015,2026):
        tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")].dropna(subset=["y"]); te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
        if len(te)==0 or len(tr)<5000: continue
        clf=builder(); clf.fit(tr[fnames].values,tr["y"].astype(int).values)
        t2=te[["date","tk"]].copy(); t2["p"]=clf.predict_proba(te[fnames].values)[:,1]; preds.append(t2)
    P=pd.concat(preds).pivot_table(index="date",columns="tk",values="p").reindex(M)
    p(f"  trained {name} t={time.time()-t0:.0f}s"); return P
# models
P_gbt=wf(lambda:HistGradientBoostingClassifier(max_iter=300,max_depth=4,learning_rate=0.04,l2_regularization=1.0,min_samples_leaf=150,random_state=0),"GBT-tuned")
P_gbt2=wf(lambda:HistGradientBoostingClassifier(max_iter=400,max_depth=6,learning_rate=0.03,l2_regularization=3.0,min_samples_leaf=300,random_state=0),"GBT-deep")
P_et=wf(lambda:ExtraTreesClassifier(n_estimators=200,max_depth=12,min_samples_leaf=100,n_jobs=4,random_state=0),"ExtraTrees")
P_lr=wf(lambda:LogisticRegression(C=0.5,max_iter=200),"Logistic")
P_old=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
# rank-average ensemble (per-date pct rank then mean)
def rankavg(Ps):
    rs=[Pi.rank(axis=1,pct=True) for Pi in Ps]
    return sum(rs)/len(rs)
ENS=rankavg([P_gbt,P_et,P_lr])
ENS4=rankavg([P_gbt,P_gbt2,P_et,P_lr])
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qret=qpx.pct_change().reindex(idx)
fwd3=(me.shift(-3)/me-1).clip(-0.9,3.0)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def ic(P):
    fr=P.where(liq); v=[]
    for dt in idx:
        if dt not in P.index: continue
        d=pd.concat([fr.loc[dt],fwd3.loc[dt]],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: v.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(v)
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
p(f"\n{'model':28} {'IC3m':>7} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
c,s,d=stats(qret); p(f"{'QQQ':28} {'':>7} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
for nm,P in [("old single GBT (current)",P_old),("GBT-tuned",P_gbt),("GBT-deep",P_gbt2),("ExtraTrees",P_et),("Logistic",P_lr),("ENS gbt+et+lr",ENS),("ENS 4-model",ENS4)]:
    r=champ(P); c,s,d=stats(r); p(f"{nm:28} {ic(P):>7.3f} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
# corr of model errors (diversity)
p(f"\nmodel pred corr (diversity): gbt-et {P_gbt.rank(axis=1,pct=True).corrwith(P_et.rank(axis=1,pct=True)).mean():.2f}  gbt-lr {P_gbt.rank(axis=1,pct=True).corrwith(P_lr.rank(axis=1,pct=True)).mean():.2f}  et-lr {P_et.rank(axis=1,pct=True).corrwith(P_lr.rank(axis=1,pct=True)).mean():.2f}")
best=rankavg([P_gbt,P_et,P_lr])
pd.to_pickle(best,"/tmp/wave/_mlprob_multiens.pkl")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); r=champ(ENS); c,s,_=stats(r[m]); qc,qs,_=stats(qret[m]); p(f"  ENS {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
