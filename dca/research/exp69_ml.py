import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl")
FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index
fnames=list(FEAT.keys())
# cross-sectional rank-normalize each feature per month
Z={}
for nm,f in FEAT.items():
    fr=f.where(liq)
    Z[nm]=fr.rank(axis=1,pct=True)
# build long sample table
mtrain=M[(M>=pd.Timestamp("2011-06-01"))&(M<=pd.Timestamp("2025-12-31"))]
recs=[]
tgt_top=0.20
for dt in mtrain:
    fv=fok.loc[dt].dropna()
    if len(fv)<50: continue
    thr=fv.quantile(1-tgt_top)
    y=(fv>=thr).astype(int)
    row={"date":dt,"tk":fv.index}
    X=np.column_stack([Z[nm].loc[dt].reindex(fv.index).values for nm in fnames])
    for i,tk in enumerate(fv.index):
        recs.append((dt,tk,*X[i],y[tk]))
ccols=["date","tk"]+fnames+["y"]
DF=pd.DataFrame.from_records(recs,columns=ccols)
p(f"samples {len(DF)} feats {len(fnames)} t={time.time()-t0:.0f}s")
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception as e:
    p(f"no sklearn: {e}"); raise SystemExit
ret=(me/me.shift(1)-1).clip(-0.9,2.0)
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change())
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
# walk-forward by year
preds=[]
import warnings; warnings.filterwarnings("ignore")
for ytest in range(2015,2026):
    tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")]
    te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
    if len(te)==0 or len(tr)<5000: continue
    clf=HistGradientBoostingClassifier(max_iter=200,max_depth=4,learning_rate=0.05,
         l2_regularization=1.0,min_samples_leaf=200)
    clf.fit(tr[fnames].values,tr["y"].values)
    pr=clf.predict_proba(te[fnames].values)[:,1]
    t2=te[["date","tk"]].copy(); t2["p"]=pr; preds.append(t2)
    p(f"  trained thru {ytest-1}, tested {ytest} (tr={len(tr)}) t={time.time()-t0:.0f}s")
PR=pd.concat(preds)
# build top-N portfolio from model probs
idx=sorted(PR.date.unique())
def ml_port(N):
    rr=[]
    for dt in idx:
        d=PR[PR.date==dt].nlargest(N,"p")
        k=M.get_loc(dt)
        if k+1>=len(M): continue
        nr=ret.iloc[k+1][list(d.tk)].dropna()
        rr.append((M[k+1],nr.mean() if len(nr) else 0.0))
    return pd.Series(dict(rr))
p(f"\nWalk-forward ML portfolio (2015-2025):")
qsub=qret.reindex(pd.DatetimeIndex(idx)+pd.offsets.MonthEnd(0)).reindex([M[M.get_loc(d)+1] for d in idx if M.get_loc(d)+1<len(M)])
for N in [15,25,40]:
    r=ml_port(N); c,s,d=stats(r)
    qq=qret.reindex(r.index); qc,qs,qd=stats(qq)
    p(f"  top{N}: ENS CAGR {c:.1%} Sharpe {s:.2f} maxDD {d:.1%}  | QQQ {qc:.1%}/{qs:.2f}")
# feature importance via permutation on last model (approx: use built-in via refit on all)
clf=HistGradientBoostingClassifier(max_iter=200,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200)
clf.fit(DF[fnames].values,DF["y"].values)
from sklearn.inspection import permutation_importance
samp=DF.sample(min(40000,len(DF)),random_state=0)
pi=permutation_importance(clf,samp[fnames].values,samp["y"].values,n_repeats=3,random_state=0,n_jobs=2)
imp=sorted(zip(fnames,pi.importances_mean),key=lambda x:-x[1])
p(f"\nPermutation feature importance (top 20):")
for nm,v in imp[:20]: p(f"  {nm:20} {v:.5f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
