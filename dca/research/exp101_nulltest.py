import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M); fnames=list(FEAT.keys())
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qret=qpx.pct_change().reindex(idx); fwd3=(me.shift(-3)/me-1).clip(-0.9,3.0)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def champ(P,N=12,gate=True):
    a=P-P.shift(2); el=(liq&(me>=3.0)&(me>ma10)&((mom3>0)&(a>0) if gate else True)).fillna(False).astype(bool)
    sc=P.where(el); rank=sc.rank(axis=1,ascending=False); pos={}; cash=1.0; out=[]
    for k,dt in enumerate(didx):
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
        if k+1<len(didx):
            for tk in pos:
                r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx): out.append((didx[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)
def ic(P):
    fr=P.where(liq); v=[]
    for dt in idx:
        if dt not in P.index: continue
        d=pd.concat([fr.loc[dt],fwd3.loc[dt]],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: v.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(v)
# REAL champion
P_real=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
c,s,d=stats(champ(P_real)); p(f"REAL champion: IC {ic(P_real):+.3f}  CAGR {c:.1%} Sharpe {s:.2f}")
# NULL 1: random score through the SAME harness (gates+ride+cut) — does the harness manufacture alpha?
p(f"\nNULL-1: RANDOM score through champion harness (10 seeds):")
sh=[]
for seed in range(10):
    rng=np.random.default_rng(seed)
    Pr=pd.DataFrame(rng.random(me.shape),index=M,columns=cols)
    r=champ(Pr); c,s2,_=stats(r); sh.append(s2)
sh=np.array(sh); p(f"  random-score Sharpe: mean {sh.mean():+.2f}  std {sh.std():.2f}  max {sh.max():+.2f}  (vs QQQ {stats(qret)[1]:.2f}, real {stats(champ(P_real))[1]:.2f})")
# NULL 2: shuffle the TARGET, retrain ML, measure IC (should be ~0)
Z={nm:FEAT[nm].where(liq).rank(axis=1,pct=True) for nm in fnames}
from sklearn.ensemble import HistGradientBoostingClassifier
recs=[]
for dt in M[(M>=pd.Timestamp("2011-06-01"))]:
    fv=fok.loc[dt].dropna()
    if len(fv)<60: continue
    q1,q2=fv.quantile(1/3),fv.quantile(2/3); y=pd.Series(np.where(fv>=q2,1,np.where(fv<=q1,0,np.nan)),index=fv.index)
    X=np.column_stack([Z[nm].loc[dt].reindex(fv.index).values for nm in fnames])
    for i,tk in enumerate(fv.index): recs.append((dt,tk,*X[i],y.iloc[i]))
DF=pd.DataFrame.from_records(recs,columns=["date","tk"]+fnames+["y"]).dropna(subset=["y"])
rng=np.random.default_rng(0)
preds=[]
for ytest in range(2015,2026):
    tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")].copy(); te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
    if len(te)==0 or len(tr)<5000: continue
    yshuf=tr.groupby("date")["y"].transform(lambda s: rng.permutation(s.values))   # shuffle within month
    clf=HistGradientBoostingClassifier(max_iter=200,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200,random_state=0)
    clf.fit(tr[fnames].values,yshuf.values)
    t2=te[["date","tk"]].copy(); t2["p"]=clf.predict_proba(te[fnames].values)[:,1]; preds.append(t2)
Pnull=pd.concat(preds).pivot_table(index="date",columns="tk",values="p").reindex(M)
c,s,d=stats(champ(Pnull)); p(f"\nNULL-2: SHUFFLED-target ML: IC {ic(Pnull):+.3f}  CAGR {c:.1%} Sharpe {s:.2f}  (should be ~0 IC / ~random)")
p(f"\nVERDICT: harness clean if random-score Sharpe ~QQQ-ish & shuffled-target IC ~0, both far below real {ic(P_real):.3f}/{stats(champ(P_real))[1]:.2f}")
p(f"DONE t={time.time()-t0:.0f}s")
