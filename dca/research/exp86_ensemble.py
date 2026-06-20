import numpy as np, pandas as pd, time, warnings, re
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
# add 13F breadth-change feature
try:
    C=pd.read_pickle("/home/user/bonds/dca/research/data/sec/_13f_cusip.pkl"); cmap=pd.read_pickle("/home/user/bonds/dca/research/data/sec/_13f_cusipmap.pkl")
    nmgr=C["nmgr"]; mo={'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    def ld(l):
        m=re.search(r'-(\d{2})([a-z]{3})(\d{4})',l)
        if m: end=pd.Timestamp(int(m.group(3)),mo[m.group(2)],int(m.group(1)))
        else: mm=re.search(r'(\d{4})q([1-4])',l); end=pd.Timestamp(int(mm.group(1)),int(mm.group(2))*3,1)+pd.offsets.MonthEnd(0)
        return (end+pd.DateOffset(days=45)).to_period("M").to_timestamp()
    nm=nmgr.copy(); nm["tk"]=[cmap.get(c) for c in nm.index]; nm=nm[nm.tk.notna()].groupby("tk").sum(numeric_only=True)
    nm.columns=[ld(c) for c in nm.columns]; nm=nm.sort_index(axis=1)
    p("(13F breadth omitted from ML — all-NaN pre-2023 breaks binning)")
except Exception as e: p(f"13F feat skip {e}")
fnames=list(FEAT.keys())
Z={nm:FEAT[nm].where(liq).rank(axis=1,pct=True) for nm in fnames}
# sample table with multiple targets
recs=[]
for dt in M[(M>=pd.Timestamp("2011-06-01"))]:
    fv=fok.loc[dt].dropna()
    if len(fv)<60: continue
    q1,q2=fv.quantile(1/3),fv.quantile(2/3); q9=fv.quantile(0.9)
    y_terc=pd.Series(np.where(fv>=q2,1,np.where(fv<=q1,0,np.nan)),index=fv.index)
    y_dec=(fv>=q9).astype(int)
    X=np.column_stack([Z[nm].loc[dt].reindex(fv.index).values for nm in fnames])
    for i,tk in enumerate(fv.index): recs.append((dt,tk,*X[i],y_terc.iloc[i],int(y_dec.iloc[i])))
DF=pd.DataFrame.from_records(recs,columns=["date","tk"]+fnames+["yt","yd"])
from sklearn.ensemble import HistGradientBoostingClassifier
members=[dict(target="yt",seed=0,depth=4),dict(target="yt",seed=1,depth=3),dict(target="yd",seed=2,depth=4),
         dict(target="yd",seed=3,depth=5),dict(target="yt",seed=4,depth=5)]
preds=[]
for ytest in range(2015,2026):
    te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
    if len(te)==0: continue
    probs=np.zeros(len(te))
    for mb in members:
        tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")].dropna(subset=[mb["target"]])
        if len(tr)<5000: continue
        clf=HistGradientBoostingClassifier(max_iter=200,max_depth=mb["depth"],learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200,random_state=mb["seed"])
        clf.fit(tr[fnames].values,tr[mb["target"]].astype(int).values)
        probs+=clf.predict_proba(te[fnames].values)[:,1]
    t2=te[["date","tk"]].copy(); t2["p"]=probs/len(members); preds.append(t2)
    p(f"  ensemble {ytest} t={time.time()-t0:.0f}s")
PROBe=pd.concat(preds).pivot_table(index="date",columns="tk",values="p").reindex(M)
pd.to_pickle(PROBe,"/tmp/wave/_mlprob_ens.pkl")
# evaluate: champion sim with ensemble vs single
PROB1=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qret=qpx.pct_change().reindex(idx)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def champ(P,N=12):
    accel=P-P.shift(2); el=(liq&(me>=3.0)&(me>ma10)&(mom3>0)&(accel>0)).fillna(False).astype(bool)
    sc=P.where(el); rank=sc.rank(axis=1,ascending=False)
    pos={}; cash=1.0; out=[]
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
# IC comparison
def avgic(P):
    fok3=(me.shift(-3)/me-1).clip(-0.9,3.0).where(liq); ics=[]
    for dt in idx[::1]:
        a=P.loc[dt] if dt in P.index else None
        if a is None: continue
        d=pd.concat([a,fok3.loc[dt]],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: ics.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(ics)
p(f"\nIC(fwd3m): single {avgic(PROB1):.3f}  ensemble {avgic(PROBe):.3f}")
c,s,d=stats(qret); p(f"\n{'QQQ':28} {c:>6.1%} {s:>5.2f} {d:>6.1%}")
r1=champ(PROB1); c,s,d=stats(r1); p(f"{'champion (single ML)':28} {c:>6.1%} {s:>5.2f} {d:>6.1%}")
re_=champ(PROBe); c,s,d=stats(re_); p(f"{'champion (ENSEMBLE ML)':28} {c:>6.1%} {s:>5.2f} {d:>6.1%}")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(re_[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
