import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; fnames=list(FEAT.keys())
ret=(me/me.shift(1)-1).clip(-0.9,2.0); fwd1=ret.shift(-1)
Z={nm:FEAT[nm].where(liq).rank(axis=1,pct=True) for nm in fnames}
recs=[]
for dt in M[(M>=pd.Timestamp("2011-06-01"))]:
    fv=fok.loc[dt].dropna()
    if len(fv)<60: continue
    # target = cross-sectional tercile (top=1, bottom=0) for a cleaner L/S signal
    q1,q2=fv.quantile(1/3),fv.quantile(2/3)
    y=pd.Series(np.where(fv>=q2,1,np.where(fv<=q1,0,np.nan)),index=fv.index).dropna()
    X=np.column_stack([Z[nm].loc[dt].reindex(y.index).values for nm in fnames])
    for i,tk in enumerate(y.index): recs.append((dt,tk,*X[i],int(y.iloc[i])))
DF=pd.DataFrame.from_records(recs,columns=["date","tk"]+fnames+["y"])
from sklearn.ensemble import HistGradientBoostingClassifier
preds=[]
for ytest in range(2015,2026):
    tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")]; te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
    if len(te)==0 or len(tr)<5000: continue
    clf=HistGradientBoostingClassifier(max_iter=250,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200)
    clf.fit(tr[fnames].values,tr["y"].values)
    t2=te[["date","tk"]].copy(); t2["p"]=clf.predict_proba(te[fnames].values)[:,1]; preds.append(t2)
PR=pd.concat(preds); PROB=PR.pivot_table(index="date",columns="tk",values="p").reindex(M)
p(f"ML trained t={time.time()-t0:.0f}s")
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
LIQ=(liq&(me>=3.0)); LIQ5=(liq&(me>=5.0))
def ann(r):
    r=r.dropna(); a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
def mlls(q=0.1,borrow=0.0):
    pr=PROB.where(LIQ); rk=pr.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    prs=PROB.where(LIQ5); rks=prs.rank(axis=1,pct=True)
    sw=(rks<=q).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    lr=(lw*fwd1).sum(axis=1).reindex(idx); sr=(sw*fwd1).sum(axis=1).reindex(idx)
    return lr-sr-borrow/12, lr
p(f"{'ML market-neutral L/S':40} {'ann%':>7} {'Sharpe':>7} {'maxDD':>7} {'corrQQQ':>8}")
for q,bc,lab in [(0.1,0.0,"decile GROSS"),(0.1,0.06,"decile NET 6% borrow"),(0.2,0.0,"quintile GROSS"),(0.2,0.06,"quintile NET")]:
    ls,lo=mlls(q,bc); a,s,d=ann(ls); p(f"{'ML L/S '+lab:40} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {ls.corr(qret):>8.2f}")
mlls_net,mlls_long=mlls(0.1,0.06); mlls_gross,_=mlls(0.1,0.0)
a,s,d=ann(mlls_long); p(f"{'ML long-only top-decile':40} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {mlls_long.corr(qret):>8.2f}")
# blend ML L/S with linear factor L/S (alpha-model diversification)
Dl=pd.read_pickle("/tmp/wave/_ls.pkl"); LS=Dl["LS"]
good=[k for k in LS if LS[k].dropna().std()>0 and LS[k].mean()/LS[k].std()>0.1]
lin=pd.DataFrame({k:LS[k] for k in good}).reindex(idx).mean(axis=1)
p(f"\ncorr(ML L/S, linear factor L/S) = {mlls_gross.corr(lin):.2f}")
a,s,d=ann(lin); p(f"{'linear factor L/S (gross)':40} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {lin.corr(qret):>8.2f}")
combo=0.5*mlls_gross+0.5*lin; a,s,d=ann(combo); p(f"{'50 ML / 50 linear L/S (gross)':40} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {combo.corr(qret):>8.2f}")
# portable: QQQ + ML L/S net
for k in [1.0,1.5]:
    r=qret+k*mlls_net; a,s,d=ann(r); p(f"{'QQQ + '+str(k)+'x ML L/S (net)':40} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {r.corr(qret):>8.2f}")
# sub-periods ML L/S net
p(f"\nML L/S net sub-period:")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); a,s,_=ann(mlls_net[m]); p(f"  {lo}-{hi}: ann {a*100:>5.1f}% Sharpe {s:.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
