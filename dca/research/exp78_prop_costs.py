import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; fnames=list(FEAT.keys())
ret=(me/me.shift(1)-1).clip(-0.9,2.0); fwd1=ret.shift(-1)
# ADV for capacity (need volume) — approximate via vol chunks-derived? use price as proxy unavailable; load adv from featmat? not present.
# Build ML prob (walk-forward), save for reuse
import os
if os.path.exists("/tmp/wave/_mlprob.pkl"):
    PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl"); p("loaded cached ML prob")
else:
    Z={nm:FEAT[nm].where(liq).rank(axis=1,pct=True) for nm in fnames}
    recs=[]
    for dt in M[(M>=pd.Timestamp("2011-06-01"))]:
        fv=fok.loc[dt].dropna()
        if len(fv)<60: continue
        q1,q2=fv.quantile(1/3),fv.quantile(2/3)
        y=pd.Series(np.where(fv>=q2,1,np.where(fv<=q1,0,np.nan)),index=fv.index).dropna()
        X=np.column_stack([Z[nm].loc[dt].reindex(y.index).values for nm in fnames])
        for i,tk in enumerate(y.index): recs.append((dt,tk,*X[i],int(y.iloc[i])))
    DF=pd.DataFrame.from_records(recs,columns=["date","tk"]+fnames+["y"])
    from sklearn.ensemble import HistGradientBoostingClassifier
    pr=[]
    for ytest in range(2015,2026):
        tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")]; te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
        if len(te)==0 or len(tr)<5000: continue
        clf=HistGradientBoostingClassifier(max_iter=250,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200)
        clf.fit(tr[fnames].values,tr["y"].values)
        t2=te[["date","tk"]].copy(); t2["p"]=clf.predict_proba(te[fnames].values)[:,1]; pr.append(t2)
    PROB=pd.concat(pr).pivot_table(index="date",columns="tk",values="p").reindex(M)
    pd.to_pickle(PROB,"/tmp/wave/_mlprob.pkl")
    p(f"ML trained+saved t={time.time()-t0:.0f}s")
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
LIQ=(liq&(me>=3.0)); LIQ5=(liq&(me>=5.0))
def ann(r):
    r=r.dropna(); a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
def weights(q=0.1, rebal=1, buffer=1.0):
    prl=PROB.where(LIQ); rkl=prl.rank(axis=1,pct=True)
    prs=PROB.where(LIQ5); rks=prs.rank(axis=1,pct=True)
    Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols)
    held_l=set(); held_s=set(); curWl=pd.Series(0.0,index=cols); curWs=pd.Series(0.0,index=cols)
    for k,dt in enumerate(M):
        if k%rebal==0 and dt in PROB.index:
            # buffered selection: keep if rank within buffer*q, add new top q
            rl=rkl.loc[dt]; rs=rks.loc[dt]
            keep_l={t for t in held_l if rl.get(t,0)>=1-q*buffer}
            new_l=[t for t in rl[rl>=1-q].sort_values(ascending=False).index]
            sel_l=list(keep_l);
            for t in new_l:
                if len(sel_l)>= (rl>=1-q).sum(): break
                if t not in sel_l: sel_l.append(t)
            keep_s={t for t in held_s if rs.get(t,1)<=q*buffer}
            new_s=[t for t in rs[rs<=q].sort_values().index]
            sel_s=list(keep_s)
            for t in new_s:
                if len(sel_s)>=(rs<=q).sum(): break
                if t not in sel_s: sel_s.append(t)
            held_l=set(sel_l); held_s=set(sel_s)
            curWl=pd.Series(0.0,index=cols); curWs=pd.Series(0.0,index=cols)
            if sel_l: curWl[sel_l]=1.0/len(sel_l)
            if sel_s: curWs[sel_s]=1.0/len(sel_s)
        Wl.loc[dt]=curWl; Ws.loc[dt]=curWs
    return Wl,Ws
def perf(Wl,Ws,cost_bps=10,borrow=0.06):
    g=(Wl.shift(1)*fwd1.shift(1)).sum(axis=1)-(Ws.shift(1)*fwd1.shift(1)).sum(axis=1)  # hold prior wts
    g=( (Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1) ).reindex(idx)
    turn=( Wl.diff().abs().sum(axis=1)+Ws.diff().abs().sum(axis=1) ).reindex(idx)
    cost=turn*(cost_bps/1e4)+ (borrow/12)   # trade cost on traded notional + borrow on short
    net=g-cost
    return g,net,turn
p(f"{'config':40} {'turn/mo':>8} {'grossSh':>8} {'netSh@10':>9} {'net@20':>8} {'netAnn%':>8}")
for nm,(rebal,buf) in {"monthly decile":(1,1.0),"monthly +buffer2x":(1,2.0),"quarterly decile":(3,1.0),"quarterly +buffer2x":(3,2.0),"monthly quintile":(1,1.0)}.items():
    q=0.2 if "quintile" in nm else 0.1
    Wl,Ws=weights(q=q,rebal=rebal,buffer=buf)
    g,n10,turn=perf(Wl,Ws,10); _,n20,_=perf(Wl,Ws,20)
    ag,sg,_=ann(g); _,s10,_=ann(n10); an20,s20,_=ann(n20)
    p(f"{nm:40} {turn.mean():>8.0%} {sg:>8.2f} {s10:>9.2f} {s20:>8.2f} {an20*100:>8.1f}")
# best deployable prop = quarterly+buffer; full cost ladder
Wl,Ws=weights(q=0.1,rebal=3,buffer=2.0)
p(f"\nQuarterly+buffer2x decile — net Sharpe by cost level (incl 6%/yr borrow):")
for cb in [0,5,10,20,30]:
    _,n,_=perf(Wl,Ws,cb); a,s,d=ann(n); p(f"  {cb:>2}bps/side: ann {a*100:>5.1f}% Sharpe {s:.2f} maxDD {d:.1%}")
# capacity: position size vs ADV (approx ADV = need volume; use $3+ price * assume). Skip precise; report name counts
nL=(Wl.iloc[-1]>0).sum(); nS=(Ws.iloc[-1]>0).sum()
p(f"\nbook: ~{nL} long / {nS} short names; equal-weight. Capacity ~ (median ADV of names) x (max %ADV per name).")
p(f"DONE t={time.time()-t0:.0f}s")
