import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; ret=(me/me.shift(1)-1).clip(-0.9,2.0)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(M)
LIQ=(liq&(me>=3.0)).fillna(False); SHORT=(liq&(me>=10.0)).fillna(False)
lmrank=FEAT["log_mcap"].rank(axis=1,pct=True)
DEV=("2015-01-01","2022-12-31"); HOLD=("2023-01-01","2025-12-31"); FULL=("2015-01-01","2025-12-31")
q=qret.reindex(M); qm=q.rolling(12,min_periods=8).mean(); qv=q.rolling(12,min_periods=8).var()
rq=ret.mul(q,axis=0); cov=rq.rolling(12,min_periods=8).mean().sub(ret.rolling(12,min_periods=8).mean().mul(qm,axis=0),axis=0)
BETA=cov.div(qv,axis=0).clip(-3,3).fillna(1.0)
SZ=lmrank  # size factor for size-neutralization
def ann(r,sub=None):
    r=r.dropna()
    if sub: r=r[(r.index>=pd.Timestamp(sub[0]))&(r.index<=pd.Timestamp(sub[1]))]
    if len(r)<6: return (np.nan,)*3
    a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(index=M,columns=cols)
borrow_rate=pd.DataFrame(0.06,index=M,columns=cols).where(lmrank<0.8,0.02).where(lmrank<0.95,0.01)
def build(short_stop=None,size_neutral=False):
    rebal,buffer,qq=3,2.0,0.1
    rkl=PROB.where(LIQ).rank(axis=1,pct=True); rks=PROB.where(SHORT).rank(axis=1,pct=True)
    Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols)
    hl=set(); hs={}; cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
    for k,dt in enumerate(M):
        px=me.loc[dt]
        # short-stop: cover names that rallied >= stop since entry
        if short_stop is not None and hs:
            for tk in list(hs.keys()):
                e=hs[tk]; cpx=px.get(tk,np.nan)
                if np.isfinite(cpx) and cpx/e["px"]-1>=short_stop: hs.pop(tk)
        if k%rebal==0:
            rl=rkl.loc[dt]; rs=rks.loc[dt]
            sel_l=[t for t in hl if rl.get(t,0)>=1-qq*buffer]; nt=int((rl>=1-qq).sum())
            for t in rl[rl>=1-qq].sort_values(ascending=False).index:
                if len(sel_l)>=nt: break
                if t not in sel_l: sel_l.append(t)
            sel_s=[t for t in hs if rs.get(t,1)<=qq*buffer]; st=int((rs<=qq).sum())
            for t in rs[rs<=qq].sort_values().index:
                if len(sel_s)>=st: break
                if t not in sel_s: sel_s.append(t)
            hl=set(sel_l); newhs={}
            for t in sel_s: newhs[t]=hs.get(t) or {"px":px.get(t,np.nan)}
            hs=newhs
            cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
            if sel_l: cl[sel_l]=1.0/len(sel_l)
            if sel_s: cs[sel_s]=1.0/len(sel_s)
        else:
            cs=pd.Series(0.0,index=cols)
            if hs: cs[list(hs.keys())]=1.0/len(hs)  # re-equal-weight survivors after stops
        b=BETA.loc[dt]
        if size_neutral:  # demean weights' size exposure by tilting (approx via beta only handled separately)
            pass
        bl=(cl*b).sum(); bs=(cs*b).sum()
        if bs>0.05: cs=cs*(bl/bs)
        Wl.loc[dt]=cl; Ws.loc[dt]=cs
    return Wl,Ws
def perf(Wl,Ws,cb=10,tiered=True):
    g=((Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1)).reindex(idx)
    turn=(Wl.diff().abs().sum(axis=1)+Ws.diff().abs().sum(axis=1)).reindex(idx)
    bcost=(Ws.shift(1)*borrow_rate/12).sum(axis=1).reindex(idx) if tiered else (0.06/12)*Ws.shift(1).sum(axis=1).reindex(idx)
    return g-turn*(cb/1e4)-bcost
def row(nm,r):
    aF,sF,dF=ann(r,FULL); _,sD,_=ann(r,DEV); _,sH,_=ann(r,HOLD); _,s19,d19=ann(r,("2019-01-01","2021-12-31"))
    p(f"{nm:32}{aF*100:>6.1f}{sF:>6.2f}{dF:>7.1%} |{sD:>6.2f}{sH:>6.2f} |{r.reindex(idx).corr(qret.reindex(idx)):>6.2f} |{s19:>6.2f}{d19:>7.1%}")
p(f"{'config (beta-neutral, tiered)':32}{'ann%':>6}{'Sh':>6}{'maxDD':>7} |{'devSh':>6}{'hldSh':>6} |{'cQQQ':>6} |{'19-21Sh':>6}{'DD':>7}")
p("-"*100)
Wl,Ws=build(); row("v3 base",perf(Wl,Ws))
for ss in [0.30,0.40,0.50,0.75]:
    Wl,Ws=build(short_stop=ss); row(f"+ short-stop +{int(ss*100)}%",perf(Wl,Ws))
p(f"\nDONE t={time.time()-t0:.0f}s")
