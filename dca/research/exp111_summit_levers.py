import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; ret=(me/me.shift(1)-1).clip(-0.9,2.0)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(M)
LIQ=(liq&(me>=3.0)).fillna(False); SHORT=(liq&(me>=10.0)).fillna(False)
vol6=FEAT["vol6"]  # trailing vol for inverse-vol weighting
DEV=("2015-01-01","2022-12-31"); HOLD=("2023-01-01","2025-12-31"); FULL=("2015-01-01","2025-12-31")
# trailing 12m beta panel vs QQQ (vectorized)
q=qret.reindex(M); qm=q.rolling(12,min_periods=8).mean(); qv=q.rolling(12,min_periods=8).var()
rq=ret.mul(q,axis=0); cov=rq.rolling(12,min_periods=8).mean().sub(ret.rolling(12,min_periods=8).mean().mul(qm,axis=0),axis=0)
BETA=cov.div(qv,axis=0).clip(-3,3).fillna(1.0)
def ann(r,sub=None):
    r=r.dropna()
    if sub: r=r[(r.index>=pd.Timestamp(sub[0]))&(r.index<=pd.Timestamp(sub[1]))]
    if len(r)<6: return (np.nan,)*3
    a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
def build(PROB,q=0.1,rebal=3,buffer=2.0,wscheme="eq",betaneutral=False):
    PROB=PROB.reindex(index=M,columns=cols)
    rkl=PROB.where(LIQ).rank(axis=1,pct=True); rks=PROB.where(SHORT).rank(axis=1,pct=True)
    Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols)
    hl=set(); hs=set(); cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
    for k,dt in enumerate(M):
        if k%rebal==0:
            rl=rkl.loc[dt]; rs=rks.loc[dt]
            sel_l=[t for t in hl if rl.get(t,0)>=1-q*buffer]; nt=int((rl>=1-q).sum())
            for t in rl[rl>=1-q].sort_values(ascending=False).index:
                if len(sel_l)>=nt: break
                if t not in sel_l: sel_l.append(t)
            sel_s=[t for t in hs if rs.get(t,1)<=q*buffer]; st=int((rs<=q).sum())
            for t in rs[rs<=q].sort_values().index:
                if len(sel_s)>=st: break
                if t not in sel_s: sel_s.append(t)
            hl=set(sel_l); hs=set(sel_s)
            def wt(sel,leg):
                w=pd.Series(0.0,index=cols)
                if not sel: return w
                if wscheme=="eq": w[sel]=1.0
                elif wscheme=="conv":
                    rr=(rl if leg=="L" else (1-rs)); w[sel]=rr.reindex(sel).clip(lower=0.01).values
                elif wscheme=="ivol":
                    iv=(1.0/vol6.loc[dt].reindex(sel).clip(lower=0.02)); w[sel]=iv.values
                return w/w.sum()
            cl=wt(sel_l,"L"); cs=wt(sel_s,"S")
            if betaneutral:
                bl=(cl*BETA.loc[dt]).sum(); bs=(cs*BETA.loc[dt]).sum()
                if bs>0.05: cs=cs*(bl/bs)   # scale short notional to match long beta
        Wl.loc[dt]=cl; Ws.loc[dt]=cs
    return Wl,Ws
def perf(Wl,Ws,cb=10,borrow=0.06):
    g=((Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1)).reindex(idx)
    turn=(Wl.diff().abs().sum(axis=1)+Ws.diff().abs().sum(axis=1)).reindex(idx)
    grossshort=Ws.shift(1).sum(axis=1).reindex(idx)  # short notional (may differ from 1 if beta-neutral)
    return g-turn*(cb/1e4)-(borrow/12)*grossshort
def legperf(Wl,Ws,cb=10,borrow=0.06):
    lo=(Wl.shift(1)*ret).sum(axis=1).reindex(idx)-(Wl.diff().abs().sum(axis=1)*(cb/1e4)).reindex(idx)
    sh=-(Ws.shift(1)*ret).sum(axis=1).reindex(idx)-(borrow/12)-(Ws.diff().abs().sum(axis=1)*(cb/1e4)).reindex(idx)
    return lo,sh
def row(nm,r):
    aF,sF,dF=ann(r,FULL); _,sD,_=ann(r,DEV); _,sH,_=ann(r,HOLD)
    p(f"{nm:34}{aF*100:>6.1f}{sF:>6.2f}{dF:>7.1%} |{sD:>6.2f}{sH:>6.2f} |{r.reindex(idx).corr(qret.reindex(idx)):>7.2f}")
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl")
p(f"{'lever':34}{'ann%':>6}{'Sh':>6}{'maxDD':>7} |{'devSh':>6}{'hldSh':>6} |{'cQQQ':>7}")
p("-"*84)
Wl,Ws=build(PROB); base=perf(Wl,Ws); row("v2 BASE (eq-wt, short $10)",base)
# 1) LEG DECOMPOSITION
lo,sh=legperf(Wl,Ws)
p(f"   long-leg alone:  ann {ann(lo,FULL)[0]*100:5.1f}%  Sharpe {ann(lo,FULL)[1]:.2f}  corrQQQ {lo.corr(qret.reindex(idx)):.2f}")
p(f"   short-leg alone: ann {ann(sh,FULL)[0]*100:5.1f}%  Sharpe {ann(sh,FULL)[1]:.2f}  corrQQQ {sh.corr(qret.reindex(idx)):.2f}  (net of 6% borrow)")
p("-"*84)
# 2) ALTERNATIVE ML MODELS
for f in ["_mlprob_50feat","_mlprob_banger","_mlprob_ens","_mlprob_selens"]:
    Pm=pd.read_pickle(f"/tmp/wave/{f}.pkl"); Wl,Ws=build(Pm); row(f"model {f[7:]}",perf(Wl,Ws))
p("-"*84)
# 3) WEIGHTING SCHEMES
for ws in ["conv","ivol"]:
    Wl,Ws=build(PROB,wscheme=ws); row(f"weight={ws}",perf(Wl,Ws))
p("-"*84)
# 4) BETA-NEUTRAL
Wl,Ws=build(PROB,betaneutral=True); row("beta-neutral",perf(Wl,Ws))
Wl,Ws=build(PROB,wscheme="ivol",betaneutral=True); row("ivol + beta-neutral",perf(Wl,Ws))
p(f"\nDONE t={time.time()-t0:.0f}s")
