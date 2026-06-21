import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; ret=(me/me.shift(1)-1).clip(-0.9,2.0)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(M)
LIQ=(liq&(me>=3.0)).fillna(False); SHORT=(liq&(me>=10.0)).fillna(False)
vol6=FEAT["vol6"]; lmrank=FEAT["log_mcap"].rank(axis=1,pct=True)
DEV=("2015-01-01","2022-12-31"); HOLD=("2023-01-01","2025-12-31"); FULL=("2015-01-01","2025-12-31")
q=qret.reindex(M); qm=q.rolling(12,min_periods=8).mean(); qv=q.rolling(12,min_periods=8).var()
rq=ret.mul(q,axis=0); cov=rq.rolling(12,min_periods=8).mean().sub(ret.rolling(12,min_periods=8).mean().mul(qm,axis=0),axis=0)
BETA=cov.div(qv,axis=0).clip(-3,3).fillna(1.0)
def ann(r,sub=None):
    r=r.dropna()
    if sub: r=r[(r.index>=pd.Timestamp(sub[0]))&(r.index<=pd.Timestamp(sub[1]))]
    if len(r)<6: return (np.nan,)*3
    a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(index=M,columns=cols)
# tiered borrow: larger-cap shorts cheaper to borrow
borrow_rate=pd.DataFrame(0.06,index=M,columns=cols)
borrow_rate=borrow_rate.where(lmrank<0.8,0.02).where(lmrank<0.95,0.01)  # top 20%->2%, top5%->1%
def build(q=0.1,rebal=3,buffer=2.0,wscheme="eq",betaneutral=True):
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
                elif wscheme=="conv": w[sel]=(rl if leg=="L" else (1-rs)).reindex(sel).clip(lower=.01).values
                elif wscheme=="ivol": w[sel]=(1.0/vol6.loc[dt].reindex(sel).clip(lower=.02)).values
                return w/w.sum()
            cl=wt(sel_l,"L"); cs=wt(sel_s,"S")
            if betaneutral:
                bl=(cl*BETA.loc[dt]).sum(); bs=(cs*BETA.loc[dt]).sum()
                if bs>0.05: cs=cs*(bl/bs)
        Wl.loc[dt]=cl; Ws.loc[dt]=cs
    return Wl,Ws
def perf(Wl,Ws,cb=10,tiered=False):
    g=((Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1)).reindex(idx)
    turn=(Wl.diff().abs().sum(axis=1)+Ws.diff().abs().sum(axis=1)).reindex(idx)
    if tiered: bcost=(Ws.shift(1)*borrow_rate/12).sum(axis=1).reindex(idx)
    else: bcost=(0.06/12)*Ws.shift(1).sum(axis=1).reindex(idx)
    return g-turn*(cb/1e4)-bcost
def lever(r,tgt,cap=3.0):
    rv=r.rolling(6,min_periods=3).std()*np.sqrt(12); lv=(tgt/rv).clip(upper=cap).shift(1).fillna(1.0); return r*lv
def row(nm,r,extra=""):
    aF,sF,dF=ann(r,FULL); _,sD,_=ann(r,DEV); _,sH,_=ann(r,HOLD)
    p(f"{nm:34}{aF*100:>6.1f}{sF:>6.2f}{dF:>7.1%} |{sD:>6.2f}{sH:>6.2f} |{r.reindex(idx).corr(qret.reindex(idx)):>7.2f} {extra}")
p(f"{'config':34}{'ann%':>6}{'Sh':>6}{'maxDD':>7} |{'devSh':>6}{'hldSh':>6} |{'cQQQ':>7}")
p("-"*86)
Wl,Ws=build(wscheme="eq"); bn=perf(Wl,Ws); row("beta-neutral eq (v3)",bn,f"avgShortNotional={Ws.reindex(idx).sum(axis=1).mean():.2f}")
Wl,Ws=build(wscheme="conv"); row("beta-neutral conv",perf(Wl,Ws))
Wl,Ws=build(wscheme="ivol"); row("beta-neutral ivol",perf(Wl,Ws))
# tiered borrow on the eq beta-neutral
Wl,Ws=build(wscheme="eq"); row("beta-neutral eq + tiered borrow",perf(Wl,Ws,tiered=True))
p("-"*86)
# return-matched: lever beta-neutral book to higher vol targets (DD scales, Sharpe ~const)
Wl,Ws=build(wscheme="eq"); bn=perf(Wl,Ws)
for tg in [0.12,0.18,0.24]:
    row(f"beta-neutral, lever to {int(tg*100)}% vol",lever(bn,tg))
p(f"\nbeta-neutral native vol: {bn.std()*np.sqrt(12)*100:.1f}%/yr")
# sub-period robustness of v3
p("\nv3 (beta-neutral eq) sub-period Sharpe:")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    a,s,d=ann(bn,(lo+"-01-01",hi+"-12-31")); p(f"  {lo}-{hi}: ann {a*100:5.1f}%  Sharpe {s:.2f}  maxDD {d:.1%}")
# cost ladder
p("\ncost ladder (beta-neutral eq, incl 6% borrow):")
for cb in [0,5,10,20]:
    Wl,Ws=build(wscheme="eq"); a,s,d=ann(perf(Wl,Ws,cb),FULL); p(f"  {cb:>2}bps/side: ann {a*100:5.1f}% Sharpe {s:.2f} maxDD {d:.1%}")
pd.to_pickle({"bn":bn},"/tmp/wave/_summit_v3.pkl")
p(f"\nDONE t={time.time()-t0:.0f}s")
