import time; t0=time.time()
from costlib import *
# alternative beta estimates
def beta_win(win):
    qm=q.rolling(win,min_periods=max(6,win//2)).mean(); qv=q.rolling(win,min_periods=max(6,win//2)).var()
    cv=rq.rolling(win,min_periods=max(6,win//2)).mean().sub(ret.rolling(win,min_periods=max(6,win//2)).mean().mul(qm,axis=0),axis=0)
    return cv.div(qv,axis=0).clip(-3,3).fillna(1.0)
def build_beta(BETAuse,shrink=0.0,hedge=False):
    rebal,buffer,qq=3,2.0,0.1
    B=(1-shrink)*BETAuse+shrink*1.0
    rkl=PROB.where(LIQ).rank(axis=1,pct=True); rks=PROB.where(SHORT).rank(axis=1,pct=True)
    Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols); HB=pd.Series(0.0,index=M)
    hl=set(); hs=set(); cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
    for k,dt in enumerate(M):
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
            hl=set(sel_l); hs=set(sel_s)
            cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
            if sel_l: cl[sel_l]=1.0/len(sel_l)
            if sel_s: cs[sel_s]=1.0/len(sel_s)
        b=B.loc[dt]; bl=(cl*b).sum(); bs=(cs*b).sum()
        if hedge: css=cs; HB.loc[dt]=-(bl-bs)          # explicit index hedge of net beta
        else: css=cs*(bl/bs) if bs>0.05 else cs; HB.loc[dt]=0.0
        Wl.loc[dt]=cl; Ws.loc[dt]=css
    return Wl,Ws,HB
def netH(Wl,Ws,HB):
    base=net(Wl,Ws)
    return base+(HB.shift(1)*qret.reindex(M)).reindex(idx).fillna(0.0)
hdr()
Wl,Ws,HB=build_beta(BETA); row("v3 (12m beta, scale-short)",netH(Wl,Ws,HB))
print("-"*92)
for w in [6,18,24]:
    Wl,Ws,HB=build_beta(beta_win(w)); row(f"beta window {w}m",netH(Wl,Ws,HB))
print("-"*92)
for sh in [0.25,0.5]:
    Wl,Ws,HB=build_beta(BETA,shrink=sh); row(f"beta shrink {sh} ->1.0",netH(Wl,Ws,HB))
print("-"*92)
Wl,Ws,HB=build_beta(BETA,hedge=True); row("explicit QQQ hedge (vs scale)",netH(Wl,Ws,HB))
print(f"\nDONE t={time.time()-t0:.0f}s")
