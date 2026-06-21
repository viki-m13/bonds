import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index
ret=(me/me.shift(1)-1).clip(-0.9,2.0)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
LIQ=(liq&(me>=3.0)).fillna(False)
SHORT_U={"$5":(liq&(me>=5.0)).fillna(False),"$10":(liq&(me>=10.0)).fillna(False),
         "top40%mcap":(liq&(me>=5.0)&(FEAT["log_mcap"].rank(axis=1,pct=True)>=0.6)).fillna(False)}
DEV=("2015-01-01","2022-12-31"); HOLD=("2023-01-01","2025-12-31"); FULL=("2015-01-01","2025-12-31")
def ann(r,sub=None):
    r=r.dropna()
    if sub is not None: r=r[(r.index>=pd.Timestamp(sub[0]))&(r.index<=pd.Timestamp(sub[1]))]
    if len(r)<6: return (np.nan,)*3
    a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
def weights(score, short_mask, q=0.1, rebal=3, buffer=2.0):
    rkl=score.where(LIQ).rank(axis=1,pct=True); rks=score.where(short_mask).rank(axis=1,pct=True)
    Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols)
    held_l=set(); held_s=set(); cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
    for k,dt in enumerate(M):
        if k%rebal==0:
            rl=rkl.loc[dt]; rs=rks.loc[dt]
            sel_l=[t for t in held_l if rl.get(t,0)>=1-q*buffer]; nt=int((rl>=1-q).sum())
            for t in rl[rl>=1-q].sort_values(ascending=False).index:
                if len(sel_l)>=nt: break
                if t not in sel_l: sel_l.append(t)
            sel_s=[t for t in held_s if rs.get(t,1)<=q*buffer]; st=int((rs<=q).sum())
            for t in rs[rs<=q].sort_values().index:
                if len(sel_s)>=st: break
                if t not in sel_s: sel_s.append(t)
            held_l=set(sel_l); held_s=set(sel_s)
            cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
            if sel_l: cl[sel_l]=1.0/len(sel_l)
            if sel_s: cs[sel_s]=1.0/len(sel_s)
        Wl.loc[dt]=cl; Ws.loc[dt]=cs
    return Wl,Ws
def perf(Wl,Ws,cost_bps=10,borrow=0.06):
    g=((Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1)).reindex(idx)
    turn=(Wl.diff().abs().sum(axis=1)+Ws.diff().abs().sum(axis=1)).reindex(idx)
    return g-turn*(cost_bps/1e4)-(borrow/12)
def voltarget(r,tgt,win=6,cap=2.0):
    rv=r.rolling(win,min_periods=3).std()*np.sqrt(12)
    lev=(tgt/rv).clip(upper=cap).shift(1).fillna(1.0); return r*lev
def row(nm,r):
    aF,sF,dF=ann(r,FULL); _,sD,_=ann(r,DEV); _,sH,_=ann(r,HOLD)
    p(f"{nm:40}{aF*100:>6.1f}{sF:>6.2f}{dF:>7.1%} |{sD:>6.2f}{sH:>6.2f} |{r.reindex(idx).corr(qret):>7.2f}")
p(f"{'variant (net 10bps+6% borrow)':40}{'ann%':>6}{'Sh':>6}{'maxDD':>7} |{'devSh':>6}{'hldSh':>6} |{'cQQQ':>7}")
p("-"*90)
Wl5,Ws5=weights(PROB,SHORT_U["$5"]); base=perf(Wl5,Ws5); row("BASE ML L/S ($5 short)",base)
# vol-target sweep on BASE
for tg in [0.08,0.10,0.12,0.15]:
    row(f"BASE + vol-target {int(tg*100)}%",voltarget(base,tg))
# borrow-aware short alone (no VT)
for un in ["$10","top40%mcap"]:
    Wl,Ws=weights(PROB,SHORT_U[un]); row(f"BASE short {un}",perf(Wl,Ws))
# borrow-aware + vol-target combo
Wl,Ws=weights(PROB,SHORT_U["top40%mcap"]); b40=perf(Wl,Ws)
row("short top40% + VT 10%",voltarget(b40,0.10))
row("short top40% + VT 12%",voltarget(b40,0.12))
Wl,Ws=weights(PROB,SHORT_U["$10"]); b10=perf(Wl,Ws)
row("short $10 + VT 12%",voltarget(b10,0.12))
# scaled to match BASE vol (~ for fair return comparison): vol-target to base realized vol
bv=base.std()*np.sqrt(12)
row(f"short top40% + VT to {bv*100:.0f}% (base-vol)",voltarget(b40,bv))
p(f"\nBASE realized vol {bv*100:.1f}%/yr")
# save champion = short top40% + VT12%
champ=voltarget(b40,0.12)
pd.to_pickle({"base":base,"champ":champ,"b40":b40},"/tmp/wave/_summit_vt.pkl")
p(f"DONE t={time.time()-t0:.0f}s")
