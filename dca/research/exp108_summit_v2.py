import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; fnames=list(FEAT.keys())
ret=(me/me.shift(1)-1).clip(-0.9,2.0)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
C=pd.read_pickle("/tmp/wave/_composite.pkl"); compN=C["compN"]; tkr2sec=C["tkr2sec"]
sec=pd.Series({c:tkr2sec.get(c,-1) for c in cols}); secv=sec.values
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
LIQ=(liq&(me>=3.0)).fillna(False)
# short-universe variants (borrow-aware: bigger = easier/cheaper to borrow)
SHORT_U={"$5":(liq&(me>=5.0)).fillna(False),"$10":(liq&(me>=10.0)).fillna(False),
         "top50%mcap":(liq&(me>=3.0)&(FEAT["log_mcap"].rank(axis=1,pct=True)>=0.5)).fillna(False)}

def ann(r,sub=None):
    r=r.dropna()
    if sub is not None: r=r[(r.index>=pd.Timestamp(sub[0]))&(r.index<=pd.Timestamp(sub[1]))]
    if len(r)<6: return (np.nan,)*3
    a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d

def sec_neutralize_rank(sig):
    # within-sector demean of the cross-sectional rank each month
    fr=sig.where(LIQ); rk=fr.rank(axis=1,pct=True); arr=rk.values.copy(); out=arr.copy()
    sv=np.array([tkr2sec.get(c,-1) for c in rk.columns])
    for gi in np.unique(sv):
        if gi<0: continue
        ix=np.where(sv==gi)[0]
        if len(ix)<5: continue
        sub=arr[:,ix]; mu=np.nanmean(sub,axis=1,keepdims=True); out[:,ix]=sub-mu
    return pd.DataFrame(out,index=rk.index,columns=rk.columns)

def weights(score, short_mask, q=0.1, rebal=3, buffer=2.0):
    rkl=score.where(LIQ).rank(axis=1,pct=True)
    rks=score.where(short_mask).rank(axis=1,pct=True)
    Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols)
    held_l=set(); held_s=set(); curWl=pd.Series(0.0,index=cols); curWs=pd.Series(0.0,index=cols)
    for k,dt in enumerate(M):
        if k%rebal==0:
            rl=rkl.loc[dt]; rs=rks.loc[dt]
            keep_l={t for t in held_l if rl.get(t,0)>=1-q*buffer}
            sel_l=list(keep_l); ntar=int((rl>=1-q).sum())
            for t in rl[rl>=1-q].sort_values(ascending=False).index:
                if len(sel_l)>=ntar: break
                if t not in sel_l: sel_l.append(t)
            keep_s={t for t in held_s if rs.get(t,1)<=q*buffer}
            sel_s=list(keep_s); star=int((rs<=q).sum())
            for t in rs[rs<=q].sort_values().index:
                if len(sel_s)>=star: break
                if t not in sel_s: sel_s.append(t)
            held_l=set(sel_l); held_s=set(sel_s)
            curWl=pd.Series(0.0,index=cols); curWs=pd.Series(0.0,index=cols)
            if sel_l: curWl[sel_l]=1.0/len(sel_l)
            if sel_s: curWs[sel_s]=1.0/len(sel_s)
        Wl.loc[dt]=curWl; Ws.loc[dt]=curWs
    return Wl,Ws

def perf(Wl,Ws,cost_bps=10,borrow=0.06):
    g=((Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1)).reindex(idx)
    turn=(Wl.diff().abs().sum(axis=1)+Ws.diff().abs().sum(axis=1)).reindex(idx)
    net=g-turn*(cost_bps/1e4)-(borrow/12)
    return net

def voltarget(r,tgt=0.10,win=6,cap=2.0):
    rv=r.rolling(win,min_periods=3).std()*np.sqrt(12)
    lev=(tgt/rv).clip(upper=cap).shift(1).fillna(1.0)
    return r*lev

DEV=("2015-01-01","2022-12-31"); HOLD=("2023-01-01","2025-12-31"); FULL=("2015-01-01","2025-12-31")
def row(nm,r):
    aF,sF,dF=ann(r,FULL); aD,sD,dD=ann(r,DEV); aH,sH,dH=ann(r,HOLD)
    p(f"{nm:38}{aF*100:>6.1f}{sF:>6.2f}{dF:>7.1%} |{sD:>6.2f}{sH:>6.2f} |{r.reindex(idx).corr(qret):>7.2f}")

p(f"{'SUMMIT variant (net 10bps+6% borrow)':38}{'ann%':>6}{'Sh':>6}{'maxDD':>7} |{'devSh':>6}{'hldSh':>6} |{'cQQQ':>7}")
p("-"*88)
# --- BASE: deployable ML L/S (exp78 config) ---
mlrank=PROB.where(LIQ).rank(axis=1,pct=True)
Wl,Ws=weights(PROB,SHORT_U["$5"]); base=perf(Wl,Ws); row("BASE ML L/S q+buf2x ($5 short)",base)

# --- LEVER 1: sector-neutralized ML score ---
mlN=sec_neutralize_rank(PROB)
Wl,Ws=weights(mlN,SHORT_U["$5"]); s1=perf(Wl,Ws); row("+ sector-neutral ML",s1)

# --- LEVER 2: borrow-aware short universe ---
for un in ["$10","top50%mcap"]:
    Wl,Ws=weights(mlN,SHORT_U[un]); r=perf(Wl,Ws); row(f"+ secN + short {un}",r)

# --- LEVER 3: ensemble ML(sectorN) + linear composite(sectorN) ---
compNr=compN.where(LIQ).rank(axis=1,pct=True)
ens=(mlN.rank(axis=1,pct=True)+compNr)/2   # avg of two sector-neutral rank signals
Wl,Ws=weights(ens,SHORT_U["$10"]); s3=perf(Wl,Ws); row("+ ensemble ML+linear, short $10",s3)

# --- LEVER 4: vol-target the best book to 10% ---
best=s3
row("+ vol-target 10% (cap 2x)",voltarget(best))

# --- combined champion: ensemble + short top50% + vol-target ---
Wl,Ws=weights(ens,SHORT_U["top50%mcap"]); champ=perf(Wl,Ws)
row("CHAMP ens+top50short",champ); row("CHAMP + vol-target 10%",voltarget(champ))

p("\n# cost ladder for champion (vol-targeted ensemble, top50% short):")
champvt=voltarget(champ)
for cb in [0,5,10,20]:
    Wl,Ws=weights(ens,SHORT_U["top50%mcap"]); r=voltarget(perf(Wl,Ws,cb))
    a,s,d=ann(r,FULL); p(f"  {cb:>2}bps/side: ann {a*100:>5.1f}% Sharpe {s:.2f} maxDD {d:.1%}")
pd.to_pickle({"base":base,"champ":champ,"champvt":champvt,"ens":ens,"mlN":mlN},"/tmp/wave/_summit_v2.pkl")
p(f"\nDONE t={time.time()-t0:.0f}s")
