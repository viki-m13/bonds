import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; ret=(me/me.shift(1)-1).clip(-0.9,2.0)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(M)
LIQ=(liq&(me>=3.0)).fillna(False); SHORT=(liq&(me>=10.0)).fillna(False)
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def _qidx(df):
    df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=_qidx(rev); SH=_qidx(F["EntityCommonStockSharesOutstanding"]).reindex(columns=rev.columns)
def qmap(df):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=6)
mcap=(me*qmap(SH)); mcap=mcap.where(mcap>0); lmrank=np.log(mcap).rank(axis=1,pct=True)
ADV=mcap*0.005; sig_d=(FEAT["vol6"]/np.sqrt(252)).clip(0.005,0.20)
spread=pd.DataFrame(40.0,index=M,columns=cols).where(lmrank<0.5,20.0).where(lmrank<0.8,8.0).where(lmrank<0.95,4.0)
borrow_rate=pd.DataFrame(0.06,index=M,columns=cols).where(lmrank<0.8,0.02).where(lmrank<0.95,0.01)
q=qret.reindex(M); qm=q.rolling(12,min_periods=8).mean(); qv=q.rolling(12,min_periods=8).var()
rq=ret.mul(q,axis=0); cov=rq.rolling(12,min_periods=8).mean().sub(ret.rolling(12,min_periods=8).mean().mul(qm,axis=0),axis=0)
BETA=cov.div(qv,axis=0).clip(-3,3).fillna(1.0)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(index=M,columns=cols)
DEV=("2015-01-01","2022-12-31"); HOLD=("2023-01-01","2025-12-31"); FULL=("2015-01-01","2025-12-31")
def ann(r,sub=None):
    r=r.dropna()
    if sub: r=r[(r.index>=pd.Timestamp(sub[0]))&(r.index<=pd.Timestamp(sub[1]))]
    if len(r)<6: return (np.nan,)*3
    a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
def sec_neutralize(sig,fac):     # demean rank within deciles of a factor (e.g. size)
    rk=sig.where(LIQ).rank(axis=1,pct=True); fr=fac.where(LIQ).rank(axis=1,pct=True)
    bins=(fr*5).round(); out=rk.copy()
    for b in range(6):
        m=(bins==b); sub=rk.where(m); out=out.where(~m, rk-sub.mean(axis=1).values.reshape(-1,1))
    return out
def build(score,short_mask=SHORT,q=0.1,rebal=3,buffer=2.0,betaneutral=True,ntranche=1,wscheme="eq"):
    rkl=score.where(LIQ).rank(axis=1,pct=True); rks=score.where(short_mask).rank(axis=1,pct=True)
    def one(offset):
        Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols)
        hl=set(); hs=set(); cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
        for k,dt in enumerate(M):
            if (k-offset)%rebal==0 and k>=offset:
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
                cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
                if wscheme=="eq":
                    if sel_l: cl[sel_l]=1.0/len(sel_l)
                    if sel_s: cs[sel_s]=1.0/len(sel_s)
            b=BETA.loc[dt]
            if betaneutral:
                bl=(cl*b).sum(); bs=(cs*b).sum(); css=cs*(bl/bs) if bs>0.05 else cs
            else: css=cs
            Wl.loc[dt]=cl; Ws.loc[dt]=css
        return Wl,Ws
    if ntranche==1: return one(0)
    Wls=[];Wss=[]
    for off in range(ntranche):
        a,b=one(off); Wls.append(a); Wss.append(b)
    return sum(Wls)/ntranche, sum(Wss)/ntranche
def net(Wl,Ws,aum=100e6,exec_days=3,K=0.8,gross_scale=None):
    if gross_scale is not None: Wl=Wl.mul(gross_scale,axis=0); Ws=Ws.mul(gross_scale,axis=0)
    g=((Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1)).reindex(idx)
    traded=(Wl.diff().abs()+Ws.diff().abs())
    sp=(traded*spread/1e4).sum(axis=1).reindex(idx)
    part=(traded*aum)/(ADV*exec_days); imp=(traded*(K*sig_d*np.sqrt(part.clip(lower=0))*1e4)/1e4).sum(axis=1).reindex(idx)
    bc=(Ws.shift(1)*borrow_rate/12).sum(axis=1).reindex(idx)
    fin=((Wl.shift(1).sum(axis=1)+Ws.shift(1).sum(axis=1))*0.005/12).reindex(idx)
    return g-sp-imp-bc-fin
def row(nm,r):
    aF,sF,dF=ann(r,FULL); _,sD,_=ann(r,DEV); _,sH,_=ann(r,HOLD)
    print(f"{nm:36}{aF*100:>6.1f}{sF:>6.2f}{dF:>7.1%} |{sD:>6.2f}{sH:>6.2f} |{r.reindex(idx).corr(qret.reindex(idx)):>6.2f}",flush=True)
def hdr(): print(f"{'variant (all-in net @100M)':36}{'ann%':>6}{'Sh':>6}{'maxDD':>7} |{'devSh':>6}{'hldSh':>6} |{'cQQQ':>6}",flush=True); print("-"*92,flush=True)
