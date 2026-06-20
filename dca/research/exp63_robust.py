import numpy as np, pandas as pd, time, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
stocks=set(uni[uni.assetType=="Stock"].ticker)
CACHE="/tmp/wave/_qual_masks.pkl"
if os.path.exists(CACHE):
    ACC2,HY2,INS2,TECH2=pd.read_pickle(CACHE); p(f"loaded cached masks t={time.time()-t0:.0f}s")
else:
    rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
    qend=pd.PeriodIndex([q[2:] for q in rev.index],freq="Q").to_timestamp(how="end").normalize(); rev.index=qend
    yoy=rev/rev.shift(4)-1
    accel=((yoy.diff()>0)&(yoy.diff().shift(1)>0)&(yoy>0)); highyoy=(yoy>=0.25)
    avail=(rev.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    def to_monthly(qf):
        qf=qf.copy(); qf.index=avail; qf=qf[~qf.index.duplicated(keep="last")]
        return qf.reindex(me.index,method="ffill",limit=6)
    ACC=to_monthly(accel.astype(float)).fillna(0)>0.5; HY=to_monthly(highyoy.astype(float)).fillna(0)>0.5
    P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P["ym"]=pd.to_datetime(P.ym)
    inames=[c for c in me.columns if c in set(P.tk)]; P=P[P.tk.isin(set(inames))]
    def ipan(col): return P.pivot_table(index="ym",columns="tk",values=col,aggfunc="sum").reindex(index=me.index,columns=me.columns).fillna(0)
    nb=ipan("nbuyers"); buy=ipan("buy"); nb3=nb.rolling(3,min_periods=1).sum(); buy3=buy.rolling(3,min_periods=1).sum()
    bigthr=buy3.where(buy3>0).quantile(0.7,axis=1); INS=(nb3>=2)|buy3.gt(bigthr,axis=0)
    mom6=me/me.shift(6)-1; ma10=me.rolling(10,min_periods=10).mean(); TECH=(me>ma10)&(mom6>0)
    def alg(df): return df.reindex(columns=me.columns).reindex(index=me.index)
    ACC2=alg(ACC).fillna(False); HY2=alg(HY).fillna(False); INS2=alg(INS).fillna(False); TECH2=alg(TECH).fillna(False)
    pd.to_pickle((ACC2,HY2,INS2,TECH2),CACHE); p(f"built+cached masks t={time.time()-t0:.0f}s")
cols=[c for c in me.columns if c in stocks]
ret=(me/me.shift(1)-1).clip(-0.9,2.0); liq=(me.shift(1)>=3.0).fillna(False)
idx=me.index[(me.index>=pd.Timestamp("2012-07-01"))&(me.index<=pd.Timestamp("2025-12-31"))]
qret=(me["QQQ"]/me["QQQ"].shift(1)-1).reindex(idx)
ENS=((ACC2|HY2)&INS2&TECH2&liq)[cols]
def port_ret(mask):
    w=mask.shift(1).fillna(False).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w*ret[cols]).sum(axis=1).reindex(idx), w
er,W=port_ret(ENS)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
c,s,d=stats(er); qc,qs,qd=stats(qret)
p(f"\nENSEMBLE  CAGR {c:.1%} Sharpe {s:.2f} maxDD {d:.1%}   names/mo {ENS.sum(axis=1).reindex(idx).mean():.0f}")
p(f"QQQ       CAGR {qc:.1%} Sharpe {qs:.2f} maxDD {qd:.1%}")
p(f"corr(ENS,QQQ)={er.corr(qret):.2f}")
# annual returns
p("\nAnnual returns (ENS vs QQQ):")
for y in range(2013,2026):
    m=(idx.year==y);
    if m.sum()<6: continue
    ey=(1+er[m]).prod()-1; qy=(1+qret[m]).prod()-1
    p(f"  {y}: ENS {ey:+6.1%}   QQQ {qy:+6.1%}   {'WIN' if ey>qy else ''}")
# sub-periods
p("\nSub-period CAGR / Sharpe:")
for lo,hi in [("2012","2016"),("2017","2020"),("2021","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31"))
    cc,ss,dd=stats(er[m]); qcc,qss,qdd=stats(qret[m])
    p(f"  {lo}-{hi}: ENS {cc:+6.1%}/{ss:.2f}   QQQ {qcc:+6.1%}/{qss:.2f}")
# robustness: exclude the AI-boom window 2023-2024 entirely
mex=~((idx>=pd.Timestamp('2023-01-01'))&(idx<=pd.Timestamp('2024-12-31')))
cc,ss,dd=stats(er[mex]); qcc,qss,qdd=stats(qret[mex])
p(f"\nExcluding 2023-24 (AI boom): ENS {cc:+.1%}/{ss:.2f}  QQQ {qcc:+.1%}/{qss:.2f}")
# name concentration: top contributors (sum of weight*ret)
contrib=(W*ret[cols]).reindex(idx).sum().sort_values(ascending=False)
p(f"\nTop 12 contributing names (sum monthly w*ret):")
for nm,v in contrib.head(12).items(): p(f"   {nm:6} {v:+.2f}")
# how much CAGR survives if we drop the single best ticker each
def cagr_drop(drop):
    m2=ENS.copy();
    for dnm in drop:
        if dnm in m2.columns: m2[dnm]=False
    r2,_=port_ret(m2); cc,ss,_=stats(r2); return cc,ss
top5=list(contrib.head(5).index)
cc,ss=cagr_drop(top5); p(f"\nDrop top-5 names {top5}: ENS CAGR {cc:.1%} Sharpe {ss:.2f}")
# turnover (avg fraction of book replaced/mo)
turn=(W.diff().abs().sum(axis=1)/2).reindex(idx).mean()
p(f"avg monthly turnover ~{turn:.0%}  (cost @ 20bps roundtrip ~{turn*0.002*12*100:.1f}%/yr drag)")
# equity curve (DCA + growth)
def dcaS(r,c=1000.0):
    v=0.0;out=[]
    for x in r.fillna(0): v=v*(1+x)+c; out.append(v)
    return pd.Series(out,index=r.index)
ed=dcaS(er); qd2=dcaS(qret); cum=np.arange(1,len(idx)+1)*1000.0
fig,ax=plt.subplots(1,2,figsize=(15,6))
ge=(1+er).cumprod(); gq=(1+qret).cumprod()
ax[0].plot(ge.index,ge,label=f"Qualifier ensemble (CAGR {c:.0%}, Sh {s:.2f}, DD {d:.0%})",lw=2.2,color="#d62728")
ax[0].plot(gq.index,gq,label=f"QQQ (CAGR {qc:.0%}, Sh {qs:.2f}, DD {qd:.0%})",lw=2,color="#888")
ax[0].set_yscale("log"); ax[0].set_title("Growth of $1 (log)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
ax[1].plot(ed.index,ed/1e3,label=f"Ensemble DCA → ${ed.iloc[-1]/1e3:.0f}k",lw=2.2,color="#d62728")
ax[1].plot(qd2.index,qd2/1e3,label=f"QQQ DCA → ${qd2.iloc[-1]/1e3:.0f}k",lw=2,color="#888")
ax[1].plot(idx,cum/1e3,label=f"Contributed → ${cum[-1]/1e3:.0f}k",lw=1.2,ls="--",color="#bbb")
ax[1].set_title("$1,000/mo DCA — account value ($k)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
fig.suptitle("Reverse-engineered moonshot qualifier: (rev-accel | high-YoY) & insider-cluster & uptrend\nPIT survivorship-clean, 2012-2025, ~19 names/mo, monthly rebal",fontsize=11)
fig.tight_layout(); fig.savefig("/tmp/wave/qualifier_vs_qqq.png",dpi=110)
p(f"\nsaved /tmp/wave/qualifier_vs_qqq.png  t={time.time()-t0:.0f}s")
