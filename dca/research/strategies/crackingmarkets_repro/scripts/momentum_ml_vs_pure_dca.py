import pandas as pd, numpy as np, warnings, time; warnings.filterwarnings("ignore")
t0=time.time()
F=pd.read_pickle("/tmp/wave/_featmat.pkl"); liq,me,cols=F["liq"],F["me"],F["cols"]
M=me.index; ret=(me/me.shift(1)-1).clip(-0.95,5.0)
mom=me.shift(1)/me.shift(12)-1
U=(liq&(me.shift(1)>=5.0)).fillna(False)
ma10=me.rolling(10,min_periods=10).mean()
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(index=M,columns=cols)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]; didx=list(idx); N=15
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].reindex(idx); qret=qpx.pct_change().fillna(0)
SIG={"momentum":mom.where(U).rank(axis=1,ascending=False),
     "ML quality":PROB.where(U).rank(axis=1,ascending=False)}
def stats(r):
    r=r.dropna(); n=len(r); c=(1+r).prod()**(12/n)-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return c,s,dd
def sim(rk,rule,cost=0.0010):
    pos={}; out=[]; holds=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]; r=rk.loc[dt]; sells=0
        for t in list(pos.keys()):
            e=pos[t]; cpx=px.get(t,np.nan)
            if not np.isfinite(cpx): holds.append(k-e["i"]); pos.pop(t); continue
            if cpx>e["peak"]: e["peak"]=cpx; 
            ex=False
            if rule=="rotation": ex=r.get(t,9e9)>N
            elif rule=="ride":   ex=(cpx<ma10.loc[dt].get(t,np.nan)) or (cpx/e["peak"]-1<=-0.30)
            if ex: holds.append(k-e["i"]); sells+=1; pos.pop(t)
        if rule!="hold4ever" or k==0:
            for t in r[r<=120].sort_values().index:
                if len(pos)>=N: break
                if t not in pos and np.isfinite(px.get(t,np.nan)) and bool(U.loc[dt].get(t,False)): pos[t]={"i":k,"peak":px[t]}
        if k+1<len(didx):
            r1=ret.loc[didx[k+1]]; pr=np.mean([r1.get(t,-0.5) for t in pos]) if pos else 0.0
            out.append((didx[k+1],pr-(sells/N)*cost))
    return pd.Series(dict(out)),(np.mean(holds) if holds else np.nan)
res={}
print(f"{'signal × exit rule (top-15, 2015-2025, net 10bps)':46}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'avgHold':>9}")
for sn,rk in SIG.items():
    for rule in ["rotation","ride","hold4ever"]:
        s,avh=sim(rk,rule); c,sh,dd=stats(s); res[(sn,rule)]=s
        print(f"  {sn+' + '+rule:44}{c*100:>6.1f}%{sh:>7.2f}{dd*100:>6.0f}%{avh:>9.1f}")
c,sh,dd=stats(qret); print(f"  {'QQQ (benchmark)':44}{c*100:>6.1f}%{sh:>7.2f}{dd*100:>6.0f}%")
# DCA: contribute $1/mo; V_t=(V_{t-1}+1)*(1+r_t). Best ML strat vs QQQ.
best=max([k for k in res if k[0]=="ML quality"], key=lambda k:stats(res[k])[1])
print(f"\nbest ML strat: {best[1]}")
def dca(r):
    V=[0.0]; 
    for x in r.reindex(idx).fillna(0).values: V.append((V[-1]+1)*(1+x))
    return pd.Series(V[1:],index=idx)
dstrat=dca(res[best]); dq=dca(qret); contrib=pd.Series(np.arange(1,len(idx)+1),index=idx)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(11,5.5))
ax.plot(dstrat.index,dstrat,color="#111",lw=2.4,label=f"DCA into WAVE-style (ML quality + {best[1]})  →  ${dstrat.iloc[-1]:.0f}")
ax.plot(dq.index,dq,color="#2563eb",lw=2.0,label=f"DCA into QQQ  →  ${dq.iloc[-1]:.0f}")
ax.plot(contrib.index,contrib,color="#888",lw=1.3,ls="--",label=f"Total contributed (cost basis)  →  ${contrib.iloc[-1]:.0f}")
ax.set_title("$1/month DCA: ML-quality ride-winners book vs QQQ (2015–2025, net of 10bps costs)")
ax.grid(alpha=.3); ax.legend(loc="upper left",fontsize=9); fig.tight_layout(); fig.savefig("/home/user/dca_wave_vs_qqq.png",dpi=120)
mult=lambda d: d.iloc[-1]/contrib.iloc[-1]
print(f"\n$1/mo DCA outcome: WAVE-style ${dstrat.iloc[-1]:.0f} ({mult(dstrat):.2f}x contributed) vs QQQ ${dq.iloc[-1]:.0f} ({mult(dq):.2f}x)  | contributed ${contrib.iloc[-1]:.0f}")
print(f"DONE t={time.time()-t0:.0f}s")
