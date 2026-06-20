import numpy as np, pandas as pd, time, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P["ym"]=pd.to_datetime(P.ym)
names=[c for c in me.columns if c in set(P.tk)]; P=P[P.tk.isin(set(names))]
def pan(col): return P.pivot_table(index="ym",columns="tk",values=col,aggfunc="sum").reindex(index=me.index,columns=names).fillna(0)
buy,sell,nb,offb,ceob=pan("buy"),pan("sell"),pan("nbuyers"),pan("off_buy"),pan("ceo_buy")
nb3=nb.rolling(3,min_periods=1).sum(); buy3=buy.rolling(3,min_periods=1).sum()
off3=offb.rolling(3,min_periods=1).sum(); ceo3=ceob.rolling(3,min_periods=1).sum()
mp=me[names]; ret=(mp/mp.shift(1)-1).clip(-0.90,2.0); px_ok=mp.shift(1)>=3.0
p(f"panels built t={time.time()-t0:.0f}s")
def zc(x):
    m=x.where(x>0); return (m.sub(m.mean(axis=1),axis=0)).div(m.std(axis=1).replace(0,np.nan),axis=0)
score=(zc(nb3).fillna(0)+zc(buy3).fillna(0)+0.5*(off3>0)+0.5*(ceo3>0)).where((buy3>0)&px_ok)
start=pd.Timestamp("2011-01-01"); end=pd.Timestamp("2025-12-31")
idx=me.index[(me.index>=start)&(me.index<=end)]
def stats(r):
    r=r.dropna(); cagr=(1+r).prod()**(12/len(r))-1; sh=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return cagr,sh,dd
qret=(me["QQQ"]/me["QQQ"].shift(1)-1).reindex(idx)
def topN(N):
    sel=(score.rank(axis=1,ascending=False)<=N)
    w=sel.shift(1).fillna(False).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w*ret).sum(axis=1).reindex(idx)
p(f"{'sleeve':>10} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
res={}
for N in [10,20,30,50,100]:
    r=topN(N); res[N]=r; c,s,d=stats(r); p(f"top{N:<7d} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
c,s,d=stats(qret); p(f"{'QQQ':>10} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
for b in ["IWM","MDY"]:
    if b in me: c,s,d=stats((me[b]/me[b].shift(1)-1).reindex(idx)); p(f"{b:>10} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
# L/S: top quintile vs rest of buyers
rk=score.rank(axis=1,ascending=False,pct=True); anyb=(buy3>0)&px_ok
longs=(rk<=0.20); rest=anyb&(~longs)
wl=longs.shift(1).fillna(False).astype(float); wl=wl.div(wl.sum(axis=1).replace(0,np.nan),axis=0)
ws=rest.shift(1).fillna(False).astype(float); ws=ws.div(ws.sum(axis=1).replace(0,np.nan),axis=0)
ls=((wl*ret).sum(axis=1)-(ws*ret).sum(axis=1)).reindex(idx)
c,s,d=stats(ls); p(f"\nL/S top20%-vs-rest: CAGR {c:.1%} Sharpe {s:.2f} maxDD {d:.1%}")
# overlay: QQQ core + concentrated insider tilt
ins=res[20]
for wt in [0.0,0.10,0.20,0.30]:
    blend=(1-wt)*qret+wt*ins; c,s,d=stats(blend); p(f"QQQ {1-wt:.0%}+ins {wt:.0%}: CAGR {c:.1%} Sharpe {s:.2f} maxDD {d:.1%}")
# ---- DCA $1000/mo ----
def dca(r,c=1000.0):
    v=0.0; out=[]
    for x in r.fillna(0): v=v*(1+x)+c; out.append(v)
    return pd.Series(out,index=r.index)
blend20=0.8*qret+0.2*ins
qd=dca(qret); bd=dca(blend20); contrib=np.arange(1,len(idx)+1)*1000.0
p(f"\nDCA $1k/mo, contributed ${contrib[-1]:,.0f}:")
p(f"  QQQ          ${qd.iloc[-1]:,.0f}")
p(f"  QQQ80+ins20  ${bd.iloc[-1]:,.0f}")
# ---- figure ----
fig,ax=plt.subplots(1,2,figsize=(15,6))
gq=(1+qret).cumprod(); gi=(1+ins).cumprod(); gl=(1+ls).cumprod()
ci,si,di=stats(ins); cq,sq,dq=stats(qret); cl,sl,dl=stats(ls)
ax[0].plot(gq.index,gq,label=f"QQQ (Sh {sq:.2f})",lw=2,color="#888")
ax[0].plot(gi.index,gi,label=f"Insider top-20 concentrated (Sh {si:.2f})",lw=2,color="#1f77b4")
ax[0].plot(gl.index,gl,label=f"Insider L/S edge: top20% vs rest (Sh {sl:.2f})",lw=2,color="#2ca02c")
ax[0].set_yscale("log"); ax[0].set_title("Growth of $1 (log)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
cb,sb,db=stats(blend20)
ax[1].plot(qd.index,qd/1e3,label=f"QQQ DCA → ${qd.iloc[-1]/1e3:.0f}k (Sh {sq:.2f}, DD {dq:.0%})",lw=2,color="#888")
ax[1].plot(bd.index,bd/1e3,label=f"QQQ 80% + insider 20% DCA → ${bd.iloc[-1]/1e3:.0f}k (Sh {sb:.2f}, DD {db:.0%})",lw=2,color="#d62728")
ax[1].plot(idx,contrib/1e3,label=f"Contributed → ${contrib[-1]/1e3:.0f}k",lw=1.2,ls="--",color="#bbb")
ax[1].set_title("$1,000/mo DCA — account value ($k)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
fig.suptitle("Insider signal: real cross-sectional edge, best used as a tilt on a QQQ core  (Tiingo clean data, 2011–2025)",fontsize=12)
fig.tight_layout(); fig.savefig("/tmp/wave/insider_v2.png",dpi=110)
p(f"\nsaved /tmp/wave/insider_v2.png t={time.time()-t0:.0f}s")
