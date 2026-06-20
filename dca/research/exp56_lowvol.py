import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
def stats(r):
    r=r.dropna()
    if len(r)<6: return (np.nan,np.nan,np.nan)
    cagr=(1+r).prod()**(12/len(r))-1; sh=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return cagr,sh,dd
have=[t for t in ["QQQ","USMV","SPLV","SPY"] if t in me.columns]
p("ETFs available:",have)
R={t:(me[t]/me[t].shift(1)-1) for t in have}
# common window where USMV exists
base=me[["QQQ"]+ [t for t in ["USMV","SPLV"] if t in me]].dropna()
idx=base.index[(base.index>=pd.Timestamp("2012-01-01"))&(base.index<=pd.Timestamp("2025-12-31"))]
q=R["QQQ"].reindex(idx)
p(f"\nwindow {idx[0].date()}..{idx[-1].date()} n={len(idx)}")
p(f"{'asset':>16} CAGR Sharpe maxDD  corrQQQ")
for t in have:
    r=R[t].reindex(idx); c,s,d=stats(r); cc=r.corr(q)
    p(f"{t:>16} {c:>5.1%} {s:>5.2f} {d:>6.1%}  {cc:+.2f}")
# blends QQQ + USMV (and SPLV)
lv = "USMV" if "USMV" in R else "SPLV"
p(f"\nQQQ + {lv} blends:")
p(f"{'w_lv':>6} CAGR Sharpe maxDD   DCA$1k/mo")
def dca(r,c=1000.0):
    v=0.0
    for x in r.fillna(0): v=v*(1+x)+c
    return v
contrib=len(idx)*1000.0
best=None
for w in [0,0.1,0.2,0.3,0.4,0.5]:
    bl=(1-w)*q+w*R[lv].reindex(idx); c,s,d=stats(bl); fv=dca(bl)
    p(f"{w:>6.0%} {c:>5.1%} {s:>5.2f} {d:>6.1%}   ${fv:,.0f}")
# sub-period Sharpe stability for 70/30
p(f"\nSub-period Sharpe (QQQ vs QQQ70/{lv}30):")
bl30=0.7*q+0.3*R[lv].reindex(idx)
for lo,hi in [("2012","2015"),("2016","2019"),("2020","2022"),("2023","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31"))
    _,sq,_=stats(q[m]); _,sb,_=stats(bl30[m])
    p(f"  {lo}-{hi}:  QQQ {sq:>5.2f}   blend {sb:>5.2f}")
# figure: DCA curves
def dcaS(r,c=1000.0):
    v=0.0; out=[]
    for x in r.fillna(0): v=v*(1+x)+c; out.append(v)
    return pd.Series(out,index=r.index)
qd=dcaS(q); bd=dcaS(bl30); cum=np.arange(1,len(idx)+1)*1000.0
cq,sq,dq=stats(q); cb,sb,db=stats(bl30)
fig,ax=plt.subplots(1,2,figsize=(15,6))
gq=(1+q).cumprod(); gl=(1+R[lv].reindex(idx)).cumprod(); gb=(1+bl30).cumprod()
ax[0].plot(gq.index,gq,label=f"QQQ (Sh {sq:.2f}, DD {dq:.0%})",lw=2,color="#888")
ax[0].plot(gl.index,gl,label=f"{lv} low-vol ETF (Sh {stats(R[lv].reindex(idx))[1]:.2f}, DD {stats(R[lv].reindex(idx))[2]:.0%})",lw=2,color="#2ca02c")
ax[0].plot(gb.index,gb,label=f"QQQ70/{lv}30 (Sh {sb:.2f}, DD {db:.0%})",lw=2.4,color="#d62728")
ax[0].set_yscale("log"); ax[0].set_title("Growth of $1 (log)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
ax[1].plot(qd.index,qd/1e3,label=f"QQQ DCA → ${qd.iloc[-1]/1e3:.0f}k",lw=2,color="#888")
ax[1].plot(bd.index,bd/1e3,label=f"QQQ70/{lv}30 DCA → ${bd.iloc[-1]/1e3:.0f}k",lw=2.4,color="#d62728")
ax[1].plot(idx,cum/1e3,label=f"Contributed → ${cum[-1]/1e3:.0f}k",lw=1.2,ls="--",color="#bbb")
ax[1].set_title("$1,000/mo DCA — account value ($k)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
fig.suptitle(f"Low-volatility ETF overlay: QQQ-DCA + {lv} raises Sharpe & cuts drawdown (Tiingo clean data)",fontsize=12)
fig.tight_layout(); fig.savefig("/tmp/wave/lowvol_overlay.png",dpi=110)
p("\nsaved /tmp/wave/lowvol_overlay.png")
