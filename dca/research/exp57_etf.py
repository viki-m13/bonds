import urllib.request, ssl, os, io, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
K=os.environ["TIINGO_KEY"]
def p(*a): print(*a,flush=True)
def fetch(tk):
    u=f"https://api.tiingo.com/tiingo/daily/{tk}/prices?startDate=2011-01-01&token={K}&format=csv&resampleFreq=daily"
    d=urllib.request.urlopen(urllib.request.Request(u),timeout=60,context=ctx).read().decode()
    df=pd.read_csv(io.StringIO(d)); df["date"]=pd.to_datetime(df.date)
    return df.set_index("date")["adjClose"].astype(float)
tks=["QQQ","SPY","USMV","SPLV","MTUM","VLUE","QUAL","EFAV","TLT","IEF"]
S={}
for t in tks:
    try: S[t]=fetch(t); p(f"got {t} ({len(S[t])})")
    except Exception as e: p(f"FAIL {t}: {e}")
px=pd.DataFrame(S).sort_index()
me=px.resample("ME").last(); me.index=me.index.to_period("M").to_timestamp()
me.to_pickle("/tmp/wave/_etf_me.pkl")
R={t:(me[t]/me[t].shift(1)-1) for t in me.columns}
def stats(r):
    r=r.dropna()
    if len(r)<6: return (np.nan,np.nan,np.nan)
    cagr=(1+r).prod()**(12/len(r))-1; sh=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return cagr,sh,dd
idx=me.dropna(subset=["QQQ","USMV"]).index
idx=idx[(idx>=pd.Timestamp("2012-01-01"))&(idx<=pd.Timestamp("2025-12-31"))]
q=R["QQQ"].reindex(idx)
p(f"\nwindow {idx[0].date()}..{idx[-1].date()} n={len(idx)}")
p(f"{'asset':>8} CAGR Sharpe maxDD corrQQQ")
for t in me.columns:
    r=R[t].reindex(idx); c,s,d=stats(r)
    p(f"{t:>8} {c:>5.1%} {s:>5.2f} {d:>6.1%} {r.corr(q):+.2f}")
def dcaS(r,c=1000.0):
    v=0.0; out=[]
    for x in r.fillna(0): v=v*(1+x)+c; out.append(v)
    return pd.Series(out,index=r.index)
# blends QQQ + USMV
p("\nQQQ + USMV:")
p(f"{'w':>5} CAGR Sharpe maxDD   DCA")
for w in [0,.1,.2,.3,.4,.5]:
    bl=(1-w)*q+w*R["USMV"].reindex(idx); c,s,d=stats(bl)
    p(f"{w:>5.0%} {c:>5.1%} {s:>5.2f} {d:>6.1%}  ${dcaS(bl).iloc[-1]:,.0f}")
# three-way QQQ/USMV/TLT
p("\n3-way QQQ/USMV/TLT (diversified):")
for wq,wu,wt in [(0.6,0.3,0.1),(0.5,0.3,0.2),(0.6,0.2,0.2),(0.5,0.4,0.1)]:
    bl=wq*q+wu*R["USMV"].reindex(idx)+wt*R["TLT"].reindex(idx); c,s,d=stats(bl)
    p(f"  {wq:.0%}/{wu:.0%}/{wt:.0%}: CAGR {c:>5.1%} Sharpe {s:>5.2f} maxDD {d:>6.1%}  DCA ${dcaS(bl).iloc[-1]:,.0f}")
# sub-period
bl30=0.7*q+0.3*R["USMV"].reindex(idx)
p("\nSub-period Sharpe (QQQ vs 70/30):")
for lo,hi in [("2012","2015"),("2016","2019"),("2020","2022"),("2023","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31"))
    p(f"  {lo}-{hi}: QQQ {stats(q[m])[1]:>5.2f}  blend {stats(bl30[m])[1]:>5.2f}")
# figure
qd=dcaS(q); bd=dcaS(bl30); cum=np.arange(1,len(idx)+1)*1000.0
cq,sq,dq=stats(q); cb,sb,db=stats(bl30); cu,su,du=stats(R["USMV"].reindex(idx))
fig,ax=plt.subplots(1,2,figsize=(15,6))
gq=(1+q).cumprod(); gu=(1+R["USMV"].reindex(idx)).cumprod(); gb=(1+bl30).cumprod()
ax[0].plot(gq.index,gq,label=f"QQQ (Sh {sq:.2f}, DD {dq:.0%})",lw=2,color="#888")
ax[0].plot(gu.index,gu,label=f"USMV low-vol (Sh {su:.2f}, DD {du:.0%})",lw=2,color="#2ca02c")
ax[0].plot(gb.index,gb,label=f"QQQ70/USMV30 (Sh {sb:.2f}, DD {db:.0%})",lw=2.4,color="#d62728")
ax[0].set_yscale("log"); ax[0].set_title("Growth of $1 (log)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
ax[1].plot(qd.index,qd/1e3,label=f"QQQ DCA → ${qd.iloc[-1]/1e3:.0f}k",lw=2,color="#888")
ax[1].plot(bd.index,bd/1e3,label=f"QQQ70/USMV30 DCA → ${bd.iloc[-1]/1e3:.0f}k",lw=2.4,color="#d62728")
ax[1].plot(idx,cum/1e3,label=f"Contributed → ${cum[-1]/1e3:.0f}k",lw=1.2,ls="--",color="#bbb")
ax[1].set_title("$1,000/mo DCA — account value ($k)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
fig.suptitle("Real tradeable low-vol overlay (USMV ETF) on a QQQ DCA plan — 2012–2025",fontsize=12)
fig.tight_layout(); fig.savefig("/tmp/wave/lowvol_overlay.png",dpi=110)
p("\nsaved /tmp/wave/lowvol_overlay.png")
