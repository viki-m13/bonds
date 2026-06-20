import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")          # monthly adjClose panel (PIT, delisting-incl)
me=me.loc[:,~me.columns.duplicated()]
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P["ym"]=pd.to_datetime(P.ym)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
uni["endDate"]=pd.to_datetime(uni.endDate,errors="coerce")
deli=set(uni[(uni.assetType=="Stock")&(uni.endDate<'2025-01-01')].ticker)

names=[c for c in me.columns if c in set(P.tk)]
P=P[P.tk.isin(set(names))]
def pan(col): return P.pivot_table(index="ym",columns="tk",values=col,aggfunc="sum").reindex(index=me.index,columns=names).fillna(0)
buy,sell,nb,offb=pan("buy"),pan("sell"),pan("nbuyers"),pan("off_buy")
nb3=nb.rolling(3,min_periods=1).sum()
buy3=buy.rolling(3,min_periods=1).sum()
off3=offb.rolling(3,min_periods=1).sum()
bigthr=buy3.where(buy3>0).quantile(0.7,axis=1)

# signal: cluster>=2 OR large-$ buy (the verified-robust variants from exp51), among insider names
SIG=(nb3>=2)|buy3.gt(bigthr,axis=0)

# monthly returns on the insider names, winsorized; price>=3 filter on entry month
mp=me[names]
ret=(mp/mp.shift(1)-1).clip(-0.90,2.0)
elig=SIG.shift(1).fillna(False) & (mp.shift(1)>=3.0)    # signal & price known at prior month-end
w=elig.astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
port=(w*ret).sum(axis=1)                                  # equal-weight monthly return of insider sleeve
nhold=elig.sum(axis=1)

# QQQ benchmark
qqq=me["QQQ"].dropna(); qret=(qqq/qqq.shift(1)-1)

# common window where we actually hold names
start=pd.Timestamp("2011-01-01"); end=pd.Timestamp("2025-12-31")
idx=port.index[(port.index>=start)&(port.index<=end)&(nhold>=10)]
pr=port.reindex(idx).fillna(0); qr=qret.reindex(idx).fillna(0)

def stats(r):
    cagr=(1+r).prod()**(12/len(r))-1
    sh=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return cagr,sh,dd
ic,ish,idd=stats(pr); qc,qsh,qdd=stats(qr)
print(f"window {idx[0].date()}..{idx[-1].date()}  n={len(idx)}  avg#held={nhold.reindex(idx).mean():.0f}")
print(f"INSIDER  CAGR {ic:6.1%}  Sharpe {ish:4.2f}  maxDD {idd:6.1%}")
print(f"QQQ      CAGR {qc:6.1%}  Sharpe {qsh:4.2f}  maxDD {qdd:6.1%}")

# ---- DCA: $1000 contributed each month into each strategy ----
def dca(r, c=1000.0):
    eq=[]; v=0.0
    for x in r:
        v=v*(1+x)+c; eq.append(v)
    return pd.Series(eq,index=r.index)
ins_dca=dca(pr); qqq_dca=dca(qr)
contrib=np.arange(1,len(idx)+1)*1000.0
print(f"\nDCA $1000/mo over {len(idx)} mo (contributed ${contrib[-1]:,.0f}):")
print(f"  INSIDER final ${ins_dca.iloc[-1]:,.0f}  ({ins_dca.iloc[-1]/contrib[-1]:.2f}x)")
print(f"  QQQ     final ${qqq_dca.iloc[-1]:,.0f}  ({qqq_dca.iloc[-1]/contrib[-1]:.2f}x)")

# ---- plot ----
fig,ax=plt.subplots(1,2,figsize=(15,6))
g_i=(1+pr).cumprod(); g_q=(1+qr).cumprod()
ax[0].plot(g_i.index,g_i,label=f"Insider sleeve (Sharpe {ish:.2f}, CAGR {ic:.0%})",lw=2,color="#1f77b4")
ax[0].plot(g_q.index,g_q,label=f"QQQ (Sharpe {qsh:.2f}, CAGR {qc:.0%})",lw=2,color="#888")
ax[0].set_yscale("log"); ax[0].set_title("Growth of $1 (lump-sum, log scale)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ins_dca.index,ins_dca/1e3,label=f"Insider DCA  →  ${ins_dca.iloc[-1]/1e3:.0f}k",lw=2,color="#1f77b4")
ax[1].plot(qqq_dca.index,qqq_dca/1e3,label=f"QQQ DCA  →  ${qqq_dca.iloc[-1]/1e3:.0f}k",lw=2,color="#888")
ax[1].plot(idx,contrib/1e3,label=f"Contributed  →  ${contrib[-1]/1e3:.0f}k",lw=1.2,ls="--",color="#bbb")
ax[1].set_title("$1,000/mo DCA — account value ($k)"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle("Insider-buying strategy vs QQQ  (Tiingo delisting-inclusive data, 2011–2025)",fontsize=13)
fig.tight_layout(); fig.savefig("/tmp/wave/insider_vs_qqq.png",dpi=110)
print(f"\nsaved /tmp/wave/insider_vs_qqq.png  t={time.time()-t0:.0f}s")
