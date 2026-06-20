import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]
me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
stocks=set(uni[uni.assetType=="Stock"].ticker)
cols=[c for c in me.columns if c in stocks]
px=me[cols]
ret=(px/px.shift(1)-1)
fwd=ret.shift(-1).clip(-0.9,2.0)              # next-month return (winsorized)
liq=(px.shift(1)>=5.0)                          # price>=$5 liquidity/penny filter at decision time
start=pd.Timestamp("2011-01-01"); end=pd.Timestamp("2025-12-31")
idx=me.index[(me.index>=start)&(me.index<=end)]
qret=(me["QQQ"]/me["QQQ"].shift(1)-1).reindex(idx)
def stats(r):
    r=r.dropna();
    if len(r)<12: return (np.nan,np.nan,np.nan)
    cagr=(1+r).prod()**(12/len(r))-1; sh=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return cagr,sh,dd
def sleeve(sig, dec=0.1, longtop=True):
    s=sig.where(liq)
    rk=s.rank(axis=1,ascending=not longtop,pct=True)  # top decile = best
    sel=(rk<=dec)
    w=sel.shift(1).fillna(False).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w*fwd.shift(1)).sum(axis=1).reindex(idx)   # fwd.shift(1)=this-month realized ret aligned

# careful return alignment: hold names selected at t-1, earn ret at t
def sleeve2(sig, dec=0.1, best_high=True):
    s=sig.where(liq)
    rk=s.rank(axis=1,ascending=not best_high,pct=True)
    sel=(rk<=dec).shift(1).fillna(False)
    w=sel.astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w*ret).sum(axis=1).reindex(idx), sel.sum(axis=1).reindex(idx)

mom12=px.shift(1)/px.shift(12)-1               # 12-1 momentum
mom6=px.shift(1)/px.shift(6)-1
rev1=-(px.shift(1)/px.shift(2)-1)              # 1-month reversal (buy losers)
vol6=ret.rolling(6).std()                       # volatility
FAC={"mom12 (winners)":(mom12,True),"mom6 (winners)":(mom6,True),
     "rev1 (buy losers)":(rev1,True),"lowvol":(vol6,False)}
c,s,d=stats(qret); p(f"{'QQQ':>22} CAGR {c:>6.1%} Sharpe {s:>5.2f} maxDD {d:>6.1%}")
p("-"*64)
cq,sq,dq=c,s,d
sleeves={}
for nm,(sig,bh) in FAC.items():
    r,n=sleeve2(sig,0.1,bh); sleeves[nm]=r
    c,s,d=stats(r); p(f"{nm:>22} CAGR {c:>6.1%} Sharpe {s:>5.2f} maxDD {d:>6.1%}  avg#{n.mean():.0f}")
p("-"*64)
# combine each with QQQ 70/30 and check Sharpe
p("QQQ 70% + factor 30%:")
for nm,r in sleeves.items():
    bl=0.7*qret+0.3*r; c,s,d=stats(bl)
    flag="  <-- beats QQQ Sharpe" if s>sq else ""
    p(f"{nm:>22} CAGR {c:>6.1%} Sharpe {s:>5.2f} maxDD {d:>6.1%}{flag}")
# corr to QQQ
p("-"*64); p("corr to QQQ:")
for nm,r in sleeves.items():
    p(f"{nm:>22} {r.corr(qret):+.2f}")
p(f"\nt={time.time()-t0:.0f}s")
