import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
stocks=set(uni[uni.assetType=="Stock"].ticker)
px=me[[c for c in me.columns if c in stocks]].astype(float)
ret=(px/px.shift(1)-1); fret=ret.clip(-0.9,2.0)
mom12=px/px.shift(12)-1; mom6=px/px.shift(6)-1
hi12=px.rolling(12).max(); pctHigh=px/hi12
vol6=ret.rolling(6).std()
liq=(px.shift(1)>=3.0)
def z(x):
    return (x.sub(x.mean(axis=1),axis=0)).div(x.std(axis=1).replace(0,np.nan),axis=0)
idx=px.index[(px.index>=pd.Timestamp("2011-01-01"))&(px.index<=pd.Timestamp("2025-12-31"))]
q=(me["QQQ"]/me["QQQ"].shift(1)-1).reindex(idx)
def stats(r):
    r=r.dropna();
    if len(r)<12: return (np.nan,)*3
    c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def port(score,N,longtop=True):
    sc=score.where(liq); rk=sc.rank(axis=1,ascending=not longtop)
    sel=(rk<=N).shift(1).fillna(False)
    w=sel.astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w*fret).sum(axis=1).reindex(idx)
c,s,d=stats(q); p(f"{'QQQ':<28} CAGR {c:>6.1%} Sharpe {s:>5.2f} maxDD {d:>6.1%}")
p("-"*64)
# Archetype 1: LOTTERY (high vol, beaten down, cheap)
lott = z(vol6).fillna(0) - z(pctHigh).fillna(0) - z(np.log(px)).fillna(0)
# Archetype 2: BREAKOUT (strong momentum + near highs)
brk = z(mom12).fillna(0) + z(mom6).fillna(0) + z(pctHigh).fillna(0)
# Archetype 3: low-vol quality breakout (near high but NOT crazy vol)
qbrk = z(mom12).fillna(0) + z(pctHigh).fillna(0) - z(vol6).fillna(0)
for nm,sc in [("LOTTERY hi-vol/cheap/beaten",lott),("BREAKOUT mom+near-high",brk),("QUALITY-BREAKOUT low-vol",qbrk)]:
    for N in [20,50]:
        c,s,d=stats(port(sc,N)); p(f"{nm[:22]:<22} top{N:<3} CAGR {c:>6.1%} Sharpe {s:>5.2f} maxDD {d:>6.1%}")
    p("")
# Best breakout blended with QQQ
p("-"*64); p("BREAKOUT top50 blended w/ QQQ:")
b=port(brk,50)
for w in [0.2,0.3,0.5]:
    c,s,d=stats((1-w)*q+w*b); p(f"  QQQ{1-w:.0%}/brk{w:.0%}: CAGR {c:>6.1%} Sharpe {s:>5.2f} maxDD {d:>6.1%}")
p(f"\nDONE t={time.time()-t0:.0f}s")
