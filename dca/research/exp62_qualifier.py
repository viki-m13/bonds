import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
# ---- monthly PIT price panel ----
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
stocks=set(uni[uni.assetType=="Stock"].ticker)
# ---- SEC quarterly revenue -> monthly YoY / acceleration (reporting-lagged) ----
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
qend=pd.PeriodIndex([q[2:] for q in rev.index],freq="Q").to_timestamp(how="end").normalize()
rev.index=qend
yoy=rev/rev.shift(4)-1                              # year-over-year quarterly growth
accel=((yoy.diff()>0)&(yoy.diff().shift(1)>0)&(yoy>0))   # YoY rising 2 consecutive Qs & positive
highyoy=(yoy>=0.25)
avail=(rev.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()  # ~when 10-Q is public
def to_monthly(qf):
    qf=qf.copy(); qf.index=avail
    qf=qf[~qf.index.duplicated(keep="last")]
    return qf.reindex(me.index,method="ffill",limit=6)   # usable up to ~6 months until next filing
YOY=to_monthly(yoy); ACC=to_monthly(accel.astype(float)).fillna(0)>0.5; HY=to_monthly(highyoy.astype(float)).fillna(0)>0.5
p(f"revenue monthly panels built; tickers w/ rev data in price panel: {len([c for c in me.columns if c in YOY.columns])} t={time.time()-t0:.0f}s")
# ---- insider cluster / large-$ (trailing 3m) ----
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P["ym"]=pd.to_datetime(P.ym)
inames=[c for c in me.columns if c in set(P.tk)]; P=P[P.tk.isin(set(inames))]
def ipan(col): return P.pivot_table(index="ym",columns="tk",values=col,aggfunc="sum").reindex(index=me.index,columns=me.columns).fillna(0)
nb=ipan("nbuyers"); buy=ipan("buy")
nb3=nb.rolling(3,min_periods=1).sum(); buy3=buy.rolling(3,min_periods=1).sum()
bigthr=buy3.where(buy3>0).quantile(0.7,axis=1)
INS=(nb3>=2)|buy3.gt(bigthr,axis=0)                # insider cluster OR large-$ (validated edge)
# ---- technical timing proxy (monthly) ----
mom6=me/me.shift(6)-1; ma10=me.rolling(10,min_periods=10).mean()
TECH=(me>ma10)&(mom6>0)
# ---- returns ----
ret=(me/me.shift(1)-1).clip(-0.9,2.0); fwd12=(me.shift(-12)/me-1)
liq=(me.shift(1)>=3.0)
idx=me.index[(me.index>=pd.Timestamp("2012-07-01"))&(me.index<=pd.Timestamp("2025-12-31"))]
q=(me["QQQ"]/me["QQQ"].shift(1)-1).reindex(idx) if "QQQ" in me else None
def stats(r):
    r=r.dropna()
    if len(r)<12: return (np.nan,)*4
    c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d,len(r)
# restrict all masks to common tickers / align
cols=[c for c in me.columns if c in stocks]
def alg(df): return df.reindex(columns=me.columns).reindex(index=me.index)
ACC2=alg(ACC).fillna(False); HY2=alg(HY).fillna(False); INS2=alg(INS).fillna(False); TECH2=alg(TECH).fillna(False); LIQ=liq.fillna(False)
def port(mask,N=None):
    msk=(mask&LIQ)[cols]
    w=msk.shift(1).fillna(False).astype(float)
    if N:  # top-N by 6m momentum among qualifiers (concentrate)
        sc=mom6[cols].where(msk).rank(axis=1,ascending=False)
        w=(sc<=N).shift(1).fillna(False).astype(float)
    w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    r=(w*ret[cols]).sum(axis=1).reindex(idx)
    n=(w>0).sum(axis=1).reindex(idx)
    return r,n
QUAL={
 "rev-accel":ACC2,
 "rev-highYoY(>25%)":HY2,
 "insider-cluster/large$":INS2,
 "rev-accel & insider":ACC2&INS2,
 "rev-accel & tech-timing":ACC2&TECH2,
 "(accel|highYoY)&insider&tech":(ACC2|HY2)&INS2&TECH2,
 "rev-accel&tech  TOP20":ACC2&TECH2,
}
p(f"\n{'qualifier':32} {'avgN':>5} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'fwd12m':>7} {'2x':>5}")
if q is not None:
    c,s,d,_=stats(q); p(f"{'QQQ':32} {'-':>5} {c:>7.1%} {s:>7.2f} {d:>7.1%} {'-':>7} {'-':>5}")
for nm,mask in QUAL.items():
    N=20 if "TOP20" in nm else None
    r,n=port(mask,N)
    c,s,d,_=stats(r)
    # cross-sectional fwd12 + 2x of qualifiers
    qm=(mask&LIQ)[cols]
    f12=fwd12[cols].where(qm).reindex(idx)
    fmean=f12.stack().mean() if f12.notna().any().any() else np.nan
    two=(f12.stack()>1.0).mean() if f12.notna().any().any() else np.nan
    p(f"{nm:32} {n.mean():>5.0f} {c:>7.1%} {s:>7.2f} {d:>7.1%} {fmean:>7.1%} {two:>5.1%}")
# universe baseline fwd12
f12u=fwd12[cols].where(LIQ[cols]).reindex(idx)
p(f"\nuniverse fwd12m mean {f12u.stack().mean():.1%}  2x {(f12u.stack()>1.0).mean():.1%}")
p(f"DONE t={time.time()-t0:.0f}s")
