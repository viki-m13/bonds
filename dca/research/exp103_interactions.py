import numpy as np, pandas as pd, time, itertools, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def qidx(df):
    df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=qidx(rev)
def gq(k):
    d=F.get(k); return qidx(d).reindex(columns=rev.columns) if d is not None else None
OI,NI,AST,CASH,STI,SH=[gq(k) for k in ["OperatingIncomeLoss","NetIncomeLoss","Assets","CashAndCashEquivalentsAtCarryingValue","ShortTermInvestments","EntityCommonStockSharesOutstanding"]]
def qm(df,lim=6):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=lim)
ryoy=rev/rev.shift(4)-1; roic=(OI*4)/(AST-CASH.fillna(0)-STI.fillna(0)).clip(lower=1)
roic_m=qm(roic); ryoy_m=qm(ryoy); mcap=me*qm(SH); ps=mcap/qm(rev*4).clip(lower=1); ep=qm(NI*4)/mcap
revpersist=qm(((rev/rev.shift(4)-1).diff()>0).rolling(3).sum())  # rev-growth rising streak
mom12=me/me.shift(12)-1; mom6=me/me.shift(6)-1; vol6=FEAT["vol6"]; sharechg=FEAT["share_chg"]; lmcap=FEAT["log_mcap"]
liqf=(me.shift(1)>=3.0).fillna(False)
def rk(x): return x.where(liqf).rank(axis=1,pct=True)
# binary signal library
SIG={
 "decel_growth": (ryoy_m>0)&(ryoy_m<ryoy_m.shift(4)),
 "stable_hi_roic": (roic_m.rolling(6,min_periods=4).min()>0.05)&(rk(roic_m)>0.6),
 "mom_decay": (mom12.shift(6)>0.10)&(mom6<0.05),
 "not_rerated": ps<=ps.rolling(18,min_periods=9).median()*1.05,
 "hi_value": rk(ep)>0.66,
 "low_vol": rk(vol6)<0.33,
 "buyback": rk(-sharechg)>0.66,
 "small_cap": rk(lmcap)<0.5,
 "hi_quality": rk(roic_m)>0.66,
 "rev_persist": revpersist>=2,
 "mom_pos": mom12>0,
}
SIG={k:v.fillna(False)&liqf for k,v in SIG.items()}
idx=M[(M>=pd.Timestamp("2012-01-01"))&(M<=pd.Timestamp("2024-06-30"))]
fwd12=(me.shift(-12)/me-1).clip(-0.95,5.0)
uni=fwd12.where(liqf).loc[idx].stack(); UNI=uni.mean()
p(f"universe fwd12 mean {UNI:+.1%}  (signals={len(SIG)})")
def ealpha(mask):
    v=fwd12.where(mask).loc[idx].stack()
    n=mask.loc[idx].sum(axis=1).mean()
    if len(v)<200 or n<6: return None
    return v.mean(),v.median(),(v>0).mean(),n
res=[]
# singles
for k,m in SIG.items():
    r=ealpha(m);
    if r: res.append((k,)+r+(1,))
# 2-way and 3-way conjunctions
keys=list(SIG)
for combo in list(itertools.combinations(keys,2))+list(itertools.combinations(keys,3)):
    m=SIG[combo[0]].copy()
    for c in combo[1:]: m=m&SIG[c]
    r=ealpha(m)
    if r: res.append(("+".join(combo),)+r+(len(combo),))
ntrials=len(res)
res.sort(key=lambda x:-x[1])
p(f"\ntotal conjunctions tested: {ntrials}  (deflate accordingly)")
p(f"\nTOP 18 by entry-alpha (fwd12 mean):")
p(f"{'conjunction':52}{'mean':>7}{'med':>7}{'hit':>6}{'N/mo':>6}{'order':>6}")
for nm,mn,md,hit,n,o in res[:18]:
    p(f"{nm[:50]:52}{mn:>+7.1%}{md:>+7.1%}{hit:>6.0%}{n:>6.0f}{o:>6d}")
p(f"\nfor reference: handoff 4-way was +14.4% / hit 64% / 22 names; universe {UNI:+.1%}")
# best deflation-aware: require N/mo>=10 and hit>=0.60
robust=[r for r in res if r[4]>=10 and r[3]>=0.60 and r[5]>=2]
p(f"\nROBUST (N>=10, hit>=60%, multi-signal) top 8:")
for nm,mn,md,hit,n,o in robust[:8]:
    p(f"  {nm[:50]:52}{mn:>+7.1%} hit {hit:.0%} N {n:.0f}")
p(f"DONE t={time.time()-t0:.0f}s")
