import numpy as np, pandas as pd, time, warnings
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
OI,NI,AST,CASH,STI,GP=[gq(k) for k in ["OperatingIncomeLoss","NetIncomeLoss","Assets","CashAndCashEquivalentsAtCarryingValue","ShortTermInvestments","GrossProfit"]]
# base quarterly fundamentals (TTM-ish via *4 on quarterly flow)
roic=(OI*4)/(AST-CASH.fillna(0)-STI.fillna(0)).clip(lower=1)
opm=(OI*4)/rev.clip(lower=1); gm=(GP*4)/rev.clip(lower=1); ato=(rev*4)/AST.clip(lower=1); ryoy=rev/rev.shift(4)-1
BASE={"roic":roic,"opmargin":opm,"grossmargin":gm,"assetturn":ato,"revgrowth":ryoy}
# --- trajectory transforms (per-firm time series) ---
w=6; ii=np.arange(w); k=(ii-ii.mean()); k=k/ (k**2).sum()   # OLS slope weights
def slope(df):  # rolling 6q linear slope
    return df.rolling(w).apply(lambda y: np.dot(y,k) if np.isfinite(y).all() else np.nan,raw=True)
def persist(df):  # consecutive quarters strictly rising
    up=(df.diff()>0).astype(float);
    g=up*(up.groupby((up==0).cumsum(),axis=0).cumcount()+1) if False else None
    # consecutive-rising count via loop over quarters (cheap: ~60 rows)
    out=pd.DataFrame(0.0,index=df.index,columns=df.columns); prev=np.zeros(df.shape[1])
    d=(df.diff()>0).values
    for r in range(df.shape[0]):
        prev=np.where(d[r],prev+1,0.0); out.iloc[r]=prev
    return out
TRAJ={}
for nm,b in BASE.items():
    TRAJ[nm+"_level"]=b
    TRAJ[nm+"_slope"]=slope(b)
    TRAJ[nm+"_accel"]=slope(b)-slope(b).shift(2)              # change in slope (2nd deriv)
    TRAJ[nm+"_persist"]=persist(b)
p(f"trajectory features: {len(TRAJ)} t={time.time()-t0:.0f}s")
def qm(df):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=6)
TRAJm={nm:qm(v) for nm,v in TRAJ.items()}
liqf=(me.shift(1)>=3.0).fillna(False)
fwd3=(me.shift(-3)/me-1).clip(-0.9,3.0); fwd12=(me.shift(-12)/me-1).clip(-0.95,5.0)
ret=(me/me.shift(1)-1).clip(-0.9,2.0)
idx=M[(M>=pd.Timestamp("2012-01-01"))&(M<=pd.Timestamp("2024-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def ic(sig,fwd):
    fr=sig.where(liqf); v=[]
    for dt in idx:
        d=pd.concat([fr.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: v.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(v)
def lsh(sig,q=0.2):
    fr=sig.where(liqf); rk=fr.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    sw=(rk<=q).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    s=((lw.shift(1)*ret).sum(axis=1)-(sw.shift(1)*ret).sum(axis=1)).reindex(idx)
    return s.mean()/s.std()*np.sqrt(12) if s.std()>0 else np.nan
N_TRIALS=len(TRAJm)
p(f"\n{'trajectory feature':22}{'IC3m':>7}{'IC12m':>7}{'L/S Sh':>7}")
rows=[]
for nm,sig in TRAJm.items():
    i3=ic(sig,fwd3); i12=ic(sig,fwd12); sh=lsh(sig); rows.append((nm,i3,i12,sh))
for nm,i3,i12,sh in sorted(rows,key=lambda x:-abs(x[1])):
    p(f"{nm:22}{i3:>+7.3f}{i12:>+7.3f}{sh:>+7.2f}")
# deflation note: best-of-N Sharpe expected under null ~ sqrt(2 ln N)/sqrt(T_years) ... report context
best=max(rows,key=lambda x:abs(x[1]))
p(f"\nbest |IC3m|: {best[0]} {best[1]:+.3f} | trials={N_TRIALS} (deflate: with {N_TRIALS} trials, |IC|<~0.03 is noise)")
# entry-alpha: top-decile of the best trajectory signal vs universe
sig=TRAJm[best[0]]; rk=sig.where(liqf).rank(axis=1,pct=True)
top=fwd12.where(rk>=0.9).loc[idx].stack(); uni=fwd12.where(liqf).loc[idx].stack()
p(f"\nbest-traj top-decile fwd12 mean {top.mean():+.1%} (hit {(top>0).mean():.0%}) vs universe {uni.mean():+.1%}")
pd.to_pickle(TRAJm,"/tmp/wave/_traj.pkl")
p(f"DONE t={time.time()-t0:.0f}s")
