import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index
r=(me/me.shift(1)-1)                       # monthly returns
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; mkt=qpx.pct_change().reindex(M)
LIQ=(me.shift(1)>=3.0).fillna(False)
# ---- residual (idiosyncratic) momentum: strip market beta, momentum of residuals ----
win=36
cov=r.rolling(win,min_periods=18).cov(mkt)         # cov(r_i, mkt)
var=mkt.rolling(win,min_periods=18).var()
beta=cov.div(var,axis=0)
resid=r.sub(beta.mul(mkt,axis=0))
resmom=resid.rolling(12,min_periods=8).mean().shift(1)        # 12m residual momentum (skip latest)
residvol=resid.rolling(12,min_periods=8).std()
resmom_sharpe=(resid.rolling(12,min_periods=8).mean()/residvol).shift(1)   # risk-adjusted residual mom
# ---- raw momentum for comparison ----
rawmom=(me.shift(1)/me.shift(13)-1)
# ---- seasonality (Heston-Sadka): expanding mean of same-calendar-month past returns ----
seas=pd.DataFrame(index=M,columns=cols,dtype=float)
moy=M.month
for mm in range(1,13):
    rows=np.where(moy==mm)[0]
    sub=r.iloc[rows]
    exp=sub.expanding(min_periods=2).mean().shift(1)    # past same-month avg, no look-ahead
    seas.iloc[rows]=exp.values
# ---- short-term reversal ----
strev=-(me.shift(1)/me.shift(2)-1)
# ---- fundamental acceleration (2nd derivative of YoY revenue) ----
fund_accel=FEAT["rev_yoy"].diff()           # change in YoY growth (accel)
# ---- evaluate ----
fwd1=r.shift(-1).clip(-0.9,3.0); fwd3=(me.shift(-3)/me-1).clip(-0.9,3.0)
idx=M[(M>=pd.Timestamp("2005-01-01"))&(M<=pd.Timestamp("2025-09-30"))]
def ic(f,fwd):
    fr=f.where(LIQ); ics=[]
    for dt in idx:
        a=fr.loc[dt]; b=fwd.loc[dt]; d=pd.concat([a,b],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: ics.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(ics) if ics else np.nan
def ls(f,fwd=fwd1,q=0.2):
    fr=f.where(LIQ); rk=fr.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    sw=(rk<=q).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    s=((lw*fwd).sum(axis=1)-(sw*fwd).sum(axis=1)).reindex(idx)
    return s.mean()/s.std()*np.sqrt(12) if s.std()>0 else np.nan, s.corr(mkt.reindex(idx))
p(f"{'NEW factor':22} {'IC1m':>7} {'IC3m':>7} {'L/S Sharpe':>11} {'corrMkt':>8}")
for nm,f in [("residual-mom(12m)",resmom),("residual-mom-sharpe",resmom_sharpe),("raw-mom(12-1)",rawmom),
             ("seasonality",seas),("short-term-reversal",strev),("fundamental-accel",fund_accel)]:
    sh,cm=ls(f); p(f"{nm:22} {ic(f,fwd1):>7.3f} {ic(f,fwd3):>7.3f} {sh:>11.2f} {cm:>8.2f}")
p(f"\nwindow {idx[0].date()}..{idx[-1].date()} ({len(idx)} mo) t={time.time()-t0:.0f}s")
# detailed: residual-mom L/S by era + decile monotonicity
p(f"\nResidual-momentum L/S Sharpe by era:")
for lo,hi in [("2005","2009"),("2010","2014"),("2015","2019"),("2020","2025")]:
    sub=idx[(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31"))]
    fr=resmom.where(LIQ); rk=fr.rank(axis=1,pct=True)
    lw=(rk>=0.8).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    sw=(rk<=0.2).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    s=((lw*fwd1).sum(axis=1)-(sw*fwd1).sum(axis=1)).reindex(sub)
    p(f"  {lo}-{hi}: {s.mean()/s.std()*np.sqrt(12):.2f}")
# decile spread monotonicity (residual mom)
fr=resmom.where(LIQ); dec=fr.rank(axis=1,pct=True)
p(f"\nResidual-mom decile fwd1m mean (D1 low..D10 high):")
means=[]
for d10 in range(10):
    sel=((dec>d10/10)&(dec<=(d10+1)/10))
    rr=fwd1.where(sel).reindex(idx).stack().mean(); means.append(rr)
p("  "+" ".join(f"{m*100:+.1f}" for m in means))
p(f"DONE t={time.time()-t0:.0f}s")
