import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
D=pd.read_pickle("/tmp/wave/_mom.pkl"); P,ls,mkt,mom,U,idx,ret=D["P"],D["ls"],D["mkt"],D["mom"],D["U"],D["idx"],D["ret"]
F=pd.read_pickle("/tmp/wave/_featmat.pkl"); me=F["me"]; M=me.index
def stats(r):
    r=r.dropna(); n=len(r); c=(1+r).prod()**(12/n)-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return c,s,dd
def row(nm,r): c,s,dd=stats(r); print(f"  {nm:34}{c*100:>7.1f}%{s:>8.2f}{dd*100:>7.0f}%")
print(f"{'strategy':36}{'CAGR':>8}{'Sharpe':>7}{'maxDD':>8}")
row("D10-D1 L/S (baseline)",ls); row("D10 long-only (baseline)",P[10])
print("  "+"-"*48)
# (1) BARROSO vol-scaling: target 12% ann vol via trailing-6m realized vol of the L/S
def volscale(r,tgt=0.12,win=6,cap=3.0):
    rv=r.rolling(win,min_periods=3).std()*np.sqrt(12); lev=(tgt/rv).clip(upper=cap).shift(1).fillna(1.0); return r*lev
row("L/S + Barroso vol-scale (12%)",volscale(ls))
row("D10 long-only + vol-scale (15%)",volscale(P[10],tgt=0.15))
print("  "+"-"*48)
# (2) BETA-NEUTRALIZE the L/S: hedge its time-varying market beta (the crash is loser-leg beta)
mk=mkt.reindex(idx)
beta=(ls.rolling(24,min_periods=12).cov(mk)/mk.rolling(24,min_periods=12).var())
ls_bn=(ls-beta.shift(1)*mk).reindex(idx)
row("L/S beta-hedged",ls_bn); row("L/S beta-hedged + vol-scale",volscale(ls_bn))
print("  "+"-"*48)
# (3) RESIDUAL (idiosyncratic) momentum: sort on beta-adjusted 12-1 return
retf=(me/me.shift(1)-1).clip(-0.95,5.0)
mkt_full=((U.shift(1).div(U.shift(1).sum(axis=1).replace(0,np.nan),axis=0))*retf).sum(axis=1)  # EW mkt over full M
q=mkt_full; qv=q.rolling(36,min_periods=18).var()
cov=retf.mul(q,axis=0).rolling(36,min_periods=18).mean().sub(retf.rolling(36,min_periods=18).mean().mul(q.rolling(36,min_periods=18).mean(),axis=0),axis=0)
B=cov.div(qv,axis=0).clip(-3,3)
resid=retf.sub(B.mul(q,axis=0))                       # idiosyncratic return
rmom=resid.rolling(11,min_periods=8).sum().shift(1)   # residual 12-1
rk=rmom.where(U).rank(axis=1,pct=True)
def dec(rkk,d):
    w=((rkk>(d-1)/10)&(rkk<=d/10)).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w.shift(1)*retf).sum(axis=1).reindex(idx)
rls=(dec(rk,10)-dec(rk,1)); 
row("Residual-momentum L/S",rls); row("Residual-mom L/S + vol-scale",volscale(rls))
row("Residual-mom D10 long-only",dec(rk,10))
print("  "+"-"*48)
# (4) the works: residual-mom, beta-hedged, vol-scaled
betar=(rls.rolling(24,min_periods=12).cov(mk)/mk.rolling(24,min_periods=12).var())
rls_bn=(rls-betar.shift(1)*mk)
row("Residual L/S + beta-hedge + vol-scale",volscale(rls_bn))
print("\nWorst month, baseline vs improved:")
print(f"  baseline L/S:               {ls.min()*100:+.0f}%")
print(f"  +vol-scale:                 {volscale(ls).min()*100:+.0f}%")
print(f"  residual+beta-hedge+volscale:{volscale(rls_bn).min()*100:+.0f}%")
