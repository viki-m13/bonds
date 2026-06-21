import time; t0=time.time()
from costlib import *
C=pd.read_pickle("/tmp/wave/_composite.pkl"); comp=C["comp"].reindex(index=M,columns=cols)
hdr()
Wl,Ws=build(PROB); base=net(Wl,Ws); row("v3 baseline (ML)",base)
print("-"*92)
# --- residual (idiosyncratic) momentum ---
resid=ret.sub(BETA.mul(qret.reindex(M),axis=0))               # ret - beta*mkt
rm=resid.rolling(11,min_periods=8).sum().shift(1)             # t-12..t-2 residual cum
Wl,Ws=build(rm); row("residual-momentum alone",net(Wl,Ws))
ens_rm=(PROB.where(LIQ).rank(axis=1,pct=True)+rm.where(LIQ).rank(axis=1,pct=True))/2
Wl,Ws=build(ens_rm); row("ML + residual-mom (50/50)",net(Wl,Ws))
ens_rm7=(0.7*PROB.where(LIQ).rank(axis=1,pct=True)+0.3*rm.where(LIQ).rank(axis=1,pct=True))
Wl,Ws=build(ens_rm7); row("ML + residual-mom (70/30)",net(Wl,Ws))
print("-"*92)
# --- size neutralization (like beta, but on size factor) ---
szN=sec_neutralize(PROB,lmrank)
Wl,Ws=build(szN); row("size-neutral ML",net(Wl,Ws))
print("-"*92)
# --- beta-neutral linear ensemble (re-test under beta-neutral, not dollar-neutral) ---
ens_lin=(PROB.where(LIQ).rank(axis=1,pct=True)+comp.where(LIQ).rank(axis=1,pct=True))/2
Wl,Ws=build(ens_lin); row("ML + linear factor (50/50)",net(Wl,Ws))
ens_lin8=(0.8*PROB.where(LIQ).rank(axis=1,pct=True)+0.2*comp.where(LIQ).rank(axis=1,pct=True))
Wl,Ws=build(ens_lin8); row("ML + linear (80/20)",net(Wl,Ws))
print("-"*92)
# --- dispersion / regime gross-scaling ---
Wl,Ws=build(PROB)
disp=PROB.where(LIQ).std(axis=1)                              # cross-sectional signal dispersion
gs=(disp/disp.rolling(24,min_periods=12).median()).clip(0.5,1.5).reindex(idx).fillna(1.0)
row("v3 + dispersion gross-scale",net(Wl,Ws,gross_scale=gs))
# vol-regime: cut gross after the book's own vol spikes (realized)
bv=base.rolling(3,min_periods=2).std()*np.sqrt(12)
gs2=(0.16/bv).clip(0.5,1.2).shift(1).reindex(idx).fillna(1.0)
row("v3 + own-vol gross-scale",net(Wl,Ws,gross_scale=gs2))
print(f"\nDONE t={time.time()-t0:.0f}s")
