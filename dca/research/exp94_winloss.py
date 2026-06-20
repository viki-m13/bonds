import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; fnames=list(FEAT.keys())
LIQ=(me.shift(1)>=3.0).fillna(False)
fwd6=(me.shift(-6)/me-1); fwd12=(me.shift(-12)/me-1)
idx=M[(M>=pd.Timestamp("2011-06-01"))&(M<=pd.Timestamp("2024-12-31"))]
# winners = top decile fwd6, losers = bottom decile; profile every feature (cross-sec pctile)
Z={nm:FEAT[nm].where(LIQ).rank(axis=1,pct=True) for nm in fnames}
def profile(fwd):
    res={}
    for nm in fnames:
        wr,lr=[],[]
        zr=Z[nm]
        for dt in idx:
            f=fwd.where(LIQ).loc[dt].dropna()
            if len(f)<60: continue
            hi=f>=f.quantile(0.9); lo=f<=f.quantile(0.1)
            z=zr.loc[dt]
            wr.append(z[hi.index[hi]].mean()); lr.append(z[lo.index[lo]].mean())
        wr=np.array(wr); lr=np.array(lr); d=wr-lr
        res[nm]=(np.nanmean(wr),np.nanmean(lr),np.nanmean(d),np.nanmean(d)/(np.nanstd(d)+1e-9)*np.sqrt(len(d)))
    return res
p("=== BANGER (fwd-6m top decile) vs LOSER (bottom decile) — feature percentile profile ===")
p(f"{'feature':16} {'winner':>7} {'loser':>7} {'spread':>7} {'t-stat':>7}")
r6=profile(fwd6)
for nm,(w,l,d,t) in sorted(r6.items(),key=lambda x:-abs(x[1][2])):
    p(f"{nm:16} {w:>7.2f} {l:>7.2f} {d:>+7.2f} {t:>+7.1f}")
p(f"\nt={time.time()-t0:.0f}s")
# same for fwd-12m (longer bangers)
p("\n=== fwd-12m top vs bottom decile (longer-horizon bangers) — top discriminators ===")
r12=profile(fwd12)
for nm,(w,l,d,t) in sorted(r12.items(),key=lambda x:-abs(x[1][2]))[:12]:
    p(f"{nm:16} {w:>7.2f} {l:>7.2f} {d:>+7.2f} {t:>+7.1f}")
# which features are NOISE (low |spread| both horizons) -> candidates to drop
p("\nLIKELY-NOISE features (|spread|<0.03 on fwd6):")
noise=[nm for nm,(w,l,d,t) in r6.items() if abs(d)<0.03]
p("  "+", ".join(noise) if noise else "  (none)")
strong=[nm for nm,(w,l,d,t) in r6.items() if abs(d)>=0.06]
p(f"\nSTRONG discriminators (|spread|>=0.06): {strong}")
pd.to_pickle({"r6":r6,"r12":r12,"strong":strong,"noise":noise},"/tmp/wave/_winloss.pkl")
p(f"DONE t={time.time()-t0:.0f}s")
