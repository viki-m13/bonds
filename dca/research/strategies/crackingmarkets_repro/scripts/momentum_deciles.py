import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); liq,me,cols=D["liq"],D["me"],D["cols"]
M=me.index
ret=(me/me.shift(1)-1).clip(-0.95,5.0)
# 12-1 momentum: past 12 months EXCLUDING most recent month, known at formation
mom=me.shift(1)/me.shift(12)-1
# tradeable PIT universe: liquid & lagged price >= $5 (no lookahead)
U=(liq&(me.shift(1)>=5.0)).fillna(False)
rk=mom.where(U).rank(axis=1,pct=True)
def decile(d):
    lo,hi=(d-1)/10,d/10
    w=((rk>lo)&(rk<=hi)).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w.shift(1)*ret).sum(axis=1)
idx=M[(M>=pd.Timestamp("1991-02-01"))&(M<=pd.Timestamp("2025-12-31"))]
P={d:decile(d).reindex(idx) for d in range(1,11)}
mkt=((U.shift(1).div(U.shift(1).sum(axis=1).replace(0,np.nan),axis=0))*ret).sum(axis=1).reindex(idx)  # EW universe
ls=(P[10]-P[1]).reindex(idx)          # long-short winners-minus-losers
def stats(r):
    r=r.dropna(); n=len(r); c=(1+r).prod()**(12/n)-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return c,s,dd
print(f"=== MOMENTUM DECILES (12-1, EW, monthly, PIT survivorship-clean, {idx[0].strftime('%Y')}-{idx[-1].strftime('%Y')}) ===")
print(f"{'portfolio':16}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}")
for d in range(1,11): c,s,dd=stats(P[d]); print(f"  D{d:<2} {'(losers)' if d==1 else '(winners)' if d==10 else '':9}{c*100:>7.1f}%{s:>8.2f}{dd*100:>7.0f}%")
print("  "+"-"*40)
for nm,r in [("D10 long-only",P[10]),("D10-D1 long/short",ls),("EW market",mkt)]:
    c,s,dd=stats(r); print(f"  {nm:14}{c*100:>7.1f}%{s:>8.2f}{dd*100:>7.0f}%")
# momentum-crash months in the L/S
print("\nWorst 6 months for D10-D1 (the momentum crashes):")
for dt,v in ls.nsmallest(6).items(): print(f"  {dt.strftime('%Y-%m')}: {v*100:+.0f}%")
# equity curves (log) like the article
fig,ax=plt.subplots(figsize=(11,6))
import matplotlib.cm as cm
for d in range(1,11):
    g=(1+P[d].fillna(0)).cumprod(); ax.plot(g.index,g,lw=1.3,color=cm.RdBu_r((d-1)/9),label=f"D{d}")
g=(1+ls.fillna(0)).cumprod(); ax.plot(g.index,g,lw=2.4,color="black",label="D10-D1 (L/S)")
gm=(1+mkt.fillna(0)).cumprod(); ax.plot(gm.index,gm,lw=1.6,color="gray",ls="--",label="EW market")
ax.set_yscale("log"); ax.grid(alpha=.3,which="both"); ax.legend(ncol=3,fontsize=8)
ax.set_title(f"Momentum-sorted deciles, 12-1, equal-weight, PIT survivorship-clean ({idx[0].strftime('%Y')}-{idx[-1].strftime('%Y')})")
fig.tight_layout(); fig.savefig("/home/user/momentum_deciles.png",dpi=120)
pd.to_pickle({"P":P,"ls":ls,"mkt":mkt,"mom":mom,"U":U,"idx":idx,"ret":ret},"/tmp/wave/_mom.pkl")
print("\nsaved chart + _mom.pkl")
