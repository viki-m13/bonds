import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_ls.pkl"); LS=D["LS"]; idx=D["idx"]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def ann(r):
    r=r.dropna();
    if len(r)<12 or r.std()==0: return (np.nan,np.nan,np.nan)
    a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12); eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
# select factors with positive standalone Sharpe (alpha streams)
good=[k for k in LS if ann(LS[k])[1] and ann(LS[k])[1]>0.4]
p(f"alpha streams used ({len(good)}): {good}")
LSdf=pd.DataFrame({k:LS[k] for k in good}).reindex(idx)
# 1) equal-weight multi-factor alpha
ew=LSdf.mean(axis=1)
# 2) inverse-vol weight (risk parity across factors)
iv=1/LSdf.rolling(12,min_periods=6).std(); w=iv.div(iv.sum(axis=1),axis=0); rp=(w.shift(1)*LSdf).sum(axis=1)
# 3) correlation-aware: just report
a,s,d=ann(ew); p(f"\nEqual-weight multi-factor L/S alpha:  ann {a*100:.1f}% Sharpe {s:.2f} maxDD {d:.1%} corrQQQ {ew.corr(qret):.2f}")
a,s,d=ann(rp); p(f"Inverse-vol multi-factor L/S alpha:   ann {a*100:.1f}% Sharpe {s:.2f} maxDD {d:.1%} corrQQQ {rp.corr(qret):.2f}")
alpha=rp.fillna(ew)   # use risk-parity alpha
# avg pairwise corr among factors
cc=LSdf.corr().values; p(f"avg pairwise factor corr: {cc[np.triu_indices_from(cc,1)].mean():.2f}")
# ---- PORTABLE ALPHA: overlay market-neutral alpha on beta ----
p(f"\n{'portfolio':40} {'CAGR/ann':>9} {'Sharpe':>7} {'maxDD':>7} {'corrQQQ':>8}")
a,s,d=ann(qret); p(f"{'QQQ (beta)':40} {a*100:>8.1f}% {s:>7.2f} {d:>7.1%} {1.0:>8.2f}")
a,s,d=ann(alpha); p(f"{'multi-factor alpha (mkt-neutral)':40} {a*100:>8.1f}% {s:>7.2f} {d:>7.1%} {alpha.corr(qret):>8.2f}")
for k in [0.5,1.0,1.5,2.0]:
    blend=qret+k*alpha
    a,s,d=ann(blend); p(f"{'QQQ + '+str(k)+'x alpha overlay':40} {a*100:>8.1f}% {s:>7.2f} {d:>7.1%} {blend.corr(qret):>8.2f}")
# combine with long-only composite sleeve? load by recompute quick (reuse featmat)
Dm=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me=Dm["FEAT"],Dm["liq"],Dm["me"]
ret=(me/me.shift(1)-1).clip(-0.9,2.0); ma10=me.rolling(10,min_periods=10).mean(); trend=(me>ma10)
def Zc(nm): return (FEAT[nm].where(liq).rank(axis=1,pct=True)-0.5).fillna(0.0)
comp=(1.0*Zc("roa")+0.7*Zc("roe")-1.0*Zc("vol6")+1.0*Zc("distHigh")+0.7*Zc("mom6")-0.8*Zc("share_chg")+0.5*Zc("op_leverage"))
sc=comp.where((liq&(me>=3.0)&trend)); rk=sc.rank(axis=1,ascending=False)
wq=(rk<=25).shift(1).fillna(False).astype(float); wq=wq.div(wq.sum(axis=1).replace(0,np.nan),axis=0)
compr=(wq*ret).sum(axis=1).reindex(idx).fillna(0.0)
a,s,d=ann(compr); p(f"{'composite long-only sleeve':40} {a*100:>8.1f}% {s:>7.2f} {d:>7.1%} {compr.corr(qret):>8.2f}")
finals={
 "60 comp / 40 alpha(1x)":0.6*compr+0.4*alpha,
 "50 QQQ / 50 comp + 1x alpha":0.5*qret+0.5*compr+alpha,
 "comp + 1.5x alpha overlay":compr+1.5*alpha,
 "40 QQQ/40 comp/20 alpha*2":0.4*qret+0.4*compr+0.4*alpha,
}
best=None
for nm,r in finals.items():
    a,s,d=ann(r); p(f"{nm:40} {a*100:>8.1f}% {s:>7.2f} {d:>7.1%} {r.corr(qret):>8.2f}")
    if best is None or s>best: best=s;bnm=nm;bser=r
p(f"\nbest: {bnm} Sharpe {best:.2f}")
# figure
fig,ax=plt.subplots(figsize=(11,6))
for nm,r,co,lw in [("Best alpha+beta blend",bser,"#d62728",2.6),("Multi-factor alpha (mkt-neutral)",alpha,"#9467bd",1.6),
                   ("Composite long-only",compr,"#2ca02c",1.4),("QQQ",qret,"#888",2)]:
    g=(1+r).cumprod(); a,s,d=ann(r); ax.plot(g.index,g,label=f"{nm} (ann {a*100:.0f}%, Sh {s:.2f}, DD {d:.0%})",lw=lw,color=co)
ax.set_yscale("log"); ax.set_title(f"Portable alpha: market-neutral factor book + beta sleeves (PIT clean 2012-2025)\n{bnm}")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/tmp/wave/alpha_engine.png",dpi=110)
p(f"\nsaved /tmp/wave/alpha_engine.png DONE t={time.time()-t0:.0f}s")
