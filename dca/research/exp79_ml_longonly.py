import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)   # ML score (2015-2025)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean()
mom3=me/me.shift(3)-1; mom6=FEAT["mom6"]
def Z(nm): return (FEAT[nm].where(liq).rank(axis=1,pct=True)-0.5).fillna(0.0)
comp=(1.0*Z("roa")+0.7*Z("roe")-1.0*Z("vol6")+1.0*Z("distHigh")+0.7*Z("mom6")-0.8*Z("share_chg")+0.5*Z("op_leverage"))
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
elig=(liq&(me>=3.0)&(me>ma10))
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(score, N=12, ride=True, trail=-0.30, trend_exit=True, cull=None, runnergate=False):
    sc=score.where(elig & ((mom3>0) if runnergate else True))
    rank=sc.rank(axis=1,ascending=False)
    pos={}; cash=1.0; out=[]
    for k,dt in enumerate(didx):
        if dt<idx[0]-pd.Timedelta(days=400):
            pass
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1; age=k-e["i"]; ex=False
            if trail is not None and cpx/e["peak"]-1<=trail: ex=True
            if trend_exit and cpx<ma10.loc[dt].get(tk,np.nan): ex=True
            if cull:
                for mm,thr in cull:
                    if age==mm and rs<thr: ex=True
            if ex: cash+=e["val"]; pos.pop(tk)
        if not ride and pos:
            tot=cash+sum(e["val"] for e in pos.values()); tgt=tot/max(N,len(pos))
            for e in pos.values(): e["val"]=tgt
            cash=tot-tgt*len(pos)
        if dt in PROB.index:
            rk=rank.loc[dt]; cands=[t for t in rk[rk<=N*3].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan))]
            need=N-len(pos)
            if need>0 and cash>1e-9 and cands:
                sl=cash/need
                for tk in cands[:need]: pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":sl}; cash-=sl
        eq0=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx):
            for tk in pos:
                r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx): out.append((didx[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)
p(f"{'long-only (2015-2025)':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
c,s,d=stats(qret); p(f"{'QQQ':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
res={}
cfgs={
 "ML N12 ride + trail/trend":dict(score=PROB,N=12),
 "ML N8 ride + trail/trend":dict(score=PROB,N=8),
 "ML N20 ride + trail/trend":dict(score=PROB,N=20),
 "ML N12 + runner-gate(mom3>0)":dict(score=PROB,N=12,runnergate=True),
 "ML N12 equal-rebal":dict(score=PROB,N=12,ride=False),
 "composite N12 ride":dict(score=comp,N=12),
}
for nm,cfg in cfgs.items():
    r=sim(**cfg); res[nm]=r; c,s,d=stats(r); p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
# blend best ML sleeve + composite + QQQ
ml=res["ML N12 ride + trail/trend"]; cp=res["composite N12 ride"]
p(f"\ncorr(ML,comp)={ml.corr(cp):.2f} corr(ML,QQQ)={ml.corr(qret):.2f}")
for nm,r in {"50 ML/50 comp":0.5*ml+0.5*cp,"40 ML/40 comp/20 QQQ":0.4*ml+0.4*cp+0.2*qret,"50 ML/30comp/20QQQ":0.5*ml+0.3*cp+0.2*qret}.items():
    c,s,d=stats(r); p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
best=0.4*ml+0.4*cp+0.2*qret
p("\nSub-period (40ML/40comp/20QQQ vs QQQ):")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(best[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
fig,ax=plt.subplots(figsize=(11,6))
g=(1+best).cumprod(); gq=(1+qret).cumprod(); c,s,d=stats(best); qc,qs,qd=stats(qret)
ax.plot(g.index,g,label=f"Long-only ML+composite blend CAGR {c:.0%}/Sh {s:.2f}/DD {d:.0%}",lw=2.4,color="#1f77b4")
ax.plot(gq.index,gq,label=f"QQQ CAGR {qc:.0%}/Sh {qs:.2f}/DD {qd:.0%}",lw=2,color="#888")
ax.set_yscale("log"); ax.set_title("Long-only ML stock-picker + composite (no margin/short), 2015-2025 PIT clean")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/tmp/wave/ml_longonly.png",dpi=110)
p(f"\nsaved /tmp/wave/ml_longonly.png DONE t={time.time()-t0:.0f}s")
