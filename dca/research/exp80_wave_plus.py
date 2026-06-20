import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean()
mom1=me.pct_change(); mom3=me/me.shift(3)-1
vol6=FEAT["vol6"]
# PEAD / earnings-drift: fresh fundamental catalyst + price reacting up
highyoy=(FEAT["rev_yoy"]>=0.25); accel=(FEAT["rev_accel"]>0.5)
pead=((accel|highyoy)&(mom1>0)).astype(float)
# market regime: QQQ vs its 10-mo MA
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qma=qpx.rolling(10,min_periods=10).mean()
regime_on=(qpx>qma)   # True = risk-on
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=qpx.pct_change().reindex(idx)
elig=(liq&(me>=3.0)&(me>ma10))
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(N=12, runnergate=True, peadboost=0.0, volsize=False, regime=None, trail=-0.30):
    sc=PROB.copy()
    if peadboost>0: sc=sc.add(peadboost*pead, fill_value=0)
    sc=sc.where(elig & ((mom3>0) if runnergate else True))
    rank=sc.rank(axis=1,ascending=False)
    pos={}; cash=1.0; out=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx)
            if cpx/e["peak"]-1<=trail or cpx<ma10.loc[dt].get(tk,np.nan): cash+=e["val"]; pos.pop(tk)
        # regime throttle: target invested fraction of equity
        eq=cash+sum(e["val"] for e in pos.values())
        frac=1.0
        if regime is not None and dt in regime_on.index and not bool(regime_on.get(dt,True)): frac=regime
        # entries: refill to N among top runners; size equal or inverse-vol
        if dt in PROB.index:
            rk=rank.loc[dt]; cands=[t for t in rk[rk<=N*3].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan))]
            need=N-len(pos); target_invested=frac*eq; cur_invested=sum(e["val"] for e in pos.values())
            budget=max(0.0, target_invested-cur_invested); budget=min(budget,cash)
            if need>0 and budget>1e-9 and cands:
                pick=cands[:need]
                if volsize:
                    iv=np.array([1.0/max(vol6.loc[dt].get(t,0.1),0.02) for t in pick]); wv=iv/iv.sum()
                else: wv=np.ones(len(pick))/len(pick)
                for tk,wi in zip(pick,wv):
                    a=budget*wi; pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":a}; cash-=a
        # de-risk if over target (regime turned off): trim to cash
        eq=cash+sum(e["val"] for e in pos.values())
        if regime is not None and frac<1.0:
            over=sum(e["val"] for e in pos.values())-frac*eq
            if over>0:
                tot=sum(e["val"] for e in pos.values())
                for e in pos.values(): cut=e["val"]*(over/tot); e["val"]-=cut; cash+=cut
        eq0=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx):
            for tk in pos:
                r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx): out.append((didx[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)
p(f"{'WAVE long-only variant (2015-2025)':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
c,s,d=stats(qret); p(f"{'QQQ':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
cfgs={
 "base: ML N12 runner-gate":dict(),
 "+ PEAD boost":dict(peadboost=0.15),
 "+ vol-scaled sizing":dict(volsize=True),
 "+ regime throttle to 50%":dict(regime=0.5),
 "+ regime throttle to 0%":dict(regime=0.0),
 "ALL: PEAD+volsize+regime50":dict(peadboost=0.15,volsize=True,regime=0.5),
 "ALL + regime0":dict(peadboost=0.15,volsize=True,regime=0.0),
}
res={}
for nm,cfg in cfgs.items():
    r=sim(**cfg); res[nm]=r; c,s,d=stats(r); p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
best=max(res,key=lambda k: stats(res[k])[1]); br=res[best]
p(f"\nbest Sharpe: {best}")
p("Sub-period (best vs QQQ):")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(br[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
fig,ax=plt.subplots(figsize=(11,6))
g=(1+br).cumprod(); gq=(1+qret).cumprod(); c,s,d=stats(br); qc,qs,qd=stats(qret)
ax.plot(g.index,g,label=f"WAVE+ ({best}) CAGR {c:.0%}/Sh {s:.2f}/DD {d:.0%}",lw=2.4,color="#1f77b4")
ax.plot(gq.index,gq,label=f"QQQ CAGR {qc:.0%}/Sh {qs:.2f}/DD {qd:.0%}",lw=2,color="#888")
ax.set_yscale("log"); ax.set_title("WAVE+ : ML runner picker + PEAD + vol-sizing + regime throttle (long-only)")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/tmp/wave/wave_plus.png",dpi=110)
p(f"\nsaved /tmp/wave/wave_plus.png DONE t={time.time()-t0:.0f}s")
