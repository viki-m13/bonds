import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index
def Z(nm):
    f=FEAT[nm].where(liq); r=f.rank(axis=1,pct=True)-0.5; return r.fillna(0.0)
# discovered strong-factor composite (quality + low-vol + strength + buybacks + momentum)
comp = (1.0*Z("roa") + 0.7*Z("roe") - 1.0*Z("vol6") + 1.0*Z("distHigh")
        + 0.7*Z("mom6") - 0.8*Z("share_chg") + 0.5*Z("op_leverage"))
# insider / rev booster (does the original edge still add on top?)
_bigthr=FEAT["ins_buy$"].where(FEAT["ins_buy$"]>0).quantile(0.7,axis=1)
insb = ((FEAT["ins_clustern"]>=2)|(FEAT["ins_buy$"].gt(_bigthr,axis=0))).astype(float)
comp_ins = comp + 0.5*(insb-0.5) + 0.5*(FEAT["rev_accel"]-0.5)
ret=(me/me.shift(1)-1).clip(-0.9,2.0)
ma10=me.rolling(10,min_periods=10).mean(); trend=(me>ma10)
mom6=FEAT["mom6"]
idx=M[(M>=pd.Timestamp("2012-07-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
didx=list(M)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(score,N=25,stop=-0.25,trail=-0.35,ladder=[(3,0.10),(6,0.30)],trend_exit=True,maxhold=36,gate_trend=True):
    elig=(liq&(trend if gate_trend else True))
    sc=score.where(elig)
    rank=sc.rank(axis=1,ascending=False)
    pos={}; monthly=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1; age=k-e["i"]; ex=False
            if stop is not None and rs<=stop: ex=True
            if trail is not None and cpx/e["peak"]-1<=trail: ex=True
            if ladder:
                for mm,thr in ladder:
                    if age==mm and rs<thr: ex=True
            if trend_exit and cpx<ma10.loc[dt].get(tk,np.nan): ex=True
            if age>=maxhold: ex=True
            if ex: pos.pop(tk)
        rk=rank.loc[dt]; cands=list(rk[rk<=N].sort_values().index)
        for tk in cands:
            if len(pos)>=N: break
            if tk not in pos and np.isfinite(px.get(tk,np.nan)): pos[tk]={"i":k,"px":px[tk],"peak":px[tk]}
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx):
            held=list(pos.keys()); nr=ret.iloc[k+1][held].dropna() if held else pd.Series(dtype=float)
            monthly.append((didx[k+1],nr.mean() if len(nr) else 0.0))
    return pd.Series(dict(monthly)).reindex(idx).fillna(0.0)
p(f"{'strategy':44} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
c,s,d=stats(qret); p(f"{'QQQ':44} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
runs={
 "composite monthly-rebal (no exits)":dict(score=comp,stop=None,trail=None,ladder=None,trend_exit=False,maxhold=1),
 "composite + staged + losscut":dict(score=comp),
 "composite+insider/rev + staged+losscut":dict(score=comp_ins),
 "composite N=15 + staged+losscut":dict(score=comp,N=15),
 "composite N=40 + staged+losscut":dict(score=comp,N=40),
}
series={}
for nm,cfg in runs.items():
    r=sim(**cfg); series[nm]=r; c,s,d=stats(r)
    p(f"{nm:44} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
best=series["composite + staged + losscut"]
# sub-periods for best
p("\nSub-period (composite+staged+losscut vs QQQ):")
for lo,hi in [("2012","2016"),("2017","2020"),("2021","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31"))
    c,s,_=stats(best[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f}  vs QQQ {qc:+.1%}/{qs:.2f}")
mex=~((idx>=pd.Timestamp('2023-01-01'))&(idx<=pd.Timestamp('2024-12-31')))
c,s,_=stats(best[mex]); qc,qs,_=stats(qret[mex]); p(f"  ex-2023-24: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
# equity curve
c,s,d=stats(best); qc,qs,qd=stats(qret)
ge=(1+best).cumprod(); gq=(1+qret).cumprod()
fig,ax=plt.subplots(figsize=(11,6))
ax.plot(ge.index,ge,label=f"Quality+LowVol+Momentum composite (CAGR {c:.0%}, Sh {s:.2f}, DD {d:.0%})",lw=2.3,color="#1f77b4")
ax.plot(gq.index,gq,label=f"QQQ (CAGR {qc:.0%}, Sh {qs:.2f}, DD {qd:.0%})",lw=2,color="#888")
ax.set_yscale("log"); ax.set_title("Improved composite (discovered factors + loss-cutting) vs QQQ, PIT clean 2012-2025")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/tmp/wave/composite_vs_qqq.png",dpi=110)
p(f"\nsaved /tmp/wave/composite_vs_qqq.png  DONE t={time.time()-t0:.0f}s")
