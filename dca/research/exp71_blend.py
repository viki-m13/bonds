import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
ACC2,HY2,INS2,TECH2=pd.read_pickle("/tmp/wave/_qual_masks.pkl")
M=me.index; didx=list(M)
ret=(me/me.shift(1)-1).clip(-0.9,2.0)
ma10=me.rolling(10,min_periods=10).mean(); trend=(me>ma10)
idx=M[(M>=pd.Timestamp("2012-07-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
# ---- sleeve 1: moonshot qualifier + staged + losscut ----
ENTRY=((ACC2|HY2)&INS2&TECH2&liq)
mom6=FEAT["mom6"]
def sim_entry(N=25,stop=-0.25,trail=-0.35,ladder=[(3,0.10),(6,0.30)]):
    pos={}; out=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1; age=k-e["i"]; ex=False
            if rs<=stop or cpx/e["peak"]-1<=trail or cpx<ma10.loc[dt].get(tk,np.nan) or age>=36: ex=True
            for mm,thr in ladder:
                if age==mm and rs<thr: ex=True
            if ex: pos.pop(tk)
        ent=ENTRY.loc[dt].fillna(False).astype(bool); cands=sorted([t for t in ent.index[ent.values] if t not in pos],
            key=lambda t:-(mom6.loc[dt].get(t,-9) if np.isfinite(mom6.loc[dt].get(t,np.nan)) else -9))
        for tk in cands:
            if len(pos)>=N: break
            if np.isfinite(px.get(tk,np.nan)): pos[tk]={"i":k,"px":px[tk],"peak":px[tk]}
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx):
            held=list(pos.keys()); nr=ret.iloc[k+1][held].dropna() if held else pd.Series(dtype=float)
            out.append((didx[k+1],nr.mean() if len(nr) else 0.0))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)
# ---- sleeve 2: composite quality/lowvol/momentum monthly rebal ----
def Z(nm):
    f=FEAT[nm].where(liq); return (f.rank(axis=1,pct=True)-0.5).fillna(0.0)
comp=(1.0*Z("roa")+0.7*Z("roe")-1.0*Z("vol6")+1.0*Z("distHigh")+0.7*Z("mom6")-0.8*Z("share_chg")+0.5*Z("op_leverage"))
def sim_score(N=25):
    sc=comp.where(liq&trend); rank=sc.rank(axis=1,ascending=False)
    w=(rank<=N).shift(1).fillna(False).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    return (w*ret).sum(axis=1).reindex(idx).fillna(0.0)
moon=sim_entry(); compr=sim_score()
p(f"sleeves built t={time.time()-t0:.0f}s")
sl={"MOONSHOT (rev+ins+staged+losscut)":moon,"COMPOSITE (qual+lowvol+mom)":compr,"QQQ":qret}
p(f"\n{'sleeve/blend':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
for nm,r in sl.items():
    c,s,d=stats(r); p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
p(f"\ncorr: moon-comp {moon.corr(compr):.2f}  moon-QQQ {moon.corr(qret):.2f}  comp-QQQ {compr.corr(qret):.2f}")
p(f"\n{'blend':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
blends={
 "50 moon / 50 comp":0.5*moon+0.5*compr,
 "40 moon / 40 comp / 20 QQQ":0.4*moon+0.4*compr+0.2*qret,
 "33/33/33 moon/comp/QQQ":(moon+compr+qret)/3,
 "60 moon / 40 comp":0.6*moon+0.4*compr,
 "50 moon / 30 comp / 20 QQQ":0.5*moon+0.3*compr+0.2*qret,
}
best=None;bnm=None
for nm,r in blends.items():
    c,s,d=stats(r); p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
    if best is None or s>best: best=s;bnm=nm;bser=r
# risk-parity-ish (inverse vol) monthly blend of moon+comp
iv_m=1/moon.rolling(12,min_periods=6).std(); iv_c=1/compr.rolling(12,min_periods=6).std()
wsum=iv_m+iv_c; rp=(iv_m/wsum).shift(1)*moon+(iv_c/wsum).shift(1)*compr
c,s,d=stats(rp.dropna()); p(f"{'inverse-vol moon/comp':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
p(f"\nbest blend: {bnm} (Sharpe {best:.2f})")
# sub-periods of best blend
p(f"\nSub-period best blend vs QQQ:")
for lo,hi in [("2012","2016"),("2017","2020"),("2021","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31"))
    c,s,_=stats(bser[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
# figure: best blend + sleeves vs QQQ
fig,ax=plt.subplots(figsize=(11,6))
for nm,r,co,lw in [("Best blend",bser,"#d62728",2.6),("Moonshot sleeve",moon,"#1f77b4",1.5),
                   ("Composite sleeve",compr,"#2ca02c",1.5),("QQQ",qret,"#888",2)]:
    g=(1+r).cumprod(); c,s,d=stats(r)
    ax.plot(g.index,g,label=f"{nm} (CAGR {c:.0%}, Sh {s:.2f}, DD {d:.0%})",lw=lw,color=co)
ax.set_yscale("log"); ax.set_title(f"Blended proprietary sleeves vs QQQ (PIT clean 2012-2025)\nbest: {bnm}")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/tmp/wave/blend_vs_qqq.png",dpi=110)
p(f"\nsaved /tmp/wave/blend_vs_qqq.png DONE t={time.time()-t0:.0f}s")
