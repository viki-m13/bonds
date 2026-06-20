import numpy as np, pandas as pd, time, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M); fnames=list(FEAT.keys())
ret=(me/me.shift(1)-1).clip(-0.9,2.0); ma10=me.rolling(10,min_periods=10).mean()
Z={nm:FEAT[nm].where(liq).rank(axis=1,pct=True) for nm in fnames}
# build training table
recs=[]
for dt in M[(M>=pd.Timestamp("2011-06-01"))]:
    fv=fok.loc[dt].dropna()
    if len(fv)<50: continue
    y=(fv>=fv.quantile(0.80)).astype(int)
    X=np.column_stack([Z[nm].loc[dt].reindex(fv.index).values for nm in fnames])
    for i,tk in enumerate(fv.index): recs.append((dt,tk,*X[i],y.iloc[i]))
DF=pd.DataFrame.from_records(recs,columns=["date","tk"]+fnames+["y"])
from sklearn.ensemble import HistGradientBoostingClassifier
preds=[]
for ytest in range(2015,2026):
    tr=DF[DF.date<pd.Timestamp(f"{ytest}-01-01")]; te=DF[(DF.date>=pd.Timestamp(f"{ytest}-01-01"))&(DF.date<=pd.Timestamp(f"{ytest}-12-31"))]
    if len(te)==0 or len(tr)<5000: continue
    clf=HistGradientBoostingClassifier(max_iter=200,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200)
    clf.fit(tr[fnames].values,tr["y"].values)
    t2=te[["date","tk"]].copy(); t2["p"]=clf.predict_proba(te[fnames].values)[:,1]; preds.append(t2)
PR=pd.concat(preds); p(f"ML trained t={time.time()-t0:.0f}s")
PROB=PR.pivot_table(index="date",columns="tk",values="p")          # prob panel
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim_ml(N=25,overlay=True):
    rank=PROB.reindex(M).rank(axis=1,ascending=False); pos={}; out=[]
    for k,dt in enumerate(didx):
        if dt not in PROB.index and overlay is None: pass
        px=me.loc[dt]
        if overlay:
            for tk in list(pos.keys()):
                e=pos[tk]; cpx=px.get(tk,np.nan)
                if not np.isfinite(cpx): pos.pop(tk); continue
                e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1; age=k-e["i"]; ex=False
                if rs<=-0.25 or cpx/e["peak"]-1<=-0.35 or cpx<ma10.loc[dt].get(tk,np.nan) or age>=36: ex=True
                for mm,thr in [(3,0.10),(6,0.30)]:
                    if age==mm and rs<thr: ex=True
                if ex: pos.pop(tk)
            if dt in rank.index:
                rk=rank.loc[dt]; cands=list(rk[rk<=N].sort_values().index)
                for tk in cands:
                    if len(pos)>=N: break
                    if tk not in pos and np.isfinite(px.get(tk,np.nan)): pos[tk]={"i":k,"px":px[tk],"peak":px[tk]}
            held=list(pos.keys())
        else:
            held=list(rank.loc[dt][rank.loc[dt]<=N].index) if dt in rank.index else []
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx):
            nr=ret.iloc[k+1][held].dropna() if held else pd.Series(dtype=float)
            out.append((didx[k+1],nr.mean() if len(nr) else 0.0))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)
ml_plain=sim_ml(25,overlay=False); ml_ov=sim_ml(25,overlay=True)
# composite sleeve (recompute) over same window
def Zc(nm): return (FEAT[nm].where(liq).rank(axis=1,pct=True)-0.5).fillna(0.0)
comp=(1.0*Zc("roa")+0.7*Zc("roe")-1.0*Zc("vol6")+1.0*Zc("distHigh")+0.7*Zc("mom6")-0.8*Zc("share_chg")+0.5*Zc("op_leverage"))
trend=(me>ma10); sc=comp.where(liq&trend); rk=sc.rank(axis=1,ascending=False)
w=(rk<=25).shift(1).fillna(False).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
compr=(w*ret).sum(axis=1).reindex(idx).fillna(0.0)
p(f"\n{'strategy (2015-2025)':40} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
for nm,r in [("QQQ",qret),("ML top25 plain",ml_plain),("ML top25 + losscut overlay",ml_ov),("Composite",compr)]:
    c,s,d=stats(r); p(f"{nm:40} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
p(f"\ncorr ML_ov-comp {ml_ov.corr(compr):.2f} ML_ov-QQQ {ml_ov.corr(qret):.2f}")
p(f"\n{'GRAND BLEND':40} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
blends={"50 ML_ov / 50 comp":0.5*ml_ov+0.5*compr,
        "40 ML_ov/40 comp/20 QQQ":0.4*ml_ov+0.4*compr+0.2*qret,
        "60 ML_ov / 40 comp":0.6*ml_ov+0.4*compr,
        "50 ML_ov/30 comp/20 QQQ":0.5*ml_ov+0.3*compr+0.2*qret}
best=None
for nm,r in blends.items():
    c,s,d=stats(r); p(f"{nm:40} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
    if best is None or s>best: best=s;bnm=nm;bser=r
p(f"\nbest: {bnm} Sharpe {best:.2f}")
# figure
fig,ax=plt.subplots(figsize=(11,6))
for nm,r,co,lw in [("Grand blend",bser,"#d62728",2.6),("ML+losscut",ml_ov,"#9467bd",1.4),("Composite",compr,"#2ca02c",1.4),("QQQ",qret,"#888",2)]:
    g=(1+r).cumprod(); c,s,d=stats(r); ax.plot(g.index,g,label=f"{nm} (CAGR {c:.0%}, Sh {s:.2f}, DD {d:.0%})",lw=lw,color=co)
ax.set_yscale("log"); ax.set_title(f"Grand blend: ML(losscut)+composite+QQQ vs QQQ (PIT clean 2015-2025)\n{bnm}")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/tmp/wave/grand_blend.png",dpi=110)
p(f"\nsaved /tmp/wave/grand_blend.png DONE t={time.time()-t0:.0f}s")
