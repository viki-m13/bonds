import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean()
mom3=me/me.shift(3)-1; mom6=FEAT["mom6"]; roa=FEAT["roa"]
accel=PROB-PROB.shift(2)
hi6=me.rolling(6,min_periods=4).max(); newhi=(me>=hi6*0.98)
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qmom6=(qpx/qpx.shift(6)-1)
relstr=mom6.sub(qmom6,axis=0)>0
qualfloor=roa.where(liq).rank(axis=1,pct=True)>0.5
accel2=(accel>0)&(mom3>mom3.shift(2))   # ML rising AND price-momentum rising
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=qpx.pct_change().reindex(idx)
base_elig=(liq&(me>=3.0)&(me>ma10)&(mom3>0)&(accel>0))   # champion base: runner+accel gates
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(extra=None, N=12):
    el=base_elig if extra is None else (base_elig & extra.reindex_like(base_elig).fillna(False))
    el=el.fillna(False).astype(bool)
    sc=PROB.where(el); rank=sc.rank(axis=1,ascending=False)
    pos={}; cash=1.0; out=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx)
            if cpx/e["peak"]-1<=-0.30 or cpx<ma10.loc[dt].get(tk,np.nan): cash+=e["val"]; pos.pop(tk)
        if dt in PROB.index:
            rk=rank.loc[dt]; cands=[t for t in rk[rk<=N*4].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan))]
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
p(f"{'champion + extra gate (2015-2025)':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
c,s,d=stats(qret); p(f"{'QQQ':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
res={}
cfgs={"champion (runner+accel)":None,"+ fresh 6mo-high":newhi,"+ quality floor (ROA>med)":qualfloor,
      "+ market-relative strength":relstr,"+ double-accel (ML&price)":accel2,
      "+ qualfloor + relstr":qualfloor&relstr,"+ qualfloor + fresh-high":qualfloor&newhi}
for nm,g in cfgs.items():
    r=sim(g); res[nm]=r; c,s,d=stats(r); p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
# N sweep on best gate
best=max(res,key=lambda k: stats(res[k])[1]); bg=cfgs[best]
p(f"\nbest gate: {best} — N sweep:")
for N in [8,10,12,15,20]:
    r=sim(bg,N); c,s,d=stats(r); p(f"  N={N:>2}: {c:>6.1%}/{s:.2f}/{d:.1%}")
br=res[best]
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(br[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
