import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1
accel=PROB-PROB.shift(2)
el=(liq&(me>=3.0)&(me>ma10)&(mom3>0)&(accel>0)).fillna(False).astype(bool)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
sc=PROB.where(el); rank=sc.rank(axis=1,ascending=False)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(stop_early=-0.20, trail_proven=-0.40, prove=0.30, trend_exit_unproven=True, N=12):
    pos={}; cash=1.0; out=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1; proven=e["peak"]/e["px"]-1>=prove
            ex=False
            if proven:
                if cpx/e["peak"]-1<=trail_proven: ex=True              # loose trail for runners
            else:
                if rs<=stop_early: ex=True                              # tight stop for unproven
                if trend_exit_unproven and cpx<ma10.loc[dt].get(tk,np.nan): ex=True
            if ex: cash+=e["val"]; pos.pop(tk)
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
# champion baseline (single-stage trail -30 + trend)
def champ(N=12):
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
p(f"{'config':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
c,s,d=stats(qret); p(f"{'QQQ':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
r=champ(); c,s,d=stats(r); p(f"{'champion (single trail -30 + trend)':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
for nm,cfg in {
 "two-stage: stop-20/trend, proven>30 trail-40":dict(stop_early=-0.20,trail_proven=-0.40,prove=0.30),
 "two-stage: stop-15/trend, proven>25 trail-45":dict(stop_early=-0.15,trail_proven=-0.45,prove=0.25),
 "two-stage: stop-20/trend, proven>40 trail-50":dict(stop_early=-0.20,trail_proven=-0.50,prove=0.40),
 "two-stage tight: stop-15/trend, proven>30 trail-35":dict(stop_early=-0.15,trail_proven=-0.35,prove=0.30),
}.items():
    r=sim(**cfg); c,s,d=stats(r); p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
# best two-stage sub-periods
best=sim(stop_early=-0.15,trail_proven=-0.45,prove=0.25)
p("\nbest two-stage sub-periods vs QQQ:")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(best[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
p(f"DONE t={time.time()-t0:.0f}s")
