import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean()
mom3=me/me.shift(3)-1; roa=FEAT["roa"]; accel=PROB-PROB.shift(2)
hi12=me.rolling(12,min_periods=6).max(); distHigh=me/hi12-1
# deep drawdown in last 6 months (was >40% below 12m high), now reclaiming 10mo MA from below
wasdeep=(distHigh.rolling(6,min_periods=3).min()< -0.40)
reclaim=(me>ma10)&(me.shift(2)<ma10.shift(2))   # crossed above MA within ~2 mo
qual=(roa.where(liq).rank(axis=1,pct=True)>0.4)|(FEAT["ins_clustern"]>=2)
turn_elig=(liq&(me>=3.0)&wasdeep&reclaim&qual&(mom3>0)).fillna(False).astype(bool)
champ_elig=(liq&(me>=3.0)&(me>ma10)&(mom3>0)&(accel>0)).fillna(False).astype(bool)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qret=qpx.pct_change().reindex(idx)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(elig, score, N=12):
    sc=score.where(elig); rank=sc.rank(axis=1,ascending=False)
    pos={}; cash=1.0; out=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx)
            if cpx/e["peak"]-1<=-0.30 or cpx<ma10.loc[dt].get(tk,np.nan): cash+=e["val"]; pos.pop(tk)
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
champ=sim(champ_elig,PROB,12)
turn=sim(turn_elig,PROB.where(turn_elig).fillna(distHigh.rank(axis=1,pct=True)),10)  # rank turnarounds by ML where avail else recovery strength
p(f"{'sleeve (2015-2025)':40} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'avgN':>5}")
for nm,r,el in [("QQQ",qret,None),("WAVE champion (momentum)",champ,champ_elig),("Turnaround (recovery)",turn,turn_elig)]:
    c,s,d=stats(r); an=el.reindex(idx).sum(axis=1).mean() if el is not None else 0; p(f"{nm:40} {c:>7.1%} {s:>7.2f} {d:>7.1%} {an:>5.0f}")
p(f"\ncorr(champ,turn)={champ.corr(turn):.2f} corr(turn,QQQ)={turn.corr(qret):.2f}")
p(f"\n{'blend':40} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
best=None
for nm,r in {"70 champ / 30 turn":0.7*champ+0.3*turn,"60/40":0.6*champ+0.4*turn,"50/50":0.5*champ+0.5*turn,
             "60 champ/20 turn/20 QQQ":0.6*champ+0.2*turn+0.2*qret}.items():
    c,s,d=stats(r); p(f"{nm:40} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
    if best is None or s>best: best=s;bser=r;bnm=nm
p(f"\nbest: {bnm} Sharpe {best:.2f}")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(bser[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
fig,ax=plt.subplots(figsize=(11,6))
g=(1+bser).cumprod(); gq=(1+qret).cumprod(); c,s,d=stats(bser); qc,qs,qd=stats(qret)
ax.plot(g.index,g,label=f"WAVE+turnaround blend CAGR {c:.0%}/Sh {s:.2f}/DD {d:.0%}",lw=2.4,color="#1f77b4")
ax.plot(gq.index,gq,label=f"QQQ CAGR {qc:.0%}/Sh {qs:.2f}/DD {qd:.0%}",lw=2,color="#888")
ax.set_yscale("log"); ax.set_title("Long-only: momentum runners + turnaround recoveries (bimodal moonshot capture)")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/tmp/wave/wave_turnaround.png",dpi=110)
p(f"\nsaved /tmp/wave/wave_turnaround.png DONE t={time.time()-t0:.0f}s")
