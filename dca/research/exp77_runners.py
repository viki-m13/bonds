import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean()
mom6=FEAT["mom6"]; mom3=me/me.shift(3)-1
# ---- best long-only selection score (discovered strong factors, long side) ----
def Z(nm): return (FEAT[nm].where(liq).rank(axis=1,pct=True)-0.5).fillna(0.0)
# value via featmat? add B/M quickly from fundamentals
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def qidx(df):
    df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=qidx(rev)
def gq(k):
    d=F.get(k); return qidx(d).reindex(columns=rev.columns) if d is not None else None
EQ,GP,AST,SH=gq("StockholdersEquity"),gq("GrossProfit"),gq("Assets"),gq("EntityCommonStockSharesOutstanding")
def qmap(df):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=6)
mcap=me*qmap(SH); BM=qmap(EQ)/mcap; GPA=qmap(GP*4)/qmap(AST)
def Zr(df): return (df.where(liq).rank(axis=1,pct=True)-0.5).fillna(0.0)
SCORE=( 1.0*Zr(GPA)+0.8*Zr(BM)+1.0*Z("roa")+0.8*Z("distHigh")+0.8*Z("mom6")
        -0.8*Z("vol6")-0.6*Z("share_chg")+0.5*Z("roe")+0.4*(FEAT["rev_accel"]) )
idx=M[(M>=pd.Timestamp("2012-07-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
elig=(liq&(me>=3.0)&(me>ma10))   # long-only timing gate: above 10mo MA
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
rank=SCORE.where(elig).rank(axis=1,ascending=False)
def sim(N=15, ride=True, trail=-0.30, stop=-0.25, cull=[(3,0.05),(6,0.20)], trend_exit=True, addwin=False):
    pos={}; cash=1.0; series=[]; nheld=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        # exits
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): cash+=0.0; pos.pop(tk); continue   # delisted: value already decayed via ret; drop
            e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1; age=k-e["i"]; ex=False
            if stop is not None and rs<=stop: ex=True
            if trail is not None and cpx/e["peak"]-1<=trail: ex=True
            if cull:
                for mm,thr in cull:
                    if age==mm and rs<thr: ex=True
            if trend_exit and cpx<ma10.loc[dt].get(tk,np.nan): ex=True
            if ex: cash+=e["val"]; pos.pop(tk)
        # optional rebalance (equal weight) if not ride
        if not ride and pos:
            tot=cash+sum(e["val"] for e in pos.values()); tgt=tot/max(N,len(pos))
            for e in pos.values(): e["val"]=tgt
            cash=tot-tgt*len(pos)
        # entries: fill up to N with best-ranked non-held
        if k+1<len(didx):
            rk=rank.loc[dt]; cands=[t for t in rk[rk<=N*3].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan))]
            need=N-len(pos)
            if need>0 and cash>1e-9 and cands:
                slice_=cash/need
                for tk in cands[:need]:
                    pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":slice_}; cash-=slice_
            # pyramid: add to winners (top-half performers) from any spare cash
            if addwin and cash>1e-3 and pos:
                win=[t for t in pos if px.get(t,0)/pos[t]["px"]-1>0.2]
                if win:
                    add=cash/len(win)
                    for t in win: pos[t]["val"]+=add
                    cash=0.0
        # apply next-month return
        eq0=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx):
            for tk in list(pos.keys()):
                r1=ret.iloc[k+1].get(tk,np.nan)
                pos[tk]["val"]*= (1+ (r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx):
            series.append((didx[k+1], eq1/eq0-1 if eq0>0 else 0.0)); nheld.append(len(pos))
    return pd.Series(dict(series)).reindex(idx).fillna(0.0), np.mean(nheld)
p(f"{'long-only config':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'N':>4}")
c,s,d=stats(qret); p(f"{'QQQ':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
cfgs={
 "N15 equal-rebal (baseline)":dict(N=15,ride=False),
 "N15 RIDE winners + cut losers":dict(N=15,ride=True),
 "N10 RIDE + cut losers":dict(N=10,ride=True),
 "N20 RIDE + cut losers":dict(N=20,ride=True),
 "N15 RIDE + pyramid winners":dict(N=15,ride=True,addwin=True),
 "N10 RIDE tight (trail-25,cull hard)":dict(N=10,ride=True,trail=-0.25,cull=[(3,0.08),(6,0.25)]),
}
best=None
for nm,cfg in cfgs.items():
    r,an=sim(**cfg); c,s,d=stats(r)
    p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%} {an:>4.0f}")
# pick best by CAGR for the headline
allr={nm:sim(**cfg)[0] for nm,cfg in cfgs.items()}
bycagr=max(allr, key=lambda k: stats(allr[k])[0]); bestr=allr[bycagr]
c,s,d=stats(bestr); p(f"\nbest CAGR config: {bycagr}  {c:.1%}/{s:.2f}/{d:.1%}")
# sub-periods
p("Sub-period (best-CAGR long-only vs QQQ):")
for lo,hi in [("2012","2016"),("2017","2020"),("2021","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(bestr[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
# figure
fig,ax=plt.subplots(figsize=(11,6))
g=(1+bestr).cumprod(); gq=(1+qret).cumprod(); c,s,d=stats(bestr); qc,qs,qd=stats(qret)
ax.plot(g.index,g,label=f"Long-only runner-rider ({bycagr}) CAGR {c:.0%}/Sh {s:.2f}/DD {d:.0%}",lw=2.4,color="#1f77b4")
ax.plot(gq.index,gq,label=f"QQQ CAGR {qc:.0%}/Sh {qs:.2f}/DD {qd:.0%}",lw=2,color="#888")
ax.set_yscale("log"); ax.set_title("Long-only concentrated runner-rider (no margin/short): ride winners + cut losers\nPIT clean 2012-2025")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/tmp/wave/runners_vs_qqq.png",dpi=110)
p(f"\nsaved /tmp/wave/runners_vs_qqq.png DONE t={time.time()-t0:.0f}s")
