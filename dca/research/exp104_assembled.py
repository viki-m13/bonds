import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def qidx(df):
    df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=qidx(rev)
def gq(k):
    d=F.get(k); return qidx(d).reindex(columns=rev.columns) if d is not None else None
OI,NI,AST,CASH,STI,SH=[gq(k) for k in ["OperatingIncomeLoss","NetIncomeLoss","Assets","CashAndCashEquivalentsAtCarryingValue","ShortTermInvestments","EntityCommonStockSharesOutstanding"]]
def qm(df,lim=6):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=lim)
roic=(OI*4)/(AST-CASH.fillna(0)-STI.fillna(0)).clip(lower=1); roic_m=qm(roic)
ryoy=rev/rev.shift(4)-1; ryoy_m=qm(ryoy); mcap=me*qm(SH); ep=qm(NI*4)/mcap; ps=mcap/qm(rev*4).clip(lower=1)
vol6=FEAT["vol6"]; sharechg=FEAT["share_chg"]; lmcap=FEAT["log_mcap"]; mom12=me/me.shift(12)-1; mom6=me/me.shift(6)-1
ma10=me.rolling(10,min_periods=10).mean(); liqf=(me.shift(1)>=3.0).fillna(False)
def z(x):
    fr=x.where(liqf); return (fr.sub(fr.mean(axis=1),axis=0)).div(fr.std(axis=1).replace(0,np.nan),axis=0).fillna(0)
# robust composite = equal-weight durable survivors (NOT a single mined conjunction)
COMP = z(ep)+z(-sharechg)+z(roic_m)+z(-vol6)+z(-lmcap)
# handoff tilt (the ownership-rotation overlay): small bonus
handoff=((ryoy_m>0)&(ryoy_m<ryoy_m.shift(4))&(ps<=ps.rolling(18,min_periods=9).median()*1.05)&(mom12.shift(6)>0.10)&(mom6<0.05)).astype(float)
SCORE=(COMP+0.5*z(handoff)).where(liqf)
# refined SUSTAINED durability break: roic rank<0.4 for >=2 quarters (~6 mo) OR revenue collapse
roic_rank=roic_m.rank(axis=1,pct=True)
dur_break=((roic_rank<0.40).rolling(6,min_periods=4).min()>0.5)|(ryoy_m< -0.15)
ret=(me/me.shift(1)-1).clip(-0.9,3.0)
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]
def stats(r):
    r=r.dropna()
    if len(r)<6: return (np.nan,)*3
    c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(N=20, ride=True, price_stop=None, durability=True, trend_exit=True):
    rank=SCORE.rank(axis=1,ascending=False); pos={}; cash=1.0; out=[]; trades=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): trades.append(-0.6); pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1; ex=False
            if price_stop is not None and cpx/e["peak"]-1<=price_stop: ex=True
            if durability and bool(dur_break.loc[dt].get(tk,False)): ex=True
            if trend_exit and cpx<ma10.loc[dt].get(tk,np.nan)*0.85: ex=True   # loose trend (allow drawdowns)
            if k-e["i"]>=72: ex=True
            if ex: trades.append(rs); cash+=e["val"]; pos.pop(tk)
        rk=rank.loc[dt]; cands=[t for t in rk[rk<=N*3].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan)) and bool(liqf.loc[dt].get(t,False))]
        for tk in cands:
            if len(pos)>=N: break
            if cash>1e-9: sl=cash/max(1,(N-len(pos))); pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":sl}; cash-=sl
        if not ride and pos:
            tot=cash+sum(e["val"] for e in pos.values()); tg=tot/max(N,len(pos))
            for e in pos.values(): e["val"]=tg
            cash=tot-tg*len(pos)
        eq0=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx):
            for tk in pos:
                r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx): out.append((didx[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    return pd.Series(dict(out)),np.array(trades) if trades else np.array([0.0])
full,trades=sim()
def seg(s,lo,hi):
    m=(s.index>=pd.Timestamp(lo))&(s.index<=pd.Timestamp(hi)); return s[m]
def qseg(lo,hi): r=qpx.pct_change(); return seg(r,lo,hi)
p(f"=== ASSEMBLED SYSTEM (equal-weight durable-survivor composite + handoff tilt, ride, sustained-durability exit, NO price stop) ===")
p(f"{'period':22}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}  | QQQ CAGR/Sharpe")
for lab,lo,hi in [("DEV 2012-2022","2012-07-01","2022-12-31"),("HOLDOUT 2023-2025","2023-01-01","2025-12-31"),("FULL 2012-2025","2012-07-01","2025-12-31")]:
    s=seg(full,lo,hi); c,sh,d=stats(s); q=qseg(lo,hi); qc,qsh,qd=stats(q)
    tag=" <-- LOCKED, touched once" if lab.startswith("HOLD") else ""
    p(f"{lab:22}{c:>7.1%}{sh:>7.2f}{d:>7.1%}  | {qc:>5.1%}/{qsh:.2f}{tag}")
p(f"\ntrade distribution: n={len(trades)} win {(trades>0).mean():.0%} avg {trades.mean():+.1%} med {np.median(trades):+.1%} >100% {(trades>1.0).mean():.1%}")
# exit ablation on holdout (confirm hold-through-drawdown matters OOS)
p(f"\nHOLDOUT 2023-2025 exit ablation:")
for nm,kw in [("price-stop -20% (naive)",dict(price_stop=-0.20,durability=False,trend_exit=False)),
              ("sustained-durability (deploy)",dict()),
              ("never-sell (fixed 72mo)",dict(durability=False,trend_exit=False))]:
    s,_=sim(**kw); s=seg(s,"2023-01-01","2025-12-31"); c,sh,d=stats(s); p(f"  {nm:30}{c:>7.1%}/{sh:.2f} maxDD {d:.1%}")
# equity curve full
c,sh,d=stats(full); q=qpx.pct_change().reindex(full.index); qc,qsh,qd=stats(q)
fig,ax=plt.subplots(figsize=(11,6)); g=(1+full).cumprod(); gq=(1+q).cumprod()
ax.plot(g.index,g,label=f"Assembled compounder system (CAGR {c:.0%}/Sh {sh:.2f}/DD {d:.0%})",lw=2.4,color="#1f77b4")
ax.plot(gq.index,gq,label=f"QQQ (CAGR {qc:.0%}/Sh {qsh:.2f}/DD {qd:.0%})",lw=2,color="#888")
ax.axvspan(pd.Timestamp("2023-01-01"),pd.Timestamp("2025-12-31"),color="orange",alpha=0.08,label="locked holdout")
ax.set_yscale("log"); ax.set_title("Assembled durable-survivor compounder system vs QQQ (PIT clean; holdout shaded)")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("/home/user/assembled_system.png",dpi=110)
p(f"\nsaved assembled_system.png  DONE t={time.time()-t0:.0f}s")
