import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1
accel=PROB-PROB.shift(2); vol6=FEAT["vol6"]; roa=FEAT["roa"]; mom6=FEAT["mom6"]; lmcap=FEAT["log_mcap"]
el=(liq&(me>=3.0)&(me>ma10)&(mom3>0)&(accel>0)).fillna(False).astype(bool)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
sc=PROB.where(el); rank=sc.rank(axis=1,ascending=False)
N=12; pos={}; cash=1.0; trades=[]
for k,dt in enumerate(didx):
    px=me.loc[dt]
    for tk in list(pos.keys()):
        e=pos[tk]; cpx=px.get(tk,np.nan)
        if not np.isfinite(cpx):
            trades.append({**e["feat"],"ret":-0.6,"held":k-e["i"],"tk":tk}); pos.pop(tk); continue
        e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1
        if cpx/e["peak"]-1<=-0.30 or cpx<ma10.loc[dt].get(tk,np.nan):
            trades.append({**e["feat"],"ret":rs,"held":k-e["i"],"tk":tk}); cash+=e["val"]; pos.pop(tk)
    if dt in PROB.index and dt>=idx[0] and dt<=idx[-1]:
        rk=rank.loc[dt]; cands=[t for t in rk[rk<=N*4].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan))]
        need=N-len(pos)
        if need>0 and cash>1e-9 and cands:
            sl=cash/need
            for tk in cands[:need]:
                feat={"lmcap":lmcap.loc[dt].get(tk,np.nan),"vol6":vol6.loc[dt].get(tk,np.nan),
                      "mom6":mom6.loc[dt].get(tk,np.nan),"roa":roa.loc[dt].get(tk,np.nan),"entry":dt}
                pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":sl,"feat":feat}; cash-=sl
    if k+1<len(didx):
        for tk in pos:
            r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
T=pd.DataFrame(trades)
p(f"closed trades: {len(T)}  (2015-2025)  t={time.time()-t0:.0f}s")
p(f"win rate: {(T.ret>0).mean():.0%}  avg ret/trade: {T.ret.mean():+.1%}  median: {T.ret.median():+.1%}")
p(f"avg WIN: {T.ret[T.ret>0].mean():+.1%}  avg LOSS: {T.ret[T.ret<=0].mean():+.1%}  >50%ers: {(T.ret>0.5).mean():.0%}  >100%: {(T.ret>1.0).mean():.1%}")
p(f"avg holding: {T.held.mean():.1f} mo  (winners {T.held[T.ret>0].mean():.1f} / losers {T.held[T.ret<=0].mean():.1f})")
# by mcap bucket (log mcap quartile)
T["mcap_b"]=pd.qcut(T.lmcap,4,labels=["micro","small","mid","large"])
p(f"\nby market-cap bucket:  avg ret  /  win%  /  n")
for b,g in T.groupby("mcap_b",observed=True):
    p(f"  {str(b):6}: {g.ret.mean():+6.1%}  {(g.ret>0).mean():.0%}  n={len(g)}")
# by entry vol bucket
T["vol_b"]=pd.qcut(T.vol6,4,labels=["lowvol","q2","q3","hivol"])
p(f"\nby entry-volatility bucket: avg ret / win% / n")
for b,g in T.groupby("vol_b",observed=True):
    p(f"  {str(b):6}: {g.ret.mean():+6.1%}  {(g.ret>0).mean():.0%}  n={len(g)}")
# winners vs losers feature means
p(f"\nWINNERS vs LOSERS entry features:")
w=T[T.ret>0]; l=T[T.ret<=0]
for f in ["lmcap","vol6","mom6","roa"]:
    p(f"  {f:7}: win {w[f].mean():+.3f}  loss {l[f].mean():+.3f}")
# return contribution: top trades share
T2=T.sort_values("ret",ascending=False)
p(f"\ntop 5% of trades capture {T2.ret.head(int(len(T)*0.05)).sum()/T.ret.sum():.0%} of total trade-return (fat tails)")
p(f"DONE t={time.time()-t0:.0f}s")
