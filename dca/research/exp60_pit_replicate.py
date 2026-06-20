import glob, numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
# ---- load daily PIT panels (adjClose, adjVolume) ----
acs,vos=[],[]
for f in sorted(glob.glob("/home/user/bonds/dca/research/data/tiingo/prices/ac_*.parquet")):
    d=pd.read_parquet(f)
    if d.shape[1]==0: continue
    d.index=pd.to_datetime(d.index); acs.append(d)
for f in sorted(glob.glob("/home/user/bonds/dca/research/data/tiingo/prices/vol_*.parquet")):
    d=pd.read_parquet(f)
    if d.shape[1]==0: continue
    d.index=pd.to_datetime(d.index); vos.append(d)
C=pd.concat(acs,axis=1).astype("float32"); C=C.loc[:,~C.columns.duplicated()].sort_index()
V=pd.concat(vos,axis=1).astype("float32"); V=V.loc[:,~V.columns.duplicated()].sort_index()
C=C[C.index>=pd.Timestamp("1999-01-01")]; V=V.reindex(C.index)
V=V[[c for c in C.columns if c in V.columns]]; C=C[V.columns]
p(f"daily panel C{C.shape} V{V.shape} t={time.time()-t0:.0f}s")
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
uni["endDate"]=pd.to_datetime(uni.endDate,errors="coerce")
deli=set(uni[(uni.assetType=="Stock")&(uni.endDate<'2025-06-01')].ticker)
stocks=set(uni[uni.assetType=="Stock"].ticker)
# ---- liquidity filter: keep names whose 50d avg $vol ever >= $10M ----
dv=(C*V); adv50=dv.rolling(50,min_periods=30).mean()
liqcols=[c for c in C.columns if c in stocks and (adv50[c].max()>=10e6)]
C=C[liqcols]; V=V[liqcols]; adv50=adv50[liqcols]
p(f"liquid stock universe: {len(liqcols)}  (delisted {len([c for c in liqcols if c in deli])})  t={time.time()-t0:.0f}s")
# ---- daily technical features (close-based replication of signals.ts) ----
sma20=C.rolling(20,min_periods=20).mean(); sma50=C.rolling(50,min_periods=50).mean(); sma200=C.rolling(200,min_periods=200).mean()
bullStack=(sma20>sma50)&(sma50>sma200)&(C>sma20)&(C>sma50)&(C>sma200)
hi252=C.rolling(252,min_periods=120).max(); pctFromHigh=C/hi252-1
hi20prior=C.shift(1).rolling(20,min_periods=20).max(); newHigh20=C>hi20prior
ret21=C/C.shift(21)-1; ret63=C/C.shift(63)-1; ret126=C/C.shift(126)-1
rc=C.pct_change(); dailyVol60=rc.rolling(60,min_periods=40).std()
sma50_20ago=sma50.shift(20); sma50Rising=sma50>sma50_20ago
vol10=V.rolling(10,min_periods=10).mean(); vol50=V.rolling(50,min_periods=50).mean(); volRamp=vol10/vol50
imminent=sma50Rising&(C>sma50)&(volRamp>1.2)
# RSI14
delta=C.diff(); up=delta.clip(lower=0); dn=(-delta).clip(lower=0)
rs=up.rolling(14,min_periods=14).mean()/dn.rolling(14,min_periods=14).mean().replace(0,np.nan)
rsi=100-100/(1+rs); rsi=rsi.fillna(100)
pctAbove200=C/sma200-1
# extension guards (hard discards)
extended=((rsi>76)|(ret126>3.5)|(pctAbove200>1.5)|((dailyVol60>0.055)&((pctAbove200>0.5)|(ret126>1.0)))|((dailyVol60>0.09)&(ret126>2.0)))
rsiOK=(rsi>=40)&(rsi<=72)
near52=(pctFromHigh>-0.20)&(pctFromHigh<=-0.0)   # within 20% but not AT the high (chase penalty)
liqOK=(adv50>=10e6)&(C>=3.0)
p(f"features built t={time.time()-t0:.0f}s")
# ---- define selection variants (price-replicable Elite/Moon structure) ----
VAR={
 "BASE liquid univ":      liqOK,
 "bull MA stack":         bullStack&liqOK,
 "STRICT-core(stack+near52)": bullStack&near52&liqOK,
 "MOON-tech(+breakout/imm,guard)": bullStack&near52&(newHigh20|imminent)&(~extended)&rsiOK&liqOK,
 "MOON-tech-imminent only": bullStack&near52&imminent&(~extended)&rsiOK&liqOK,
}
# ---- monthly snapshots; forward returns (delisting-inclusive) ----
mdates=pd.date_range(C.index.min(),C.index.max(),freq="ME")
mdates=[C.index[C.index.get_indexer([d],method="ffill")[0]] for d in mdates if d>=C.index.min()]
mdates=sorted(set([d for d in mdates if d>=pd.Timestamp("2000-01-01")]))
qqq=C["QQQ"] if "QQQ" in C.columns else None
def fwd(dt,h):  # forward h-trading-day return per stock, last-price if delisted before h
    i=C.index.get_loc(dt)
    if i+1>=len(C.index): return None
    cur=C.iloc[i]
    j=min(i+h,len(C.index)-1)
    fwdpx=C.iloc[i+1:j+1].ffill().iloc[-1]   # last available price within horizon (captures delist loss via ffill of last real)
    # if a name has NO price after i (delisted exactly), fwdpx stays NaN -> treat as -100% if it was priced
    r=(fwdpx/cur-1)
    return r
H=252  # 12-month forward
p(f"\n=== forward 12m (252td) returns by selection variant, 2000-2024 ===")
p(f"snapshots={len(mdates)}  horizon={H}td")
rowsdt=[d for d in mdates if d<=pd.Timestamp("2024-06-30")]
res={k:{"r":[],"deli":[]} for k in VAR}
univ_r=[]; qqq_r=[]
for dt in rowsdt:
    r=fwd(dt,H)
    if r is None: continue
    rc_=r.clip(-1.0,10.0)  # winsorize forward returns (cap 10x for sanity in means)
    base=VAR["BASE liquid univ"].loc[dt]
    base=base[base].index
    if len(base)<20: continue
    univ_r.append(rc_.reindex(base).mean())
    if qqq is not None:
        qi=C.index.get_loc(dt); qj=min(qi+H,len(C.index)-1)
        qqq_r.append(C["QQQ"].iloc[qj]/C["QQQ"].iloc[qi]-1)
    for k,flag in VAR.items():
        sel=flag.loc[dt]; sel=sel[sel].index
        if len(sel)==0: continue
        rr=rc_.reindex(sel).dropna()
        res[k]["r"].append(rr.mean())
        res[k]["deli"].append(len([s for s in sel if s in deli])/max(1,len(sel)))
p(f"\n{'variant':32} {'avgN/mo':>8} {'mean12m':>8} {'median':>8} {'hit>0':>7} {'2x rate':>7} {'%deli':>6}")
# need per-snapshot counts and pooled stock-level stats too; recompute pooled
def pooled(flag):
    allr=[]; n=[]
    for dt in rowsdt:
        r=fwd(dt,H)
        if r is None: continue
        rc_=r.clip(-1.0,10.0)
        sel=flag.loc[dt]; sel=sel[sel].index
        rr=rc_.reindex(sel).dropna()
        if len(rr): allr.append(rr); n.append(len(rr))
    if not allr: return None
    A=pd.concat(allr)
    return np.mean(n),A.mean(),A.median(),(A>0).mean(),(A>1.0).mean()
for k,flag in VAR.items():
    out=pooled(flag)
    if out is None: p(f"{k:32} (none)"); continue
    avgn,mean_,med_,hit_,two_=out
    deli_=np.mean(res[k]["deli"]) if res[k]["deli"] else 0
    p(f"{k:32} {avgn:>8.0f} {mean_:>8.1%} {med_:>8.1%} {hit_:>7.1%} {two_:>7.1%} {deli_:>6.0%}")
p(f"\nQQQ fwd12m mean over snapshots: {np.mean(qqq_r):.1%}" if qqq_r else "QQQ n/a")
p(f"Universe(liquid) fwd12m mean:    {np.mean(univ_r):.1%}")
# ---- by-era for the MOON-tech variant ----
p(f"\n=== MOON-tech by era (mean fwd12m, 2x rate, avgN) ===")
flag=VAR["MOON-tech(+breakout/imm,guard)"]
for lo,hi in [("2000","2009"),("2010","2019"),("2020","2024")]:
    sub=[d for d in rowsdt if pd.Timestamp(lo)<=d<=pd.Timestamp(hi+"-12-31")]
    allr=[]
    for dt in sub:
        r=fwd(dt,H);
        if r is None: continue
        sel=flag.loc[dt]; sel=sel[sel].index
        rr=r.clip(-1,10).reindex(sel).dropna()
        if len(rr): allr.append(rr)
    if allr:
        A=pd.concat(allr); p(f"  {lo}-{hi}: mean {A.mean():>6.1%}  2x {(A>1.0).mean():>5.1%}  n/mo {np.mean([len(x) for x in allr]):>5.0f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
