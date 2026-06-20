import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
stocks=set(uni[uni.assetType=="Stock"].ticker)
ACC2,HY2,INS2,TECH2=pd.read_pickle("/tmp/wave/_qual_masks.pkl")
cols=[c for c in me.columns if c in stocks]; me=me[cols]
ret=(me/me.shift(1)-1).clip(-0.9,2.0); liq=(me.shift(1)>=3.0).fillna(False)
ma10=me.rolling(10,min_periods=10).mean(); mom6=me/me.shift(6)-1
ENTRY=((ACC2|HY2)&INS2&TECH2&liq)[cols]
didx=list(me.index)
idx=me.index[(me.index>=pd.Timestamp("2012-07-01"))&(me.index<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(stop=None, trail=None, ladder=None, trend_exit=True, maxhold=36, maxpos=25):
    # ladder: list of (months, min_ret) cull rungs (SOS style): if at month>=k and ret<thr -> cull
    pos={}; monthly=[]; nh=[]; trades=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx):
                trades.append(-0.9); pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx); r_since=cpx/e["px"]-1; age=k-e["i"]; ex=False
            if stop is not None and r_since<=stop: ex=True
            if trail is not None and cpx/e["peak"]-1<=trail: ex=True
            if ladder:
                for (mm,thr) in ladder:
                    if age==mm and r_since<thr: ex=True
            if trend_exit and cpx<ma10.loc[dt].get(tk,np.nan): ex=True
            if age>=maxhold: ex=True
            if ex: trades.append(r_since); pos.pop(tk)
        ent=ENTRY.loc[dt]; cands=[t for t in ent.index[ent.values] if t not in pos]
        cands=sorted(cands,key=lambda t:-(mom6.loc[dt].get(t,-9) if np.isfinite(mom6.loc[dt].get(t,np.nan)) else -9))
        for tk in cands:
            if len(pos)>=maxpos: break
            pos[tk]={"i":k,"px":px.get(tk,np.nan),"peak":px.get(tk,np.nan)}
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx):
            held=list(pos.keys()); nr=ret.iloc[k+1][held].dropna() if held else pd.Series(dtype=float)
            monthly.append((didx[k+1],nr.mean() if len(nr) else 0.0)); nh.append(len(held))
    s=pd.Series(dict(monthly)).reindex(idx).fillna(0.0)
    tr=np.array(trades) if trades else np.array([0])
    return s,np.mean(nh),(tr>0).mean(),tr.mean()
p(f"{'exit rule (cap25, staged hold)':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'win%':>6} {'avgTr':>6}")
c,s,d=stats(qret); p(f"{'QQQ':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
CFG={
 "trend-only (base)":dict(),
 "stop -20%":dict(stop=-0.20),
 "stop -30%":dict(stop=-0.30),
 "trailing -25%":dict(trail=-0.25),
 "trailing -35%":dict(trail=-0.35),
 "SOS ladder (3m<10%,6m<30% cull)":dict(ladder=[(3,0.10),(6,0.30)]),
 "SOS ladder + stop-25%":dict(ladder=[(3,0.10),(6,0.30)],stop=-0.25),
 "ladder(1m<0,3m<10%)+trail35":dict(ladder=[(1,0.0),(3,0.10)],trail=-0.35),
 "stop-25 + trail-35 + ladder":dict(stop=-0.25,trail=-0.35,ladder=[(3,0.10),(6,0.30)]),
 "no trend-exit, stop-30 only":dict(trend_exit=False,stop=-0.30,maxhold=36),
}
best=None;bn=None
for nm,cfg in CFG.items():
    s_,an,win,avgtr=sim(**cfg); c,sh,d=stats(s_)
    p(f"{nm:46} {c:>7.1%} {sh:>7.2f} {d:>7.1%} {win:>6.0%} {avgtr:>6.1%}")
    if best is None or sh>best: best=sh;bn=nm
p(f"\nbest Sharpe config: {bn} ({best:.2f})")
p(f"DONE t={time.time()-t0:.0f}s")
