import pandas as pd, numpy as np, warnings, time; warnings.filterwarnings("ignore")
t0=time.time()
F=pd.read_pickle("/tmp/wave/_featmat.pkl"); liq,me,cols=F["liq"],F["me"],F["cols"]
M=me.index; ret=(me/me.shift(1)-1).clip(-0.95,5.0)
mom=me.shift(1)/me.shift(12)-1
U=(liq&(me.shift(1)>=5.0)).fillna(False)
ma10=me.rolling(10,min_periods=10).mean()
rk=mom.where(U).rank(axis=1,ascending=False)
idx=M[(M>=pd.Timestamp("1991-02-01"))&(M<=pd.Timestamp("2025-12-31"))]
didx=list(idx); N=20
# precompute per-month: top-200 candidate tickers (sorted), and dict lookups
cand_by={}; rkrow={}; pxrow={}; maro={}; r1row={}
for k,dt in enumerate(didx):
    r=rk.loc[dt]; top=r[r<=200].sort_values()
    cand_by[k]=list(top.index)
    rkrow[k]=r.to_dict(); pxrow[k]=me.loc[dt].to_dict(); maro[k]=ma10.loc[dt].to_dict()
    if k+1<len(didx): r1row[k]=ret.loc[didx[k+1]].to_dict()
print(f"precompute done t={time.time()-t0:.0f}s",flush=True)
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].reindex(idx); qret=qpx.pct_change()
def stats(r):
    r=r.dropna(); n=len(r); c=(1+r).prod()**(12/n)-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return c,s,dd
def sim(rule):
    pos={}; out=[]; holds=[]; turn=[]
    for k,dt in enumerate(didx):
        px=pxrow[k]; sells=0
        for t in list(pos.keys()):
            e=pos[t]; cpx=px.get(t,np.nan)
            if not np.isfinite(cpx): holds.append(k-e["i"]); pos.pop(t); continue
            if cpx>e["peak"]: e["peak"]=cpx
            ex=False
            if rule=="rotation":   ex=rkrow[k].get(t,9e9)>N
            elif rule=="ride_ma":  ex=cpx<maro[k].get(t,np.nan)
            elif rule=="ride_stop":ex=(cpx/e["peak"]-1)<=-0.25
            if ex: holds.append(k-e["i"]); sells+=1; pos.pop(t)
        if rule!="hold4ever" or k==0:
            for t in cand_by[k]:
                if len(pos)>=N: break
                if t not in pos and np.isfinite(px.get(t,np.nan)): pos[t]={"i":k,"peak":px[t]}
        turn.append(sells/N)
        if k+1<len(didx):
            r1=r1row[k]; pr=np.mean([r1.get(t,-0.5) for t in pos]) if pos else 0.0
            out.append((didx[k+1],pr))
    return pd.Series(dict(out)),np.mean(turn),(np.mean(holds) if holds else np.nan)
print(f"{'long-only rule (top-20, monthly)':34}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'turn/mo':>8}{'avgHold':>9}",flush=True)
for nm,rule in [("monthly rotation (always top-20)","rotation"),("ride winners, sell < 10-mo MA","ride_ma"),
                ("ride winners, sell -25% trailing","ride_stop"),("buy top-20 once, HOLD FOREVER","hold4ever")]:
    s,tn,avh=sim(rule); c,sh,dd=stats(s)
    print(f"  {nm:32}{c*100:>6.1f}%{sh:>7.2f}{dd*100:>6.0f}%{tn*100:>7.0f}%{avh:>9.1f}",flush=True)
c,sh,dd=stats(qret); print(f"  {'QQQ (benchmark)':32}{c*100:>6.1f}%{sh:>7.2f}{dd*100:>6.0f}%",flush=True)
print(f"DONE t={time.time()-t0:.0f}s",flush=True)
