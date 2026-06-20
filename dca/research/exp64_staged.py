import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
stocks=set(uni[uni.assetType=="Stock"].ticker)
ACC2,HY2,INS2,TECH2=pd.read_pickle("/tmp/wave/_qual_masks.pkl")
cols=[c for c in me.columns if c in stocks]
me=me[cols]
ret=(me/me.shift(1)-1).clip(-0.9,2.0); liq=(me.shift(1)>=3.0).fillna(False)
ma10=me.rolling(10,min_periods=10).mean(); mom6=me/me.shift(6)-1
ENTRY=((ACC2|HY2)&INS2&TECH2&liq)[cols]            # qualifier fires (entry trigger)
trendOK=(me>ma10)                                   # hold while trend intact
idx=me.index[(me.index>=pd.Timestamp("2012-07-01"))&(me.index<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
# ---- event-driven staged-hold simulator ----
def simulate(maxhold=36, cull_m=6, cull_thr=0.10, maxpos=40, trend_exit=True):
    pos={}            # tk -> dict(entry_idx, entry_px)
    monthly=[]; nheld=[]
    didx=list(me.index)
    for k,dt in enumerate(didx):
        if dt<idx[0] or dt>idx[-1]:
            # still need to process holding before window? just skip pre-window
            if dt<idx[0]:
                pass
        px=me.loc[dt]
        # exits (decide at dt using info up to dt)
        for tk in list(pos.keys()):
            e=pos[tk]; age=k-e["i"]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue       # delisted -> exit (loss already booked via ret)
            r_since=cpx/e["px"]-1
            ex=False
            if trend_exit and (cpx<ma10.loc[dt].get(tk,np.nan)): ex=True
            if age>=maxhold: ex=True
            if age>=cull_m and r_since<cull_thr: ex=True
            if ex: pos.pop(tk)
        # entries
        ent=ENTRY.loc[dt]; cands=[tk for tk in ent.index[ent.values] if tk not in pos]
        if cands:
            cands=sorted(cands,key=lambda t:-(mom6.loc[dt].get(t,-9) if np.isfinite(mom6.loc[dt].get(t,np.nan)) else -9))
            for tk in cands:
                if len(pos)>=maxpos: break
                pos[tk]={"i":k,"px":px.get(tk,np.nan)}
        # book next-month return on currently held (equal weight)
        if dt>=idx[0] and dt<=idx[-1]:
            held=list(pos.keys())
            if k+1<len(didx) and held:
                nr=ret.iloc[k+1][held].dropna()
                monthly.append((didx[k+1],nr.mean() if len(nr) else 0.0)); nheld.append(len(held))
            elif held:
                monthly.append((dt,0.0)); nheld.append(len(held))
    s=pd.Series(dict(monthly)).reindex(idx).fillna(0.0)
    return s, np.mean(nheld)
p(f"{'strategy':40} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'avgN':>5}")
c,s,d=stats(qret); p(f"{'QQQ':40} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
configs={
 "monthly rebal (baseline)":dict(maxhold=1,cull_m=99,cull_thr=-9,trend_exit=False,maxpos=9999),
 "hold-while-trend, max36m, cull6m":dict(maxhold=36,cull_m=6,cull_thr=0.10,trend_exit=True,maxpos=40),
 "hold-while-trend, max60m, cull6m":dict(maxhold=60,cull_m=6,cull_thr=0.10,trend_exit=True,maxpos=40),
 "let-run trend-only, max60m":dict(maxhold=60,cull_m=99,cull_thr=-9,trend_exit=True,maxpos=40),
 "hold max36m cull6m, cap25":dict(maxhold=36,cull_m=6,cull_thr=0.10,trend_exit=True,maxpos=25),
}
best=None
for nm,cfg in configs.items():
    r,an=simulate(**cfg); c,s,d=stats(r)
    p(f"{nm:40} {c:>7.1%} {s:>7.2f} {d:>7.1%} {an:>5.0f}")
    if nm.startswith("hold-while-trend, max36"): best=r
# equity curve: best staged vs QQQ
c,s,d=stats(best); qc,qs,qd=stats(qret)
ge=(1+best).cumprod(); gq=(1+qret).cumprod()
fig,ax=plt.subplots(figsize=(11,6))
ax.plot(ge.index,ge,label=f"Staged-hold qualifier (CAGR {c:.0%}, Sh {s:.2f}, DD {d:.0%})",lw=2.3,color="#d62728")
ax.plot(gq.index,gq,label=f"QQQ (CAGR {qc:.0%}, Sh {qs:.2f}, DD {qd:.0%})",lw=2,color="#888")
ax.set_yscale("log"); ax.set_title("Staged multi-year hold qualifier vs QQQ (PIT clean, 2012-2025)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("/tmp/wave/staged_vs_qqq.png",dpi=110)
p(f"\nsaved /tmp/wave/staged_vs_qqq.png")
# ---- DECAY DIAGNOSIS: is it the signal or the small-cap pond? ----
p(f"\n=== DECAY DIAGNOSIS (cross-sectional fwd-12m, by era) ===")
fwd12=(me.shift(-12)/me-1)
def xs(mask,m):
    f=fwd12[cols].where((mask&liq)[cols]).reindex(me.index)[m]
    u=fwd12[cols].where(liq[cols]).reindex(me.index)[m]
    return f.stack().mean(), u.stack().mean(), (f.stack().mean()-u.stack().mean())
for lab,(lo,hi) in [("2012-2016",("2012","2016")),("2017-2020",("2017","2020")),("2021-2025",("2021","2025"))]:
    m=(me.index>=pd.Timestamp(lo))&(me.index<=pd.Timestamp(hi+"-12-31"))
    ens=((ACC2|HY2)&INS2&TECH2);
    fa,ua,da=xs(ACC2,m); fi,ui,di=xs(INS2,m); fe,ue,de=xs(ens,m)
    p(f"  {lab}: rev-accel edge {da:+.1%} | insider edge {di:+.1%} | ENSEMBLE edge {de:+.1%} (vs univ {ua:.1%})")
p(f"\nDONE t={time.time()-t0:.0f}s")
