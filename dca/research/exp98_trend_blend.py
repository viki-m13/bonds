import urllib.request, ssl, os, io, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
K=os.environ.get("TIINGO_KEY","2897486ab136799678eda8c333ea343811bb0af4")
def p(*a): print(*a,flush=True)
t0=time.time()
def fetch(tk):
    u=f"https://api.tiingo.com/tiingo/daily/{tk}/prices?startDate=2005-01-01&token={K}&format=csv&resampleFreq=daily"
    d=urllib.request.urlopen(urllib.request.Request(u),timeout=60,context=ctx).read().decode()
    df=pd.read_csv(io.StringIO(d)); df["date"]=pd.to_datetime(df.date); return df.set_index("date")["adjClose"].astype(float)
ETFS=["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ","BIL"]
S={}
for t in ETFS:
    try: S[t]=fetch(t);
    except Exception as e: p(f"fail {t} {e}")
px=pd.DataFrame(S).sort_index(); me=px.resample("ME").last(); me.index=me.index.to_period("M").to_timestamp()
p(f"ETFs {list(me.columns)} t={time.time()-t0:.0f}s")
R={t:me[t].pct_change() for t in me.columns}
mom=me/me.shift(12)-1; ma10=me.rolling(10,min_periods=10).mean()
risk=["SPY","QQQ","IWM","EFA","EEM","GLD","DBC","VNQ","TLT","LQD","HYG"]; cash="BIL" if "BIL" in me else "IEF"
idx=me.index[(me.index>=pd.Timestamp("2010-01-01"))&(me.index<=pd.Timestamp("2025-12-31"))]
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
# dual-momentum tactical trend sleeve: top-K trending risk-assets above their 10mo MA, else cash
def trend(K_=4):
    out=[]
    for i,dt in enumerate(me.index):
        if i+1>=len(me.index): continue
        elig=[t for t in risk if me[t].loc[dt]>ma10[t].loc[dt] and mom[t].loc[dt]>0 and np.isfinite(mom[t].loc[dt])]
        elig=sorted(elig,key=lambda t:-mom[t].loc[dt])[:K_]
        nxt=me.index[i+1]
        if elig: r=np.mean([R[t].loc[nxt] for t in elig])
        else: r=R[cash].loc[nxt] if cash in R else 0.0
        out.append((nxt,r))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)
tr=trend(4)
spy=R["SPY"].reindex(idx); qqq=R["QQQ"].reindex(idx)
p(f"\n{'sleeve':28}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'corrSPY':>8}")
for nm,r in [("SPY",spy),("QQQ",qqq),("Trend (dual-mom, long-only)",tr)]:
    c,s,d=stats(r); p(f"{nm:28}{c:>7.1%}{s:>7.2f}{d:>7.1%}{r.corr(spy):>8.2f}")
# WAVE champion returns (recompute from cached ML prob)
try:
    D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,meS=D["FEAT"],D["liq"],D["me"]
    PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(meS.index)
    MS=meS.index; retS=(meS/meS.shift(1)-1).clip(-0.9,3.0); ma10s=meS.rolling(10,min_periods=10).mean(); mom3=meS/meS.shift(3)-1
    a=PROB-PROB.shift(2); el=(liq&(meS>=3.0)&(meS>ma10s)&(mom3>0)&(a>0)).fillna(False).astype(bool)
    sc=PROB.where(el); rk=sc.rank(axis=1,ascending=False); pos={}; cash_=1.0; out=[]
    didx=list(MS)
    for k,dt in enumerate(didx):
        pxr=meS.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=pxr.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx)
            if cpx/e["peak"]-1<=-0.30 or cpx<ma10s.loc[dt].get(tk,np.nan): cash_+=e["val"]; pos.pop(tk)
        if dt in PROB.index:
            rkk=rk.loc[dt]; cands=[t for t in rkk[rkk<=48].sort_values().index if t not in pos and np.isfinite(pxr.get(t,np.nan))]
            need=12-len(pos)
            if need>0 and cash_>1e-9 and cands:
                sl=cash_/need
                for tk in cands[:need]: pos[tk]={"i":k,"px":pxr[tk],"peak":pxr[tk],"val":sl}; cash_-=sl
        eq0=cash_+sum(e["val"] for e in pos.values())
        if k+1<len(didx):
            for tk in pos:
                r1=retS.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash_+sum(e["val"] for e in pos.values())
        if k+1<len(didx): out.append((didx[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    wave=pd.Series(dict(out))
    cidx=idx.intersection(wave.index); wave=wave.reindex(cidx); trC=tr.reindex(cidx); qC=qqq.reindex(cidx)
    c,s,d=stats(wave); p(f"\n{'WAVE champion':28}{c:>7.1%}{s:>7.2f}{d:>7.1%}{wave.corr(qC):>8.2f}")
    p(f"\ncorr(WAVE,Trend)={wave.corr(trC):.2f}  corr(WAVE,QQQ)={wave.corr(qC):.2f}  corr(Trend,QQQ)={trC.corr(qC):.2f}")
    p(f"\n{'PORTFOLIO blend (2015-2025)':30}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}")
    for nm,r in {"100% WAVE":wave,"70 WAVE/30 Trend":0.7*wave+0.3*trC,"60/40":0.6*wave+0.4*trC,
                 "50 WAVE/30 Trend/20 QQQ":0.5*wave+0.3*trC+0.2*qC,"50/50 WAVE/Trend":0.5*wave+0.5*trC}.items():
        c,s,d=stats(r); p(f"{nm:30}{c:>7.1%}{s:>7.2f}{d:>7.1%}")
except Exception as e:
    import traceback; p("WAVE blend err"); traceback.print_exc()
p(f"\nDONE t={time.time()-t0:.0f}s")
