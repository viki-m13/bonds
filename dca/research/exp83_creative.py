import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean()
mom3=me/me.shift(3)-1; vol6=FEAT["vol6"]
# 13F breadth-change monthly (smart money), mapped to tickers
try:
    C=pd.read_pickle("/home/user/bonds/dca/research/data/sec/_13f_cusip.pkl"); cmap=pd.read_pickle("/home/user/bonds/dca/research/data/sec/_13f_cusipmap.pkl")
    nmgr=C["nmgr"]; labs=C["labs"]
    import re
    mo={'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    def ld(l):
        m=re.search(r'-(\d{2})([a-z]{3})(\d{4})',l)
        if m: end=pd.Timestamp(int(m.group(3)),mo[m.group(2)],int(m.group(1)))
        else: mm=re.search(r'(\d{4})q([1-4])',l); end=pd.Timestamp(int(mm.group(1)),int(mm.group(2))*3,1)+pd.offsets.MonthEnd(0)
        return (end+pd.DateOffset(days=45)).to_period("M").to_timestamp()
    nm=nmgr.copy(); nm["tk"]=[cmap.get(c) for c in nm.index]; nm=nm[nm.tk.notna()].groupby("tk").sum(numeric_only=True)
    nm.columns=[ld(c) for c in nm.columns]; nm=nm.sort_index(axis=1)
    bmonth=nm.T.reindex(M,method="ffill",limit=4).reindex(columns=cols)
    bm_chg=(bmonth-bmonth.shift(3))
except Exception as e:
    p(f"13F load fail {e}"); bm_chg=pd.DataFrame(0.0,index=M,columns=cols)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]; qret=qpx.pct_change().reindex(idx)
elig=(liq&(me>=3.0)&(me>ma10))
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def rz(df): return df.where(elig).rank(axis=1,pct=True)
# super-score = ML + smart-money + signal acceleration
accel=PROB-PROB.shift(2)
SUPER = rz(PROB) + 0.25*rz(FEAT["ins_clustern"]) + 0.15*rz(bm_chg.fillna(0)) + 0.20*rz(accel)
def sim(score, N=12, conviction=False, volstop=False, trail=-0.30, accel_gate=False, runnergate=True):
    sc=score.where(elig & ((mom3>0) if runnergate else True) & ((accel>0) if accel_gate else True))
    rank=sc.rank(axis=1,ascending=False)
    pos={}; cash=1.0; out=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx)
            tr = -(2.2*vol6.loc[dt].get(tk,0.12)*np.sqrt(21)) if volstop else trail   # ~2.2 monthly-sigma stop
            tr=max(tr,-0.5)
            if cpx/e["peak"]-1<=tr or cpx<ma10.loc[dt].get(tk,np.nan): cash+=e["val"]; pos.pop(tk)
        if dt in PROB.index:
            rk=rank.loc[dt]; sv=sc.loc[dt]
            cands=[t for t in rk[rk<=N*3].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan))]
            need=N-len(pos)
            if need>0 and cash>1e-9 and cands:
                pick=cands[:need]
                if conviction:
                    sc_v=np.array([max(sv.get(t,0),1e-6) for t in pick]); w=sc_v/sc_v.sum()
                else: w=np.ones(len(pick))/len(pick)
                for tk,wi in zip(pick,w): a=cash*wi if False else (cash/need); a=cash*wi; pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":a}; cash-=a
        eq0=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx):
            for tk in pos:
                r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx): out.append((didx[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)
p(f"{'creative long-only (2015-2025)':46} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
c,s,d=stats(qret); p(f"{'QQQ':46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
cfgs={
 "base ML N12 runner-gate":dict(score=PROB),
 "+ conviction weighting":dict(score=PROB,conviction=True),
 "+ vol-calibrated stop":dict(score=PROB,volstop=True),
 "+ signal-acceleration gate":dict(score=PROB,accel_gate=True),
 "SUPER-score (ML+insider+13F+accel)":dict(score=SUPER),
 "SUPER + conviction + volstop":dict(score=SUPER,conviction=True,volstop=True),
 "SUPER + conviction + volstop N10":dict(score=SUPER,conviction=True,volstop=True,N=10),
 "SUPER + conviction N8":dict(score=SUPER,conviction=True,N=8),
}
res={}
for nm,cfg in cfgs.items():
    r=sim(**cfg); res[nm]=r; c,s,d=stats(r); p(f"{nm:46} {c:>7.1%} {s:>7.2f} {d:>7.1%}")
best=max(res,key=lambda k: stats(res[k])[1]); br=res[best]
p(f"\nbest Sharpe: {best}")
for lo,hi in [("2015","2018"),("2019","2021"),("2022","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(br[m]); qc,qs,_=stats(qret[m]); p(f"  {lo}-{hi}: {c:+.1%}/{s:.2f} vs QQQ {qc:+.1%}/{qs:.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
