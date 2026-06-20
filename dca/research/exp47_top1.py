"""Loop2/exp47 — pick the TOP-1 stock each month, accumulate (never sell), beat
QQQ-DCA honestly. Rules: momentum top-1, insider-conviction top-1, insider+mom,
vs QQQ-DCA and RANDOM top-1 controls (isolate skill from single-stock variance).
Survivorship-caveated (priced universe). Terminal x contributions + sub-period."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
me = pd.read_pickle("/tmp/wave/_ins_px.pkl"); names=[x for x in me.columns if x not in ("SPY","QQQ")]
q = me["QQQ"]; mom6 = me[names]/me[names].shift(6)-1
P = pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P=P[P.tk.isin(names)]
offv = P.pivot_table(index="ym",columns="tk",values="off_buy",aggfunc="sum").reindex(index=me.index,columns=names).fillna(0)
off3 = offv.rolling(3,min_periods=1).sum()                       # trailing-3m officer buy $

def dca(picker, lo, hi, seed=0):
    rng=np.random.default_rng(seed); sh={}; n=0
    for i in range(6,len(me.index)-1):
        d=me.index[i]
        if not (pd.Timestamp(lo)<=d<pd.Timestamp(hi)): continue
        pick=picker(d,rng)
        if pick is None or not np.isfinite(me.at[d,pick]) or me.at[d,pick]<=0: continue
        sh[pick]=sh.get(pick,0.0)+1.0/me.at[d,pick]; n+=1
    last=me.iloc[-1]; term=sum(s*last[t] for t,s in sh.items() if np.isfinite(last[t]))
    return term/max(n,1)

def qqqdca(lo,hi):
    sh=0; n=0
    for i in range(6,len(me.index)-1):
        d=me.index[i]
        if not (pd.Timestamp(lo)<=d<pd.Timestamp(hi)): continue
        sh+=1.0/q[d]; n+=1
    return sh*q.iloc[-1]/max(n,1)

def p_mom(d,rng):
    m=mom6.loc[d].dropna(); return m.idxmax() if len(m) else None
def p_ins(d,rng):
    b=off3.loc[d]; b=b[b>0]; return b.idxmax() if len(b) else p_mom(d,rng)
def p_insmom(d,rng):
    b=off3.loc[d]; cand=list(b[b>0].index)
    if not cand: return p_mom(d,rng)
    m=mom6.loc[d,cand].dropna(); return m.idxmax() if len(m) else None
def p_rand(d,rng):
    av=[t for t in names if np.isfinite(me.at[d,t])]; return rng.choice(av) if av else None
def p_randins(d,rng):
    b=off3.loc[d]; cand=[t for t in b[b>0].index if np.isfinite(me.at[d,t])]; return rng.choice(cand) if cand else None

for tag,lo,hi in (("FULL 2010-25","2010-01-01","2026-12-31"),("TRAIN<2018","2010-01-01","2018-01-01"),("TEST2018+","2018-01-01","2026-12-31")):
    qd=qqqdca(lo,hi)
    rows={}
    for nm,fn in (("momentum top1",p_mom),("insider top1",p_ins),("insider+mom top1",p_insmom)):
        rows[nm]=dca(fn,lo,hi)
    rand=np.mean([dca(p_rand,lo,hi,s) for s in range(15)])
    randins=np.mean([dca(p_randins,lo,hi,s) for s in range(15)])
    print(f"\n{tag}: QQQ-DCA={qd:.2f}x")
    for nm,v in rows.items(): print(f"  {nm:18s} {v:.2f}x  ({v/qd:+.0%} vs QQQ)")
    print(f"  {'RANDOM top1 (avg15)':18s} {rand:.2f}x  ({rand/qd:+.0%})    {'RANDOM among insider':18s} {randins:.2f}x")
print("\nDONE")
