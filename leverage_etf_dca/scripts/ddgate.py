"""Drawdown-based cash gate on VOLT: sit in cash/defense while QQQ is in a drawdown
beyond a threshold (away from its highs), re-enter on recovery (hysteresis). This is the
'stay in cash during drawdowns / only invest near highs' idea — distinct from a 200MA gate.
Signal read at prior month-end (no look-ahead)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import strategy as S
close=S.close; mgrid=S.mgrid; mret=S.mret

def dd_state(ref="QQQ", exit_dd=0.10, reenter_dd=0.05, use_daily_peak=True):
    """risk-on/off series on mgrid. off when ref drawdown-from-peak <= -exit_dd; back on when
    recovered to within reenter_dd of peak. Hysteresis prevents flip-flop."""
    px = close[ref]
    peak = px.cummax()
    dd = px/peak - 1                                  # daily drawdown
    ddm = dd.reindex(mgrid, method="ffill")           # at each month-end
    on=[]; state=True
    for v in ddm.values:
        if state and v <= -exit_dd: state=False
        elif (not state) and v >= -reenter_dd: state=True
        on.append(state)
    return pd.Series(on, index=mgrid).shift(1).fillna(True)   # act next month (no look-ahead)

def volt_gated(start,end,gate=None,defense=("GLD","TLT"),to_cash=False,partial=0.0,target=0.30,cost=0.001):
    w=S.tqqq_weight(target); g=mgrid[(mgrid>=start)&(mgrid<=end)]; rr=[]; prevw=0.0
    for dt in g:
        wt=w.get(dt,0.0); wt=float(wt) if np.isfinite(wt) else 0.0
        off = gate is not None and not bool(gate.get(dt,True))
        if off: wt=wt*partial
        rt=mret.loc[dt,"TQQQ"]; rt=rt if np.isfinite(rt) else 0.0
        dfa=("BIL",) if (off and to_cash) else defense
        rd=np.nanmean([mret.loc[dt,d] for d in dfa]); rd=rd if np.isfinite(rd) else 0.0
        rr.append((dt,wt*rt+(1-wt)*rd-abs(wt-prevw)*2*cost)); prevw=wt
    return pd.Series(dict(rr))

def stats(r):
    r=r.dropna();cum=(1+r).cumprod();cagr=cum.iloc[-1]**(12/len(r))-1;sh=r.mean()/r.std()*np.sqrt(12)
    d=(cum/cum.cummax()-1);w12=(cum/cum.shift(12)-1).min();return cagr,sh,d.min(),w12
def dfin(r):
    V=0
    for x in r: V=V*(1+x)+1000
    return V
def qfin(g):
    V=0
    for x in mret["QQQ"].reindex(g).fillna(0): V=V*(1+x)+1000
    return V
ERAS=[("2006-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),("2006-01","2026-06")]
Sd,Ed=pd.Timestamp("2006-01-01"),pd.Timestamp("2026-07-01")
print(f"{'config':40} "+" ".join(f"{a[:7]:>7}" for a,b in ERAS)+"   CAGR  Shrp  maxDD wrst12  %inMkt")
def row(nm,gate=None,**kw):
    out=[]
    for a,b in ERAS:
        s,e=pd.Timestamp(a+"-01"),pd.Timestamp(b+"-01");gg=mgrid[(mgrid>=s)&(mgrid<=e)]
        out.append(dfin(volt_gated(s,e,gate=gate,**kw))/qfin(gg))
    c,sh,dd,w12=stats(volt_gated(Sd,Ed,gate=gate,**kw))
    inmkt = 1.0 if gate is None else float(gate.reindex(mgrid[(mgrid>=Sd)&(mgrid<=Ed)]).mean())
    print(f"{nm:40} "+" ".join(f"{v:>7.2f}" for v in out)+f"  {c*100:5.1f}% {sh:4.2f} {dd*100:6.1f}% {w12*100:6.1f}% {inmkt*100:5.0f}%")
row("VOLT base (no gate)")
for xd,rd in [(0.10,0.05),(0.15,0.07),(0.20,0.10)]:
    g=dd_state("QQQ",xd,rd)
    row(f"QQQ dd<-{int(xd*100)}%->defense (reenter -{int(rd*100)}%)",gate=g)
    row(f"QQQ dd<-{int(xd*100)}%->CASH",gate=g,to_cash=True)
# near-highs-only (invested only within 5% of QQQ high)
g5=dd_state("QQQ",0.05,0.02); row("near-highs only (QQQ within 5%)->cash",gate=g5,to_cash=True)
# drawdown gate on TQQQ itself
gt=dd_state("TQQQ",0.30,0.15); row("TQQQ dd<-30%->CASH (reenter -15%)",gate=gt,to_cash=True)
