"""Test a SPY risk-off gate on VOLT: when SPY is below its 200d MA (measured at prior
month-end, no look-ahead), force TQQQ weight to 0 and sit in cash/defense — to cut drawdown.
Compare to base VOLT (no gate). Era ratios vs QQQ-DCA, drawdown, Sharpe, phase robustness."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import strategy as S
close=S.close; mgrid=S.mgrid; mret=S.mret
# SPY 200d MA gate (causal, prior month-end)
spy=close["SPY"]; spy_ma=spy.rolling(200,min_periods=150).mean()
spy_up=(spy>spy_ma).reindex(mgrid,method="ffill").shift(1)          # risk-on if SPY>200dMA last month-end
spy_mom=(spy.reindex(mgrid)/spy.reindex(mgrid).shift(6)-1)          # 6m momentum
gate_mom=((spy>spy_ma).reindex(mgrid,method="ffill")&(spy_mom>0)).shift(1)

def strat_gated(start,end,target=0.30,defense=("GLD","TLT"),cost=0.001,gate=None,partial=0.0):
    w=S.tqqq_weight(target); g=mgrid[(mgrid>=start)&(mgrid<=end)]
    rr=[];prevw=0.0
    for dt in g:
        wt=w.get(dt,0.0); wt=float(wt) if np.isfinite(wt) else 0.0
        if gate is not None and not bool(gate.get(dt,True)):
            wt=wt*partial                                          # risk-off: scale TQQQ (0=full cash to defense)
        rt=mret.loc[dt,"TQQQ"]; rt=rt if np.isfinite(rt) else 0.0
        rd=np.nanmean([mret.loc[dt,d] for d in defense]); rd=rd if np.isfinite(rd) else 0.0
        rr.append((dt,wt*rt+(1-wt)*rd-abs(wt-prevw)*2*cost)); prevw=wt
    return pd.Series(dict(rr))

def stats(r):
    r=r.dropna();cum=(1+r).cumprod();cagr=cum.iloc[-1]**(12/len(r))-1;sh=r.mean()/r.std()*np.sqrt(12)
    d=(cum/cum.cummax()-1);w12=(cum/cum.shift(12)-1).min();return cagr,sh,d.min(),w12,cum.iloc[-1]
def dca_final(r):
    V=0
    for x in r: V=V*(1+x)+1000
    return V
def qfinal(g):
    V=0
    for x in mret["QQQ"].reindex(g).fillna(0): V=V*(1+x)+1000
    return V

ERAS=[("2006-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),("2006-01","2026-06")]
Sd,Ed=pd.Timestamp("2006-01-01"),pd.Timestamp("2026-07-01")
print(f"{'config':34} "+" ".join(f"{a[:7]:>7}" for a,b in ERAS)+"   CAGR  Sharpe  maxDD  wrst12")
def row(nm,**kw):
    out=[]
    for a,b in ERAS:
        s,e=pd.Timestamp(a+"-01"),pd.Timestamp(b+"-01")
        gg=mgrid[(mgrid>=s)&(mgrid<=e)]
        out.append(dca_final(strat_gated(s,e,**kw))/qfinal(gg))
    c,sh,dd,w12,_=stats(strat_gated(Sd,Ed,**kw))
    print(f"{nm:34} "+" ".join(f"{v:>7.2f}" for v in out)+f"  {c*100:5.1f}% {sh:5.2f} {dd*100:6.1f}% {w12*100:6.1f}%")

row("VOLT base (no gate)")
row("VOLT + SPY<200MA -> cash",gate=spy_up,partial=0.0)
row("VOLT + SPY<200MA -> half TQQQ",gate=spy_up,partial=0.5)
row("VOLT + SPY<200MA & mom<0 -> cash",gate=gate_mom,partial=0.0)
# risk-off routes to defense blend already (partial=0 => all in GLD-TLT). also test cash(BIL) route:
def strat_cash(start,end,target=0.30,gate=spy_up):
    w=S.tqqq_weight(target);g=mgrid[(mgrid>=start)&(mgrid<=end)];rr=[];prevw=0
    for dt in g:
        wt=w.get(dt,0.0);wt=float(wt) if np.isfinite(wt) else 0.0
        riskoff=gate is not None and not bool(gate.get(dt,True))
        if riskoff: wt=0.0
        rt=mret.loc[dt,"TQQQ"];rt=rt if np.isfinite(rt) else 0
        rd=mret.loc[dt,"BIL"] if riskoff else np.nanmean([mret.loc[dt,d] for d in ("GLD","TLT")])
        rd=rd if np.isfinite(rd) else 0
        rr.append((dt,wt*rt+(1-wt)*rd-abs(wt-prevw)*2*0.001));prevw=wt
    return pd.Series(dict(rr))
out=[]
for a,b in ERAS:
    s,e=pd.Timestamp(a+"-01"),pd.Timestamp(b+"-01");gg=mgrid[(mgrid>=s)&(mgrid<=e)]
    out.append(dca_final(strat_cash(s,e))/qfinal(gg))
c,sh,dd,w12,_=stats(strat_cash(Sd,Ed))
print(f"{'VOLT + SPY<200MA -> BIL cash':34} "+" ".join(f"{v:>7.2f}" for v in out)+f"  {c*100:5.1f}% {sh:5.2f} {dd*100:6.1f}% {w12*100:6.1f}%")

