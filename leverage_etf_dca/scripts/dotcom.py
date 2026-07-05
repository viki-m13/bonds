"""VOLT through the DOT-COM BUBBLE (1999-2026). QQQ history goes to 1999-03; reconstruct
TQQQ = 3x daily QQQ - fees; defensive sleeve = CASH (1-mo T-bill from FRED, conservative,
no bond tailwind). Question: does vol-targeting protect the leveraged NASDAQ exposure when
QQQ falls ~83% in 2000-2002? Compares vs QQQ-DCA and vs buy&hold TQQQ (the catastrophe case).
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
REPO="/home/user/bonds"
def load(tk):
    d=pd.read_csv(f"{REPO}/data/etfs/{tk}.csv"); d["Date"]=pd.to_datetime(d["Date"])
    return d.drop_duplicates("Date").set_index("Date")["Close"].sort_index()
qqq=load("QQQ")
# cash: 1-mo T-bill; fill pre-2001 with 1-yr; daily return = annual_yield/252
t1mo=pd.read_csv(f"{REPO}/data/fred/DGS1MO.csv"); t1mo["Date"]=pd.to_datetime(t1mo["Date"]); t1mo=t1mo.set_index("Date")["DGS1MO"]
t1=pd.read_csv(f"{REPO}/data/fred/DGS1.csv"); t1["Date"]=pd.to_datetime(t1["Date"]); t1=t1.set_index("Date")["DGS1"]
cashy=t1mo.reindex(qqq.index).ffill(); cashy=cashy.fillna(t1.reindex(qqq.index).ffill()).fillna(4.0)
cash_dret=(cashy/100.0)/252.0
# reconstruct TQQQ (3x daily) with fees
qret=qqq.pct_change().fillna(0)
exp,borrow,L=0.0095,0.03,3
tqqq_dret=L*qret-(exp+(L-1)*borrow)/252.0
tqqq=(1+tqqq_dret).cumprod()*100
# monthly grid
mg=pd.DatetimeIndex(sorted(qqq.groupby(qqq.index.to_period("M")).apply(lambda x:x.index[-1]).values))
qm=qqq.reindex(mg); tqm=tqqq.reindex(mg)
qmret=qm.pct_change(); tqmret=tqm.pct_change()
cashm=(1+cash_dret).groupby(cash_dret.index.to_period("M")).prod().reindex([p for p in qqq.index.to_period("M").unique()])
cashm.index=mg[:len(cashm)] if len(cashm)==len(mg) else mg
# recompute cash monthly cleanly
cm=(1+cash_dret).resample("ME").prod()-1; cm.index=pd.DatetimeIndex([d for d in cm.index]); 
cashmret=cm.reindex(mg,method="nearest")
# vol-target weight from TQQQ daily vol, at prior month-end
vol=(tqqq_dret.rolling(63,min_periods=40).std()*np.sqrt(252)).reindex(mg,method="ffill")
w=(0.30/vol).clip(0,1).shift(1)
def volt(start,end,target=0.30,cost=0.001):
    ww=(0.30/vol).clip(0,1).shift(1); g=mg[(mg>=start)&(mg<=end)]; rr=[];prevw=0
    for dt in g:
        wt=ww.get(dt,0.0);wt=float(wt) if np.isfinite(wt) else 0.0
        rt=tqmret.get(dt,0.0);rt=rt if np.isfinite(rt) else 0.0
        rc=cashmret.get(dt,0.0);rc=rc if np.isfinite(rc) else 0.0
        rr.append((dt,wt*rt+(1-wt)*rc-abs(wt-prevw)*2*cost));prevw=wt
    return pd.Series(dict(rr))
def dca(r):
    V=0;rows=[]
    for dt,x in r.items():V=V*(1+x)+1000;rows.append((dt,V))
    return pd.Series(dict(rows))
def dcaq(g):
    V=0;rows=[]
    for dt in g:
        x=qmret.get(dt,0);x=x if np.isfinite(x) else 0;V=V*(1+x)+1000;rows.append((dt,V))
    return pd.Series(dict(rows))
def lstat(r):
    r=r.dropna();cum=(1+r).cumprod();cagr=cum.iloc[-1]**(12/len(r))-1;sh=r.mean()/r.std()*np.sqrt(12)
    dd=(cum/cum.cummax()-1);return cagr,sh,dd.min(),cum.iloc[-1]

print("=== VOLT through the dot-com bubble (cash defense, 1999-2026) ===")
print("QQQ history:",qqq.index[0].date(),"->",qqq.index[-1].date())
# 2000-2002 peak-to-trough of the underlyings
qpk=qqq["2000-01":"2000-04"].max(); qtr=qqq["2002-09":"2002-11"].min()
tpk=tqqq["2000-01":"2000-04"].max(); ttr=tqqq["2002-09":"2002-11"].min()
print(f"Dot-com underlying drawdown: QQQ {qtr/qpk-1:.0%},  buy&hold TQQQ(3x recon) {ttr/tpk-1:.1%}")

print("\n--- lump-sum $1 (invest at start, no contributions) ---")
for tag,st,en in [("full 1999-2026","1999-04","2026-06"),("dot-com 2000-2003","2000-01","2003-12"),("2000-2026","2000-01","2026-06")]:
    s,e=pd.Timestamp(st+"-01"),pd.Timestamp(en+"-01");g=mg[(mg>=s)&(mg<=e)]
    c,sh,dd,mult=lstat(volt(s,e))
    qc,qsh,qdd,qmu=lstat(qmret.reindex(g))
    tc,tsh,tdd,tmu=lstat(tqmret.reindex(g))
    print(f"{tag:18} VOLT: CAGR {c*100:5.1f}% Sh {sh:.2f} maxDD {dd*100:6.1f}% ({mult:.1f}x) | QQQ {qc*100:.1f}%/{qdd*100:.0f}% | B&H TQQQ {tc*100:.0f}%/{tdd*100:.0f}% ({tmu:.2f}x)")

print("\n--- monthly DCA $1000, ratio vs QQQ-DCA, era-sliced ---")
ERAS=[("2000-01","2002-12","dot-com bust"),("2000-01","2009-12","dotcom+GFC"),("2003-01","2007-12","recovery"),
      ("2010-01","2019-12","2010s bull"),("2020-01","2026-06","2020s"),("2000-01","2026-06","full 2000-26"),
      ("1999-04","2026-06","full 1999-26")]
for a,b,lab in ERAS:
    s,e=pd.Timestamp(a+"-01"),pd.Timestamp(b+"-01");g=mg[(mg>=s)&(mg<=e)]
    v=dca(volt(s,e)).iloc[-1]; q=dcaq(g).iloc[-1]
    c,sh,dd,_=lstat(volt(s,e))
    print(f"  {lab:14} {a[:4]}-{b[:4]}: VOLT/QQQ-DCA {v/q:5.2f}x   CAGR {c*100:5.1f}%  Sh {sh:.2f}  maxDD {dd*100:6.1f}%")
# what was VOLT's TQQQ weight during the crash?
print("\n--- vol-target weight during dot-com (did it de-risk?) ---")
for dt in pd.date_range("2000-03-01","2002-12-01",freq="6MS"):
    d=mg[mg<=dt][-1] if len(mg[mg<=dt]) else None
    if d is not None:
        print(f"  {d.date()}: TQQQ vol {vol.get(d,np.nan)*100:.0f}%  -> weight {w.get(d,np.nan)*100:.0f}% TQQQ / {100-w.get(d,np.nan)*100:.0f}% cash")
