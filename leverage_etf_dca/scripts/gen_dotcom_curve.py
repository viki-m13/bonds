"""Generate the dot-com stress equity curve: monthly DCA $1000, VOLT (vol-targeted TQQQ +
accel overlay, CASH defense — matches the dot-com section) vs QQQ-DCA, 1999-2006. Output
compact JSON to ../dotcom_curve.json for embedding in the factsheet."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
REPO="/home/user/bonds"
def load(tk):
    d=pd.read_csv(f"{REPO}/data/etfs/{tk}.csv"); d["Date"]=pd.to_datetime(d["Date"])
    return d.drop_duplicates("Date").set_index("Date")["Close"].sort_index()
qqq=load("QQQ")
t1mo=pd.read_csv(f"{REPO}/data/fred/DGS1MO.csv"); t1mo["Date"]=pd.to_datetime(t1mo["Date"]); t1mo=t1mo.set_index("Date")["DGS1MO"]
t1=pd.read_csv(f"{REPO}/data/fred/DGS1.csv"); t1["Date"]=pd.to_datetime(t1["Date"]); t1=t1.set_index("Date")["DGS1"]
cashy=t1mo.reindex(qqq.index).ffill(); cashy=cashy.fillna(t1.reindex(qqq.index).ffill()).fillna(4.0)
cash_dret=(cashy/100)/252
qret=qqq.pct_change().fillna(0)
tqqq_dret=3*qret-(0.0095+2*0.03)/252
tqqq=(1+tqqq_dret).cumprod()*100
mg=pd.DatetimeIndex(sorted(qqq.groupby(qqq.index.to_period("M")).apply(lambda x:x.index[-1]).values))
qmret=qqq.reindex(mg).pct_change(); tqmret=tqqq.reindex(mg).pct_change()
cm=(1+cash_dret).resample("ME").prod()-1; cashm=cm.reindex(mg,method="nearest")
# vol-target weight WITH acceleration overlay (matches v2)
def av(w): return (tqqq_dret.rolling(w,min_periods=int(w*.7)).std()*np.sqrt(252)).reindex(mg,method="ffill")
v63,v20=av(63),av(20)
veff=v63*(v20/v63).clip(lower=1.0)**2.0
w=(0.30/veff).clip(0,1).shift(1)
S,E=pd.Timestamp("1999-03-01"),pd.Timestamp("2006-12-31")
g=mg[(mg>=S)&(mg<=E)]
Vv=0;Vq=0;prevw=0;rows=[]
peakv=peakq=0
for dt in g:
    wt=w.get(dt,0);wt=float(wt) if np.isfinite(wt) else 0
    rt=tqmret.get(dt,0);rt=rt if np.isfinite(rt) else 0
    rc=cashm.get(dt,0);rc=rc if np.isfinite(rc) else 0
    rq=qmret.get(dt,0);rq=rq if np.isfinite(rq) else 0
    rvolt=wt*rt+(1-wt)*rc-abs(wt-prevw)*2*0.001; prevw=wt
    Vv=Vv*(1+rvolt)+1000; Vq=Vq*(1+rq)+1000
    rows.append({"d":dt.strftime("%Y-%m"),"v":round(Vv,0),"q":round(Vq,0),"w":round(wt,3)})
out={"rows":rows,"final_v":round(Vv),"final_q":round(Vq),"ratio":round(Vv/Vq,2)}
json.dump(out,open("../dotcom_curve.json","w"),separators=(",",":"))
print("dotcom curve",len(rows),"months, final VOLT",f"${Vv:,.0f}","QQQ-DCA",f"${Vq:,.0f}","ratio",round(Vv/Vq,2))
# quick sanity: values at the 2002 bottom
b=[r for r in rows if r["d"]=="2002-09"][0]; print("at 2002-09 bottom: VOLT",f"${b['v']:,.0f}","vs QQQ-DCA",f"${b['q']:,.0f}","w",b["w"])
