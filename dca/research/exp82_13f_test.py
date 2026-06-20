import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
C=pd.read_pickle("/home/user/bonds/dca/research/data/sec/_13f_cusip.pkl")
cmap=pd.read_pickle("/home/user/bonds/dca/research/data/sec/_13f_cusipmap.pkl")
val,nmgr,labs=C["val"],C["nmgr"],C["labs"]
p(f"13F: val{val.shape} mapped cusips {len(cmap)}")
# map cusip->ticker, aggregate to ticker per period
def to_ticker(df):
    df=df.copy(); df["tk"]=[cmap.get(c) for c in df.index]; df=df[df.tk.notna()]
    return df.groupby("tk").sum(numeric_only=True)
V=to_ticker(val); Nm=to_ticker(nmgr)
# period label -> filing-available month (window end + ~45d)
def lab_date(l):
    import re
    mo={'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    m=re.search(r'-(\d{2})([a-z]{3})(\d{4})',l)
    if m:
        d,mm,yy=int(m.group(1)),mo[m.group(2)],int(m.group(3))
        end=pd.Timestamp(yy,mm,d)
    else:
        m=re.search(r'(\d{4})q([1-4])',l); yy=int(m.group(1)); q=int(m.group(2))
        end=pd.Timestamp(yy,q*3,1)+pd.offsets.MonthEnd(0)
    return (end+pd.DateOffset(days=45)).to_period("M").to_timestamp()
dates={l:lab_date(l) for l in labs}
V.columns=[dates[c] for c in V.columns]; Nm.columns=[dates[c] for c in Nm.columns]
V=V.sort_index(axis=1); Nm=Nm.sort_index(axis=1)
# load price panel + align
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
M=me.index; cols=[c for c in me.columns if c in V.index]
p(f"tickers with 13F + price: {len(cols)}")
def monthly(df):
    d=df.T.reindex(M,method="ffill",limit=4)   # ffill quarterly into months, expire after 4mo
    return d.reindex(columns=cols)
Vm=monthly(V); Nmm=monthly(Nm)
# features: institutional $ growth, breadth (manager-count) change
inst_valg=(Vm/Vm.shift(3)-1)            # QoQ value growth
inst_breadth=(Nmm-Nmm.shift(3))         # change in # managers
inst_breadthpct=(Nmm/Nmm.shift(3)-1)
liq=(me[cols].shift(1)>=3.0)
fwd3=(me[cols].shift(-3)/me[cols]-1).clip(-0.9,3.0)
idx=M[(M>=pd.Timestamp(str(min(V.columns).year+1)))&(M<=pd.Timestamp("2025-09-30"))]
def ic(f):
    fr=f.where(liq); ics=[]
    for dt in idx[::1]:
        a=fr.loc[dt]; b=fwd3.loc[dt]; d=pd.concat([a,b],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: ics.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(ics) if ics else np.nan
for nm,f in [("inst_value_growth",inst_valg),("inst_breadth_chg",inst_breadth),("inst_breadth_pct",inst_breadthpct),("log_nmgr(level)",np.log(Nmm.clip(lower=1)))]:
    p(f"  IC(fwd3m) {nm:20} {ic(f):+.3f}")
# L/S decile of breadth change
ret=(me[cols]/me[cols].shift(1)-1).clip(-0.9,2.0); fwd1=ret.shift(-1)
def ls(f,q=0.2):
    fr=f.where(liq); rk=fr.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    sw=(rk<=q).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    r=((lw.shift(1)*ret).sum(axis=1)-(sw.shift(1)*ret).sum(axis=1)).reindex(idx)
    return r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan, r.mean()*12
for nm,f in [("inst_breadth_chg",inst_breadth),("inst_value_growth",inst_valg)]:
    sh,an=ls(f); p(f"  L/S {nm:20} Sharpe {sh:+.2f} ann {an*100:+.1f}%")
p(f"\nwindow {idx[0].date()}..{idx[-1].date()} ({len(idx)} mo) DONE t={time.time()-t0:.0f}s")
