import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
nm=uni.set_index("ticker")
stocks=set(uni[uni.assetType=="Stock"].ticker)
ACC2,HY2,INS2,TECH2=pd.read_pickle("/tmp/wave/_qual_masks.pkl")
cols=[c for c in me.columns if c in stocks]
liq=(me.shift(1)>=3.0).fillna(False)
ENS=((ACC2|HY2)&INS2&TECH2&liq)[cols]
mom6=(me[cols]/me[cols].shift(6)-1)
# revenue YoY for display
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
qend=pd.PeriodIndex([q[2:] for q in rev.index],freq="Q").to_timestamp(how="end").normalize(); rev.index=qend
yoy=rev/rev.shift(4)-1
# insider buyers last 3m for display (recent only = fast)
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P["ym"]=pd.to_datetime(P.ym)
P=P[P.ym>=pd.Timestamp("2024-06-01")]
last=me.index[-1]
p(f"panel ends {last.date()}; data is right-censored near the end (insider/rev feeds lag).")
# list qualifiers in the most recent 3 months
for dt in me.index[-3:]:
    names=[c for c in cols if ENS.loc[dt].get(c,False)]
    p(f"\n=== Qualifiers as of {dt.date()}  ({len(names)} names) ===")
    rows=[]
    for tk in names:
        px=me[tk].loc[dt]
        m6=mom6[tk].loc[dt]
        yq=yoy[tk].dropna() if tk in yoy.columns else pd.Series(dtype=float)
        ry=yq.iloc[-1] if len(yq) else np.nan
        ins=P[(P.tk==tk)&(P.ym<=dt)&(P.ym>dt-pd.DateOffset(months=4))]
        nb=int(ins.nbuyers.sum()); bd=float(ins.buy.sum())
        leg=[]
        if ACC2.loc[dt].get(tk,False): leg.append("rev-accel")
        if HY2.loc[dt].get(tk,False): leg.append("highYoY")
        nmm=str(nm.loc[tk,"ticker"]) if tk in nm.index else tk
        rows.append((tk,px,m6,ry,nb,bd,"+".join(leg)))
    df=pd.DataFrame(rows,columns=["ticker","price","mom6","revYoY","insBuyers3m","insBuy$","legs"]).sort_values("mom6",ascending=False)
    pd.set_option("display.width",160)
    p(df.to_string(index=False,float_format=lambda x:f"{x:,.2f}"))
# save the latest month's screen
dt=me.index[-1]; names=[c for c in cols if ENS.loc[dt].get(c,False)]
out=pd.DataFrame({"ticker":names,
    "price":[me[t].loc[dt] for t in names],
    "mom6":[mom6[t].loc[dt] for t in names],
    "revYoY":[ (yoy[t].dropna().iloc[-1] if (t in yoy.columns and len(yoy[t].dropna())) else np.nan) for t in names]})
out=out.sort_values("mom6",ascending=False)
out.to_csv("/home/user/bonds/dca/research/figures/screener_latest.csv",index=False)
p(f"\nsaved dca/research/figures/screener_latest.csv ({len(out)} names)  t={time.time()-t0:.0f}s")
