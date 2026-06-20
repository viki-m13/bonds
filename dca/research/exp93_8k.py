import urllib.request, ssl, io, time, os, re
import pandas as pd, numpy as np
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
UA={"User-Agent":"research viktormashalov@gmail.com"}
def p(*a): print(*a,flush=True)
def get(u,t=120): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=t,context=ctx).read()
t0=time.time()
import json
m=json.loads(get("https://www.sec.gov/files/company_tickers.json"))
cik2tkr={}
for v in m.values():
    c=int(v["cik_str"]);
    if c not in cik2tkr: cik2tkr[c]=v["ticker"]
rows=[]
for y in range(2012,2026):
    for q in (1,2,3,4):
        try:
            b=get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/master.idx",180)
            txt=b.decode("latin-1","ignore")
            for ln in txt.splitlines():
                if "|8-K|" in ln:
                    parts=ln.split("|")
                    if len(parts)>=5:
                        cik=int(parts[0]); date=parts[3]
                        if cik in cik2tkr: rows.append((cik2tkr[cik],date))
            p(f"  {y}Q{q}: cum 8-K rows {len(rows)} t={time.time()-t0:.0f}s")
            time.sleep(0.1)
        except Exception as e: p(f"  {y}Q{q} FAIL {repr(e)[:80]}")
df=pd.DataFrame(rows,columns=["tk","date"]); df["date"]=pd.to_datetime(df.date,errors="coerce")
df["ym"]=df.date.values.astype("datetime64[M]")
cnt=df.groupby(["ym","tk"]).size().unstack(fill_value=0)
cnt.index=pd.DatetimeIndex(cnt.index)
pd.to_pickle(cnt,"/home/user/bonds/dca/research/data/sec/_8k_counts.pkl")
p(f"saved _8k_counts.pkl {cnt.shape} t={time.time()-t0:.0f}s")
# ---- test as a signal ----
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
M=me.index; cols=[c for c in me.columns if c in cnt.columns]
C=cnt.reindex(index=M,columns=cols).fillna(0)
base=C.rolling(12,min_periods=6).mean()
surge=(C+1)/(base+1)                          # 8-K activity surge vs own baseline
liq=(me[cols].shift(1)>=3.0).fillna(False)
fwd1=(me[cols].shift(-1)/me[cols]-1).clip(-0.9,2.0); fwd3=(me[cols].shift(-3)/me[cols]-1).clip(-0.9,3.0)
idx=M[(M>=pd.Timestamp("2013-01-01"))&(M<=pd.Timestamp("2025-09-30"))]
def ic(f,fwd):
    fr=f.where(liq); ics=[]
    for dt in idx:
        d=pd.concat([fr.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: ics.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(ics)
ret=(me[cols]/me[cols].shift(1)-1).clip(-0.9,2.0)
def ls(f,q=0.2):
    fr=f.where(liq); rk=fr.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    sw=(rk<=q).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    s=((lw.shift(1)*ret).sum(axis=1)-(sw.shift(1)*ret).sum(axis=1)).reindex(idx)
    return s.mean()/s.std()*np.sqrt(12) if s.std()>0 else np.nan
p(f"\n8-K signal tests (2013-2025, {len(cols)} tickers):")
for nm,f in [("8K_surge_vs_base",surge),("8K_count_level",C),("8K_count_chg",C-C.shift(1))]:
    p(f"  {nm:20} IC1m {ic(f,fwd1):+.3f}  IC3m {ic(f,fwd3):+.3f}  L/S Sharpe {ls(f):+.2f}")
p(f"DONE t={time.time()-t0:.0f}s")
