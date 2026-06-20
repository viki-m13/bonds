import urllib.request, ssl, io, zipfile, json, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
UA={"User-Agent":"research viktormashalov@gmail.com"}
def p(*a): print(*a,flush=True)
def get(u,t=120): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=t,context=ctx).read()
t0=time.time()
# cik->ticker
m=json.loads(get("https://www.sec.gov/files/company_tickers.json")); cik2tkr={}
for v in m.values():
    c=int(v["cik_str"]);
    if c not in cik2tkr: cik2tkr[c]=v["ticker"]
# cik->sic from financial-statement-data-sets sub.txt
cik2sic={}
for q in ["2024q3","2023q3","2022q3","2021q3"]:
    try:
        b=get(f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip",180)
        z=zipfile.ZipFile(io.BytesIO(b)); sub=pd.read_csv(z.open("sub.txt"),sep="\t",usecols=["cik","sic"],dtype={"cik":int,"sic":"float"})
        for c,s in zip(sub.cik,sub.sic):
            if c not in cik2sic and np.isfinite(s): cik2sic[c]=int(s)
        p(f"  {q}: cum sic {len(cik2sic)} t={time.time()-t0:.0f}s")
    except Exception as e: p(f"  {q} fail {repr(e)[:80]}")
tkr2sec={cik2tkr[c]:(s//100) for c,s in cik2sic.items() if c in cik2tkr}   # SIC 2-digit major group
p(f"tickers with sector: {len(tkr2sec)} t={time.time()-t0:.0f}s")
# ---- factors (durable survivors) ----
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def qidx(df):
    df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=qidx(rev)
def gq(k):
    d=F.get(k); return qidx(d).reindex(columns=rev.columns) if d is not None else None
GP,NI,AST,EQ,SH,CASH,STI=[gq(k) for k in ["GrossProfit","NetIncomeLoss","Assets","StockholdersEquity","EntityCommonStockSharesOutstanding","CashAndCashEquivalentsAtCarryingValue","ShortTermInvestments"]]
def qmap(df):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=6)
mcap=me*qmap(SH)
nca=AST-(CASH.fillna(0)+STI.fillna(0)); accr=-qmap(nca.diff()/AST.shift(1))   # low accruals good
FACT={
 "momentum": me.shift(1)/me.shift(13)-1,
 "value_BM": qmap(EQ)/mcap, "value_EP": qmap(NI*4)/mcap,
 "qual_ROA": FEAT["roa"], "qual_GPA": qmap(GP*4)/qmap(AST), "qual_ROE": FEAT["roe"],
 "lowvol": -FEAT["vol6"], "buyback": -FEAT["share_chg"], "accruals": accr,
}
def z(df):
    fr=df.where(liq); return fr.sub(fr.mean(axis=1),axis=0).div(fr.std(axis=1).replace(0,np.nan),axis=0)
comp=sum(z(f).fillna(0) for f in FACT.values())/len(FACT)
comp=comp.where(liq)
# sector-neutralize: within each sector each month, demean the composite
sec=pd.Series({c:tkr2sec.get(c,-1) for c in cols})
secvals=sec.values
def sector_neutralize(X):
    Xn=X.copy().values; arr=X.values
    for gi in np.unique(secvals):
        if gi<0: continue
        idxs=np.where(secvals==gi)[0]
        if len(idxs)<5: continue
        sub=arr[:,idxs]; mu=np.nanmean(sub,axis=1,keepdims=True)
        Xn[:,idxs]=sub-mu
    return pd.DataFrame(Xn,index=X.index,columns=X.columns)
compN=sector_neutralize(comp)
ret=(me/me.shift(1)-1).clip(-0.9,2.0); fwd1=ret.shift(-1)
idx=M[(M>=pd.Timestamp("2012-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
LIQ=(me.shift(1)>=3.0).fillna(False)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def ic(sig):
    fr=sig.where(LIQ); fwd3=(me.shift(-3)/me-1).clip(-0.9,3.0); v=[]
    for dt in idx:
        d=pd.concat([fr.loc[dt],fwd3.loc[dt]],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: v.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(v)
def ls(sig,q=0.1):
    fr=sig.where(LIQ); rk=fr.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    sw=(rk<=q).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    lo=(lw.shift(1)*ret).sum(axis=1).reindex(idx); l_s=((lw.shift(1)-sw.shift(1))*ret).sum(axis=1).reindex(idx) if False else ((lw.shift(1)*ret).sum(axis=1)-(sw.shift(1)*ret).sum(axis=1)).reindex(idx)
    return lo,l_s
p(f"\n{'composite':24}{'IC3m':>7}{'longCAGR':>9}{'longSh':>7}{'L/S Sh':>7}{'corrQQQ':>8}")
for nm,sig in [("raw multi-factor",comp),("SECTOR-NEUTRAL",compN)]:
    lo,l_s=ls(sig); c,s,d=stats(lo); _,sl,_=stats(l_s)
    p(f"{nm:24}{ic(sig):>7.3f}{c:>9.1%}{s:>7.2f}{sl:>7.2f}{l_s.corr(qret):>8.2f}")
# long-only top decile sector-neutral vs QQQ + sub-periods
lo,l_s=ls(compN);
c,s,d=stats(lo); p(f"\nSector-neutral long-decile: CAGR {c:.1%} Sharpe {s:.2f} maxDD {d:.1%} | QQQ {stats(qret)[0]:.1%}/{stats(qret)[1]:.2f}")
for loo,hi in [("2012","2016"),("2017","2020"),("2021","2025")]:
    mk=(idx>=pd.Timestamp(loo))&(idx<=pd.Timestamp(hi+"-12-31")); c,s,_=stats(lo[mk]); p(f"  {loo}-{hi}: {c:+.1%}/{s:.2f}")
pd.to_pickle({"comp":comp,"compN":compN,"tkr2sec":tkr2sec},"/tmp/wave/_composite.pkl")
p(f"\nDONE t={time.time()-t0:.0f}s")
