import urllib.request,json,os,time,pandas as pd,re
from concurrent.futures import ThreadPoolExecutor,as_completed
TK=os.environ["TIINGO"]
raw="KRFT,XLNX,FOXA,FOX,CERN,CHKP,TCOM,TRIP,INCY,LBTYA,LBTYK,EXPE,ULTA,CTXS,BMRN,MTCH,LMCA,LMCK,BATRA,BATRK,SNDK,CHRW,EXPD,GMCR,GRMN,SPLS,VIP,WYNN,LILA,LILAK,VRSN,SWKS,NTES,SPLK,BIDU,DOCU,OKTA,ENPH,JBHT,CDW,CSGP,ATVI,RIVN,VRSK,ZM,ILMN,MRNA,SMCI,DLTR,WBA,SIRI,PTON,TTWO,MXIM,ALXN,FWLT,HOLX,LOGI,PDCO,AKAM,HANS,IACI,PPDI,RYAAY,STLD,JNPR,JAVA,FMCN,AMLN,CDNS,DISCA,LAMR,LEAP,LVLT,PETM,PETM,SIRI,VMED,WFMI,MNST,UAUA,FLIR,TLAB,BEAS,ERIC,PTEN,SEPR,XMSR,CKFR,CDWC,NLTI,BMET,MEDI,AEOS,APCC,CMVT,NCLH,DISCK,TSCO,VIAB,MAT,SBAC,NXPI,BBBY,NTAP,SRCL,WFM,LLTC,ENDP,CTRX,TMUS,TEVA,FSLR,JOYG,CEPH,MICC,GENZ,NIHD,QGEN,URBN,ROST,MCHP,TEAM,ANSS,TRI,MDB,CTSH,INSM,ZS,CHT,AZN,BIIB,GFS,LULU,ON,TTD,ORGN,VSNT"
have=set(pd.read_pickle("/tmp/ndx_daily.pkl").keys())
rem=[t for t in dict.fromkeys(raw.split(",")) if t and t not in have]
print(f"removed/ever-member candidates to fetch: {len(rem)}",flush=True)
def fetch(t):
    for i in range(2):
        try:
            u=f"https://api.tiingo.com/tiingo/daily/{t}/prices?startDate=2000-01-01&token={TK}&format=json"
            d=json.loads(urllib.request.urlopen(u,timeout=40).read())
            if not d: return t,None
            df=pd.DataFrame(d); df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None); df=df.set_index("date")
            o=df[["adjOpen","adjHigh","adjLow","adjClose","adjVolume"]].copy(); o.columns=["o","h","l","c","v"]; return t,o
        except Exception:
            if i==1: return t,None
            time.sleep(1.0)
data={}; ok=0; miss=[]
with ThreadPoolExecutor(max_workers=10) as ex:
    for fut in as_completed({ex.submit(fetch,t):t for t in rem}):
        t,r=fut.result()
        if r is not None and len(r)>0: data[t]=r; ok+=1
        else: miss.append(t)
pd.to_pickle(data,"/tmp/ndx_removed_daily.pkl")
print(f"fetched {ok}/{len(rem)} removed names; missing (no Tiingo data): {miss[:20]}",flush=True)
