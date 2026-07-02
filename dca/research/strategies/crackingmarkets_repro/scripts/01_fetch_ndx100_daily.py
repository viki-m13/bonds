import urllib.request,json,os,time,pandas as pd
from concurrent.futures import ThreadPoolExecutor,as_completed
TK=os.environ["TIINGO"]
TICKERS="ADBE,AMD,ABNB,ALNY,GOOGL,GOOG,AMZN,AEP,AMGN,ADI,AAPL,AMAT,APP,ARM,ASML,ALAB,ADSK,ADP,AXON,BKR,BKNG,AVGO,CDNS,CTAS,CSCO,CCEP,CMCSA,CEG,CPRT,CRWV,COST,CRWD,CSX,DDOG,DXCM,FANG,DASH,EA,EXC,FAST,FER,FTNT,GEHC,GILD,HON,IDXX,INTC,INTU,ISRG,KDP,KLAC,KHC,LRCX,LIN,LITE,MAR,MRVL,MELI,META,MCHP,MU,MSFT,MSTR,MDLZ,MPWR,MNST,NBIS,NFLX,NVDA,NXPI,ORLY,ODFL,PCAR,PLTR,PANW,PAYX,PYPL,PDD,PEP,QCOM,REGN,RKLB,ROP,ROST,SNDK,STX,SHOP,SBUX,SNPS,TMUS,TTWO,TER,TSLA,TXN,TRI,VRTX,WMT,WBD,WDC,WDAY,XEL".split(",")
def fetch(t):
    for i in range(3):
        try:
            u=f"https://api.tiingo.com/tiingo/daily/{t}/prices?startDate=2000-01-01&token={TK}&format=json"
            d=json.loads(urllib.request.urlopen(u,timeout=45).read())
            if not d: return t,None
            df=pd.DataFrame(d); df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None); df=df.set_index("date")
            out=df[["adjOpen","adjHigh","adjLow","adjClose","adjVolume"]].copy(); out.columns=["o","h","l","c","v"]
            return t,out
        except Exception as e:
            if i==2: return t,f"ERR {repr(e)[:50]}"
            time.sleep(1.5*2**i)
data={}; errs=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    for fut in as_completed({ex.submit(fetch,t):t for t in TICKERS}):
        t,r=fut.result()
        if isinstance(r,pd.DataFrame): data[t]=r
        else: errs.append((t,r))
pd.to_pickle(data,"/tmp/ndx_daily.pkl")
spans=[(t,str(df.index.min())[:7],str(df.index.max())[:7],len(df)) for t,df in list(data.items())[:3]]
print(f"fetched {len(data)}/{len(TICKERS)} tickers; errors: {errs[:6]}")
print("sample spans:",spans)
print("saved /tmp/ndx_daily.pkl")
