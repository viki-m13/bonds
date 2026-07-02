import urllib.request,json,os,time,pandas as pd
from concurrent.futures import ThreadPoolExecutor,as_completed
TK=os.environ["TIINGO"]
SP="MMM,AOS,ABT,ABBV,ACN,ADBE,AMD,AES,AFL,A,APD,ABNB,AKAM,ALB,ARE,ALGN,ALLE,LNT,ALL,GOOGL,GOOG,MO,AMZN,AMCR,AEE,AEP,AXP,AIG,AMT,AWK,AMP,AME,AMGN,APH,ADI,AON,APA,APO,AAPL,AMAT,APP,APTV,ACGL,ADM,ARES,ANET,AJG,AIZ,T,ATO,ADSK,ADP,AZO,AVB,AVY,AXON,BKR,BALL,BAC,BAX,BDX,BRK-B,BBY,TECH,BIIB,BLK,BX,XYZ,BNY,BA,BKNG,BSX,BMY,AVGO,BR,BRO,BF-B,BLDR,BG,BXP,CHRW,CDNS,CPT,COF,CAH,CCL,CARR,CVNA,CASY,CAT,CBOE,CBRE,CDW,COR,CNC,CNP,CF,CRL,SCHW,CHTR,CVX,CMG,CB,CHD,CIEN,CI,CINF,CTAS,CSCO,C,CFG,CLX,CME,CMS,KO,CTSH,COHR,COIN,CL,CMCSA,FIX,COP,ED,STZ,CEG,COO,CPRT,GLW,CPAY,CTVA,CSGP,COST,CRH,CRWD,CCI,CSX,CMI,CVS,DHR,DRI,DDOG,DVA,DECK,DE,DELL,DAL,DVN,DXCM,FANG,DLR,DG,DLTR,D,DPZ,DASH,DOV,DOW,DHI,DTE,DUK,DD,ETN,EBAY,ECL,EIX,EW,EA,ELV,EME,EMR,ETR,EOG,EQT,EFX,EQIX,EQR,ERIE,ESS,EL,EG,EVRG,ES,EXC,EXPE,EXPD,EXR,XOM,FFIV,FDS,FICO,FAST,FRT,FDX,FIS,FITB,FSLR,FE,FLEX,F,FTNT,FTV,FOXA,FOX,BEN,FCX,GRMN,IT,GE,GEHC,GEV,GEN,GNRC,GD,GIS,GM,GPC,GILD,GPN,GL,GDDY,GS,HAL,HIG,HAS,HCA,DOC,HSIC,HSY,HPE,HLT,HD,HON,HRL,HST,HWM,HPQ,HUBB,HUM,HBAN,HII,IBM,IEX,IDXX,ITW,INCY,IR,PODD,INTC,IBKR,ICE,IFF,IP,INTU,ISRG,IVZ,INVH,IQV,IRM,JBHT,JBL,JKHY,J,JNJ,JCI,JPM,KVUE,KDP,KEY,KEYS,KMB,KIM,KMI,KKR,KLAC,KHC,KR,LHX,LH,LRCX,LVS,LDOS,LEN,LII,LLY,LIN,LYV,LMT,L,LOW,LULU,LITE,LYB,MTB,MPC,MAR,MLM,MRVL,MAS,MA,MKC,MCD,MCK,MDT,MRK,META,MET,MTD,MGM,MCHP,MU,MSFT,MAA,MRNA,TAP,MDLZ,MPWR,MNST,MCO,MS,MOS,MSI,MSCI,NDAQ,NTAP,NFLX,NEM,NWSA,NWS,NEE,NKE,NI,NDSN,NSC,NTRS,NOC,NCLH,NRG,NUE,NVDA,NVR,NXPI,ORLY,OXY,ODFL,OMC,ON,OKE,ORCL,OTIS,PCAR,PKG,PLTR,PANW,PH,PAYX,PYPL,PNR,PEP,PFE,PCG,PM,PSX,PNW,PNC".split(",")
have=set(pd.read_pickle("/tmp/ndx_daily.pkl").keys())
need=[t for t in dict.fromkeys(SP) if t not in have]
def fetch(t):
    for i in range(3):
        try:
            u=f"https://api.tiingo.com/tiingo/daily/{t}/prices?startDate=2000-01-01&token={TK}&format=json"
            d=json.loads(urllib.request.urlopen(u,timeout=45).read())
            if not d: return t,None
            df=pd.DataFrame(d); df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None); df=df.set_index("date")
            o=df[["adjOpen","adjHigh","adjLow","adjClose","adjVolume"]].copy(); o.columns=["o","h","l","c","v"]; return t,o
        except Exception as e:
            if i==2: return t,None
            time.sleep(1.2*2**i)
data=dict(pd.read_pickle("/tmp/ndx_daily.pkl")); nnew=0
with ThreadPoolExecutor(max_workers=8) as ex:
    for fut in as_completed({ex.submit(fetch,t):t for t in need}):
        t,r=fut.result()
        if r is not None: data[t]=r; nnew+=1
pd.to_pickle(data,"/tmp/sp_daily.pkl")
print(f"sp universe: {len(data)} tickers total ({nnew} new fetched, {len(need)} needed)")
