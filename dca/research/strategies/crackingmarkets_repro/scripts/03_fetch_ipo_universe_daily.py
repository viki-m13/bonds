import urllib.request,json,os,time,pandas as pd
from concurrent.futures import ThreadPoolExecutor,as_completed
TK=os.environ["TIINGO"]
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet").drop_duplicates("ticker")
uni["sd"]=pd.to_datetime(uni["startDate"],errors="coerce")
ipo=uni[(uni["sd"]>=pd.Timestamp("2003-01-01"))&(uni["sd"]<=pd.Timestamp("2025-04-30"))&(uni["assetType"].astype(str).str.contains("Stock",na=False))]
ipo=ipo[~ipo["ticker"].str.contains(r"(?:-|U$|W$|WS$|R$|RT$)",regex=True,na=False)]
sd={r.ticker:r.sd for r in ipo.itertuples()}
existing=pd.read_pickle("/tmp/ipo_daily.pkl")
have=set(existing["data"].keys()); need=[t for t in ipo["ticker"] if t not in have]
print(f"total IPO candidates {len(ipo)}, already have {len(have)}, fetching {len(need)}",flush=True)
def fetch(t):
    for i in range(2):
        try:
            u=f"https://api.tiingo.com/tiingo/daily/{t}/prices?startDate=2002-06-01&token={TK}&format=json"
            d=json.loads(urllib.request.urlopen(u,timeout=40).read())
            if not d: return t,None
            df=pd.DataFrame(d); df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None); df=df.set_index("date")
            o=df[["adjOpen","adjHigh","adjLow","adjClose","adjVolume","close","volume"]].copy()
            o.columns=["o","h","l","c","v","rc","rv"]; return t,o
        except Exception:
            if i==1: return t,None
            time.sleep(1.0)
data=dict(existing["data"]); ok=0
with ThreadPoolExecutor(max_workers=12) as ex:
    for fut in as_completed({ex.submit(fetch,t):t for t in need}):
        t,r=fut.result()
        if r is not None and len(r)>0: data[t]=r; ok+=1
# normalize: earlier sample had different col set (o,h,l,c,v,ro,rc,rv). unify to o,h,l,c,v,rc,rv
for t,df in data.items():
    if "ro" in df.columns: data[t]=df[["o","h","l","c","v","rc","rv"]]
pd.to_pickle({"data":data,"ipo_date":{**existing["ipo_date"],**{t:sd.get(t) for t in data}}},"/tmp/ipo_daily.pkl")
print(f"fetched {ok} new; total IPO universe now {len(data)}",flush=True)
