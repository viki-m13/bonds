import urllib.request, ssl, os, time, io, sys
import pandas as pd
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
K = os.environ["TIINGO_KEY"]; t0=time.time()
uni = pd.read_pickle("/tmp/wave/_tiingo_universe.pkl"); have=list(dict.fromkeys(uni.ticker))
key = ["QQQ","SPY","IWM","MDY","DIA","TLT","IEF","LQD","HYG","GLD","DBC","EEM","EFA","VNQ",
       "XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLB","XLU","SMH","AAPL","MSFT","NVDA","UNH"]
idx=set()
for f in ("sp500_universe.txt","xuniverse_sp400.txt","xuniverse_ndx.txt"):
    txt=open(f"/tmp/wave/{f}").read(); idx |= set(txt.split()) if " " in txt else set(l.strip() for l in txt.splitlines() if l.strip())
hs=set(have); prio=[t for t in key if t in hs]+[t for t in sorted(idx) if t in hs and t not in key]
prio += [t for t in have if t not in set(prio)]
# skip tickers already downloaded in prior chunks
done_already=set()
os.makedirs("/tmp/wave/tiingo_chunks",exist_ok=True)
import glob
for fp in glob.glob("/tmp/wave/tiingo_chunks/ac_*.parquet"):
    done_already |= set(pd.read_parquet(fp).columns)
prio=[t for t in prio if t not in done_already]
existing=len(glob.glob("/tmp/wave/tiingo_chunks/ac_*.parquet"))
print(f"target {len(prio)} (skipping {len(done_already)} already done); start chunk {existing}",flush=True)
ac={}; vol={}; done=got=0; chunk=existing
for tk in prio:
    try:
        u=f"https://api.tiingo.com/tiingo/daily/{tk}/prices?startDate=2000-01-01&token={K}&format=csv&resampleFreq=daily"
        d=urllib.request.urlopen(urllib.request.Request(u),timeout=40,context=ctx).read().decode()
        df=pd.read_csv(io.StringIO(d))
        if len(df) and "adjClose" in df.columns:
            df["date"]=pd.to_datetime(df.date); df=df.set_index("date")
            ac[tk]=df["adjClose"].astype("float32"); vol[tk]=df["adjVolume"].astype("float32"); got+=1
    except urllib.error.HTTPError as e:
        if e.code==429: print(f"RATELIMIT done={done} got={got}",flush=True); break
    except Exception: pass
    done+=1
    if done%250==0: print(f"  {done} tried, {got} got, t={time.time()-t0:.0f}s",flush=True)
    if len(ac)>=1500:
        pd.DataFrame(ac).to_parquet(f"/tmp/wave/tiingo_chunks/ac_{chunk:02d}.parquet")
        pd.DataFrame(vol).to_parquet(f"/tmp/wave/tiingo_chunks/vol_{chunk:02d}.parquet")
        print(f"  >> chunk {chunk} saved",flush=True); ac={}; vol={}; chunk+=1
if ac:
    pd.DataFrame(ac).to_parquet(f"/tmp/wave/tiingo_chunks/ac_{chunk:02d}.parquet")
    pd.DataFrame(vol).to_parquet(f"/tmp/wave/tiingo_chunks/vol_{chunk:02d}.parquet")
print(f"BATCH DONE got={got}/{done} t={time.time()-t0:.0f}s",flush=True)
