import urllib.request, json, time, os, csv, datetime as dt
OUT="/home/user/bonds/data/crypto_hourly_cb"
COINS=["BTC","ETH","SOL","XRP","DOGE","ADA","AVAX","LINK","LTC","DOT","BCH","UNI",
       "ATOM","ETC","AAVE","NEAR","APT","ARB","OP","INJ","SUI","FIL","STX","LDO","MKR","RENDER","IMX","SEI"]
DAYS=730
def get(prod,s,e,g=3600):
    url=f"https://api.exchange.coinbase.com/products/{prod}/candles?granularity={g}&start={s}&end={e}"
    req=urllib.request.Request(url,headers={'User-Agent':'research'})
    return json.loads(urllib.request.urlopen(req,timeout=25).read())
now=int(time.time()); start=now-DAYS*86400
for c in COINS:
    rows={}; cur=start
    while cur<now:
        s=dt.datetime.utcfromtimestamp(cur).isoformat()
        e=dt.datetime.utcfromtimestamp(min(cur+300*3600,now)).isoformat()
        try: d=get(f"{c}-USD",s,e)
        except Exception as ex:
            if '429' in str(ex): time.sleep(1.0); continue
            time.sleep(0.4); cur+=300*3600; continue
        for r in d: rows[r[0]]=r
        cur+=300*3600; time.sleep(0.13)
    rr=[rows[k] for k in sorted(rows)]
    if rr:
        with open(os.path.join(OUT,f"{c}.csv"),"w",newline="") as f:
            w=csv.writer(f); w.writerow(["ts","low","high","open","close","volume"]); w.writerows(rr)
        print(c,len(rr),dt.datetime.utcfromtimestamp(rr[0][0]).date(),"->",dt.datetime.utcfromtimestamp(rr[-1][0]).date(),flush=True)
    else: print(c,"NONE",flush=True)
print("DONE",flush=True)
