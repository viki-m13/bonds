import urllib.request, json, time, os, csv, datetime as dt
OUT="/home/user/bonds/data/crypto_1min"
COINS=["BTC","ETH","SOL","XRP","DOGE","ADA","AVAX","LINK","LTC","DOT","BCH","UNI","ATOM","ETC","AAVE"]
DAYS=60
def get(prod, s, e, g=60):
    url=f"https://api.exchange.coinbase.com/products/{prod}/candles?granularity={g}&start={s}&end={e}"
    req=urllib.request.Request(url, headers={'User-Agent':'research'})
    return json.loads(urllib.request.urlopen(req,timeout=20).read())
now=int(time.time()); start=now-DAYS*86400
for c in COINS:
    rows={}; cur=start
    while cur<now:
        s=dt.datetime.utcfromtimestamp(cur).isoformat()
        e=dt.datetime.utcfromtimestamp(min(cur+300*60, now)).isoformat()
        try:
            d=get(f"{c}-USD", s, e)
        except Exception as ex:
            if '429' in str(ex): time.sleep(1.0); continue
            time.sleep(0.5); cur+=300*60; continue
        for row in d:  # [time, low, high, open, close, volume]
            rows[row[0]]=row
        cur+=300*60
        time.sleep(0.12)
    rr=[rows[k] for k in sorted(rows)]
    with open(os.path.join(OUT,f"{c}.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["ts","low","high","open","close","volume"]); w.writerows(rr)
    print(c, len(rr), dt.datetime.utcfromtimestamp(rr[0][0]).date() if rr else "-", "->", dt.datetime.utcfromtimestamp(rr[-1][0]).date() if rr else "-", flush=True)
print("DONE", flush=True)
