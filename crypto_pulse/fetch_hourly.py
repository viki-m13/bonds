import urllib.request, json, time, os, sys
OUT="/home/user/bonds/data/crypto_hourly"
BASE="https://api.binance.us/api/v3/klines"
SYMS=["BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX","LINK","LTC","DOT","MATIC","ATOM","UNI","ETC","BCH","FIL","APT","NEAR","ALGO"]
def fetch(sym):
    pair=sym+"USDT"; rows=[]; start=None
    # find earliest by walking forward from 2021-01-01
    cur=int(time.mktime(time.strptime("2021-06-01","%Y-%m-%d")))*1000
    end=int(time.time()*1000)
    while cur<end:
        url=f"{BASE}?symbol={pair}&interval=1h&limit=1000&startTime={cur}"
        try:
            r=urllib.request.urlopen(url,timeout=20); d=json.loads(r.read())
        except Exception as e:
            print(sym,"err",str(e)[:60]); time.sleep(1.0); 
            if 'HTTP Error 4' in str(e): return rows
            continue
        if not d: break
        rows+=d
        cur=d[-1][0]+3600_000
        if len(d)<1000: break
        time.sleep(0.12)
    return rows
import csv
for s in SYMS:
    rows=fetch(s)
    if not rows: print(s,"NONE"); continue
    p=os.path.join(OUT,f"{s}.csv")
    with open(p,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["ts","open","high","low","close","volume"])
        for r in rows:
            w.writerow([r[0], r[1],r[2],r[3],r[4],r[5]])
    print(s,len(rows),"bars", time.strftime("%Y-%m-%d",time.gmtime(rows[0][0]/1000)),"->",time.strftime("%Y-%m-%d",time.gmtime(rows[-1][0]/1000)))
