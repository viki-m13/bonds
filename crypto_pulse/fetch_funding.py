import urllib.request, json, time, os, csv, glob
OV=['AAVE','ADA','ALGO','APT','AR','ARB','ARK','ATOM','AVAX','AXS','BCH','BNB','BNT','BTC','COMP','CRV','DASH','DOGE','DOT','DYDX','ETC','ETH','FIL','FTM','FTT','GALA','HBAR','ILV','INJ','IOTA','JUP','KAS','LINK','LTC','MATIC','MKR','NEAR','NEO','OP','PYTH','RNDR','RUNE','SAND','SEI','SNX','SOL','STX','SUI','SUSHI','TIA','TRX','UNI','USTC','XLM','XMR','XRP','ZEC']
OUT="/home/user/bonds/data/hl_funding"
def post(b):
    r=urllib.request.Request('https://api.hyperliquid.xyz/info',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=25).read())
for c in OV:
    rows=[]; start=0
    while True:
        try: fh=post({'type':'fundingHistory','coin':c,'startTime':start})
        except Exception as e:
            time.sleep(1.0); 
            if '429' in str(e): time.sleep(10); continue
            break
        if not fh: break
        rows+=[(x['time'],x['fundingRate']) for x in fh]
        if len(fh)<500: break
        start=fh[-1]['time']+3600_000
        time.sleep(0.15)
    if rows:
        seen=set(); ded=[]
        for t,r in rows:
            if t in seen: continue
            seen.add(t); ded.append((t,r))
        with open(os.path.join(OUT,f"{c}.csv"),"w",newline="") as f:
            w=csv.writer(f); w.writerow(["ts","funding"]); w.writerows(ded)
        print(c,len(ded),time.strftime("%Y-%m-%d",time.gmtime(ded[0][0]/1000)),"->",time.strftime("%Y-%m-%d",time.gmtime(ded[-1][0]/1000)),flush=True)
    else:
        print(c,"NONE",flush=True)
print("DONE",flush=True)
