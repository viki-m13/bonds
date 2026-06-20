"""Record HL HIP-3 mark/oracle/funding/premium to test cross-sectional basis
mean-reversion. Polls metaAndAssetCtxs(dex=xyz) every ~5s -> data/hl_hip3/."""
import urllib.request,json,time,os,csv,sys
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data","hl_hip3")
def post(b):
    r=urllib.request.Request('https://api.hyperliquid.xyz/info',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=20).read())
def main():
    secs=int(sys.argv[1]) if len(sys.argv)>1 else 1500
    os.makedirs(OUT,exist_ok=True)
    path=os.path.join(OUT,f"hip3_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    t0=time.time(); n=0
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["ts","coin","mark","oracle","funding","premium","dayvol"])
        while time.time()-t0<secs:
            try: m,ctx=post({'type':'metaAndAssetCtxs','dex':'xyz'})
            except Exception as e: time.sleep(3); continue
            ts=time.time()
            for i,c in enumerate(m['universe']):
                cx=ctx[i]; mk=cx.get('markPx'); orc=cx.get('oraclePx')
                if not mk or not orc: continue
                w.writerow([round(ts,1),c['name'],mk,orc,cx.get('funding'),cx.get('premium'),cx.get('dayNtlVlm')])
            n+=1; f.flush()
            if n%30==0: print(f"{int(time.time()-t0)}s polls={n}",flush=True)
            time.sleep(5)
    print("DONE",path,"polls",n,flush=True)
if __name__=="__main__": main()
