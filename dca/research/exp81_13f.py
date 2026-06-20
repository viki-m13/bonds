import urllib.request, ssl, io, zipfile, re, time, os
import pandas as pd, numpy as np
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
UA={"User-Agent":"research viktormashalov@gmail.com"}
def p(*a): print(*a,flush=True)
def get(u,t=180): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=t,context=ctx).read()
t0=time.time()
idxpg=get("https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets",60).decode("utf-8","ignore")
links=re.findall(r'href=\"(/files/structureddata/data/form-13f-data-sets/[^\"]*_form13f\.zip)\"',idxpg)
links=list(dict.fromkeys(links))
p(f"found {len(links)} 13F data-set links; newest: {links[:3]}")
use=links[:16]   # ~4 years of quarterly windows
agg={}   # period_label -> Series(cusip -> [value_sum, n_managers])
for ln in use:
    lab=ln.split("/")[-1].replace("_form13f.zip","")
    try:
        b=get("https://www.sec.gov"+ln,300); z=zipfile.ZipFile(io.BytesIO(b))
        nm=[n for n in z.namelist() if "INFOTABLE" in n.upper()][0]
        df=pd.read_csv(z.open(nm),sep="\t",usecols=lambda c:c.upper() in ("CUSIP","VALUE","SSHPRNAMT"),dtype=str)
        df.columns=[c.upper() for c in df.columns]
        df["VALUE"]=pd.to_numeric(df["VALUE"],errors="coerce")
        g=df.groupby("CUSIP").agg(val=("VALUE","sum"),nmgr=("VALUE","size"))
        agg[lab]=g
        p(f"  {lab}: {len(df)} rows -> {len(g)} cusips  t={time.time()-t0:.0f}s")
    except Exception as e:
        p(f"  {lab} FAIL {repr(e)[:120]}")
# build CUSIP x period panels
labs=list(agg.keys())
val=pd.DataFrame({l:agg[l]["val"] for l in labs}); nmgr=pd.DataFrame({l:agg[l]["nmgr"] for l in labs})
val=val.sort_index(axis=1); nmgr=nmgr.sort_index(axis=1)
pd.to_pickle({"val":val,"nmgr":nmgr,"labs":labs},"/home/user/bonds/dca/research/data/sec/_13f_cusip.pkl")
p(f"saved _13f_cusip.pkl val{val.shape} nmgr{nmgr.shape} t={time.time()-t0:.0f}s")
# map top CUSIPs by latest value -> ticker via OpenFIGI
latest=val[labs[-1]].dropna().sort_values(ascending=False)
top=list(latest.head(3500).index)
import json
def figi_batch(cusips):
    req=urllib.request.Request("https://api.openfigi.com/v3/mapping",
        data=json.dumps([{"idType":"ID_CUSIP","idValue":c} for c in cusips]).encode(),
        headers={"Content-Type":"application/json","User-Agent":"r"})
    return json.loads(urllib.request.urlopen(req,timeout=40,context=ctx).read())
cmap={}; i=0
while i<len(top):
    batch=top[i:i+10]
    try:
        res=figi_batch(batch)
        for c,r in zip(batch,res):
            d=r.get("data") if isinstance(r,dict) else None
            if d:
                tk=[x for x in d if x.get("exchCode") in ("US","UW","UN","UQ","UA")]
                cmap[c]=(tk[0]["ticker"] if tk else d[0]["ticker"])
        i+=10; time.sleep(0.25)
    except Exception as e:
        time.sleep(3);
        if "rate" in str(e).lower() or "429" in str(e): time.sleep(20)
        else: i+=10
    if i%500==0: p(f"  mapped {len(cmap)}/{i} t={time.time()-t0:.0f}s")
p(f"CUSIP->ticker mapped: {len(cmap)}")
pd.to_pickle(cmap,"/home/user/bonds/dca/research/data/sec/_13f_cusipmap.pkl")
p(f"saved cusipmap DONE t={time.time()-t0:.0f}s")
