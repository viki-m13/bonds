import urllib.request, ssl, json, time, os
import pandas as pd, numpy as np
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
UA={"User-Agent":"research viktormashalov@gmail.com"}
def p(*a): print(*a,flush=True)
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=50,context=ctx).read()
OUT="/home/user/bonds/dca/research/data/sec"; os.makedirs(OUT,exist_ok=True)
t0=time.time()
# cik -> ticker (first/primary ticker per cik)
m=json.loads(get("https://www.sec.gov/files/company_tickers.json"))
cik2tkr={}
for v in m.values():
    c=int(v["cik_str"])
    if c not in cik2tkr: cik2tkr[c]=v["ticker"]
p(f"cik->ticker: {len(cik2tkr)} t={time.time()-t0:.0f}s")
CONCEPTS=["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues",
          "RevenueFromContractWithCustomerIncludingAssessedTax","SalesRevenueNet"]
quarters=[f"CY{y}Q{q}" for y in range(2011,2026) for q in (1,2,3,4)]
# collect: dict[(cik,quarter)] = revenue, preferring earlier concept in CONCEPTS list
rev={}
for ci,concept in enumerate(CONCEPTS):
    got=0
    for qd in quarters:
        u=f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{qd}.json"
        try:
            fr=json.loads(get(u))
        except Exception:
            continue
        for d in fr.get("data",[]):
            key=(d["cik"],qd)
            if key not in rev:                  # prefer higher-priority concept already stored
                rev[key]=d["val"]; got+=1
        time.sleep(0.08)
    p(f"concept {concept}: +{got} new pts (total {len(rev)}) t={time.time()-t0:.0f}s")
# build cik x quarter panel
ciks=sorted(set(c for c,_ in rev));
qidx=quarters
df=pd.DataFrame(index=qidx,columns=ciks,dtype="float64")
for (c,qd),val in rev.items():
    df.at[qd,c]=val
df.index.name="quarter"
# map to tickers (drop ciks with no ticker)
keep=[c for c in df.columns if c in cik2tkr]
df=df[keep]; df.columns=[cik2tkr[c] for c in keep]
df=df.loc[:,~df.columns.duplicated()]
df.to_parquet(f"{OUT}/sec_revenue_quarterly.parquet")
p(f"revenue panel {df.shape} (quarters x tickers) saved t={time.time()-t0:.0f}s")
p(f"NVDA tail:\n{df['NVDA'].dropna().tail(6) if 'NVDA' in df else 'no NVDA'}")
