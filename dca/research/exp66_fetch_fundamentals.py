import urllib.request, ssl, json, time, os
import pandas as pd, numpy as np
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
UA={"User-Agent":"research viktormashalov@gmail.com"}
def p(*a): print(*a,flush=True)
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60,context=ctx).read()
OUT="/home/user/bonds/dca/research/data/sec"; os.makedirs(OUT,exist_ok=True)
t0=time.time()
m=json.loads(get("https://www.sec.gov/files/company_tickers.json"))
cik2tkr={}
for v in m.values():
    c=int(v["cik_str"]);
    if c not in cik2tkr: cik2tkr[c]=v["ticker"]
quarters=[f"CY{y}Q{q}" for y in range(2011,2026) for q in (1,2,3,4)]
# flow concepts (duration, quarterly frame) ; stock concepts (instant, append 'I')
FLOW=["GrossProfit","OperatingIncomeLoss","NetIncomeLoss","ResearchAndDevelopmentExpense"]
STOCK=["Assets","StockholdersEquity","CashAndCashEquivalentsAtCarryingValue",
       "ShortTermInvestments","LongTermDebt"]
SHARES=["EntityCommonStockSharesOutstanding"]  # dei taxonomy, instant
panels={}
def fetch_concept(tax,concept,instant):
    vals={}
    got=0
    for qd in quarters:
        frame=qd+("I" if instant else "")
        u=f"https://data.sec.gov/api/xbrl/frames/{tax}/{concept}/{'shares' if concept.endswith('SharesOutstanding') else 'USD'}/{frame}.json"
        try: fr=json.loads(get(u))
        except Exception: continue
        for d in fr.get("data",[]):
            vals[(d["cik"],qd)]=d["val"]; got+=1
        time.sleep(0.07)
    return vals,got
for concept in FLOW:
    vals,got=fetch_concept("us-gaap",concept,False); panels[concept]=vals
    p(f"{concept}: {got} pts t={time.time()-t0:.0f}s")
for concept in STOCK:
    vals,got=fetch_concept("us-gaap",concept,True); panels[concept]=vals
    p(f"{concept}: {got} pts t={time.time()-t0:.0f}s")
for concept in SHARES:
    vals,got=fetch_concept("dei",concept,True); panels[concept]=vals
    p(f"{concept}: {got} pts t={time.time()-t0:.0f}s")
# build {concept: DataFrame(quarter x ticker)}
out={}
for concept,vals in panels.items():
    ciks=sorted(set(c for c,_ in vals))
    df=pd.DataFrame(index=quarters,columns=ciks,dtype="float64")
    for (c,qd),v in vals.items(): df.at[qd,c]=v
    keep=[c for c in df.columns if c in cik2tkr]; df=df[keep]; df.columns=[cik2tkr[c] for c in keep]
    df=df.loc[:,~df.columns.duplicated()]
    out[concept]=df
pd.to_pickle(out,f"{OUT}/sec_fundamentals.pkl")
p(f"saved sec_fundamentals.pkl: {list(out.keys())} t={time.time()-t0:.0f}s")
for k,v in out.items(): p(f"  {k}: {v.shape}")
