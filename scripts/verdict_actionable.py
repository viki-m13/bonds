"""Actionable-layer evidence: appends to /tmp/verdict_evidence.json.
- lottery: 2005->2026 (21y) single-stock outcome distribution incl. P(10x), P(100x)
- blends: QQQ/bond DCA mixes 2006-2026 (final wealth, maxDD on the account)
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = f"{ROOT}/dca/research/strategies/ascent/scripts"
def _load(name):
    for p in (os.path.join(os.environ.get("ASCENT_WORK","/tmp/ascent_work"),name), f"{A}/{name}"):
        if os.path.exists(p): return pd.read_pickle(p)
    raise FileNotFoundError(name)
ME=_load("_me_monthly.pkl"); DV=_load("_dv_monthly.pkl")
uni=pd.read_parquet(f"{ROOT}/dca/research/data/tiingo/tiingo_universe_pit.parquet")
_st=uni[uni.assetType=="Stock"].ticker
_st=_st[~_st.str.contains("-",na=False) & ~_st.str.match(r".*(?:U|W|WS|R|RT)$",na=False)]
stocks=set(_st)&set(ME.columns)
S=ME[[c for c in ME.columns if c in stocks]]
liq=(DV.reindex(columns=S.columns)>=2e6)&(S>=3.0)
OUT=json.load(open("/tmp/verdict_evidence.json"))
# ---- lottery 2005 -> 2026 ----
i0=ME.index.get_indexer([pd.Timestamp("2005-06-01")],method="nearest")[0]
i1=len(ME.index)-1
p0=S.iloc[i0]; el=liq.iloc[i0]&p0.notna()
lastp=S.iloc[i0:].ffill().iloc[-1]
r=(lastp/p0)[el[el].index].dropna()          # total multiple over ~21y
q_mult=float(ME["QQQ"].iloc[i1]/ME["QQQ"].iloc[i0])
OUT["lottery"]={"n":int(len(r)),"years":21,
 "p_lose":float((r<1).mean()),"p_10x":float((r>=10).mean()),"p_100x":float((r>=100).mean()),
 "p_beat_qqq":float((r>=q_mult).mean()),"qqq_mult":q_mult,
 "pct":{k:float(np.percentile(r,v)) for k,v in [("p10",10),("p25",25),("p50",50),("p75",75),("p90",90),("p99",99)]},
 "mean":float(r.mean()),"died":int(S.iloc[i1][el[el].index].isna().sum())}
# ---- blends: QQQ/AGG DCA 2006-2026 ----
def monthly(t):
    df=pd.read_csv(f"{ROOT}/data/etfs/{t}.csv",parse_dates=["Date"]).set_index("Date")
    px=df["Adj Close"] if ("Adj Close" in df.columns and df["Adj Close"].notna().sum()>100) else df["Close"]
    return px.resample("ME").last()
q=monthly("QQQ"); a=monthly("AGG")
idx=q.index.intersection(a.index); idx=idx[(idx>=pd.Timestamp("2006-01-01"))&(idx<=pd.Timestamp("2026-06-30"))]
qr=q.reindex(idx).pct_change().fillna(0); ar=a.reindex(idx).pct_change().fillna(0)
def dca_stats(w):
    v=0.0; path=[]
    for x,y in zip(qr.values,ar.values):
        v=(v+1000.0)*(1+w*x+(1-w)*y); path.append(v)
    p=pd.Series(path); return round(float(p.iloc[-1])), round(float((p/p.cummax()-1).min())*100)
rows=[]
for w in [1.0,0.8,0.6,0.4]:
    fw,dd=dca_stats(w); rows.append({"w":int(w*100),"final":fw,"dd":dd})
OUT["blends"]={"rows":rows,"n_months":int(len(idx))}
json.dump(OUT,open("/tmp/verdict_evidence.json","w"))
print("lottery:",{k:round(v,4) if isinstance(v,float) else v for k,v in OUT["lottery"].items() if k!="pct"})
print("pct:",{k:round(v,2) for k,v in OUT["lottery"]["pct"].items()})
print("blends:",OUT["blends"]["rows"])
