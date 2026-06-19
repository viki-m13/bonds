"""Exp 35 — refine insider buying to the literature's stronger variants:
CLUSTER buys (>=2, >=3 distinct insiders same stock-month) and OFFICER / CEO-CFO
buys. Rebuild rich panel (join REPORTINGOWNER for role/title), then long-only
portfolio of each refined signal, net 20bps, vs RANDOM same-size + EW + SPY +
QQQ, full + sub-period. Does refinement strengthen the edge?"""
import warnings, time, io, zipfile, urllib.request, ssl
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "research viktormashalov@gmail.com"}
t0 = time.time()
def get(u):
    for k in range(3):
        try: return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90, context=ctx).read()
        except Exception: time.sleep(2*(k+1))
    return None
frames = []
for yr in range(2010, 2026):
    for q in (1, 2, 3, 4):
        if (yr, q) > (2025, 2): break
        raw = get(f"https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{yr}q{q}_form345.zip")
        if not raw: continue
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
            sub = pd.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), sep="\t", dtype=str,
                              usecols=["ACCESSION_NUMBER","FILING_DATE","DOCUMENT_TYPE","ISSUERTRADINGSYMBOL"])
            nt = pd.read_csv(io.BytesIO(z.read("NONDERIV_TRANS.tsv")), sep="\t", dtype=str,
                             usecols=["ACCESSION_NUMBER","TRANS_CODE","TRANS_SHARES","TRANS_PRICEPERSHARE"])
            ro = pd.read_csv(io.BytesIO(z.read("REPORTINGOWNER.tsv")), sep="\t", dtype=str,
                             usecols=["ACCESSION_NUMBER","RPTOWNER_RELATIONSHIP","RPTOWNER_TITLE"])
        except Exception: continue
        sub = sub[sub.DOCUMENT_TYPE.isin(["4","4/A"])]
        nt = nt[nt.TRANS_CODE.isin(["P","S"])].copy()
        nt["val"] = pd.to_numeric(nt.TRANS_SHARES, errors="coerce") * pd.to_numeric(nt.TRANS_PRICEPERSHARE, errors="coerce")
        nt = nt.dropna(subset=["val"])
        # per accession: buy/sell totals
        ab = nt.groupby(["ACCESSION_NUMBER","TRANS_CODE"]).val.sum().unstack(fill_value=0.0)
        ab = ab.rename(columns={"P":"buy","S":"sell"}).reset_index()
        for c in ("buy","sell"):
            if c not in ab: ab[c] = 0.0
        ro = ro.drop_duplicates("ACCESSION_NUMBER")
        ab = ab.merge(sub, on="ACCESSION_NUMBER").merge(ro, on="ACCESSION_NUMBER", how="left")
        ab = ab[(ab.ISSUERTRADINGSYMBOL.notna()) & (ab.ISSUERTRADINGSYMBOL != "NONE")]
        ab["fd"] = pd.to_datetime(ab.FILING_DATE, errors="coerce"); ab = ab.dropna(subset=["fd"])
        ab["ym"] = ab.fd.values.astype("datetime64[M]")
        rel = ab.RPTOWNER_RELATIONSHIP.fillna(""); tit = ab.RPTOWNER_TITLE.fillna("").str.upper()
        ab["is_officer"] = rel.str.contains("Officer")
        ab["is_ceocfo"] = tit.str.contains("CEO|CFO|CHIEF|PRESIDENT", regex=True)
        ab["is_buyer"] = ab.buy > 0
        ab["offbuy_v"] = np.where(ab.is_officer & ab.is_buyer, ab.buy, 0.0)
        ab["ceobuy_v"] = np.where(ab.is_ceocfo & ab.is_buyer, ab.buy, 0.0)
        g = ab.groupby(["ISSUERTRADINGSYMBOL","ym"]).agg(
            buy=("buy","sum"), sell=("sell","sum"),
            nbuyers=("is_buyer","sum"),
            off_buy=("offbuy_v","sum"), ceo_buy=("ceobuy_v","sum")).reset_index()
        frames.append(g)
    print(f"  through {yr}  t={time.time()-t0:.0f}s", flush=True)
P = pd.concat(frames).groupby(["ISSUERTRADINGSYMBOL","ym"]).sum().reset_index()
P.columns = ["tk","ym","buy","sell","nbuyers","off_buy","ceo_buy"]
P.to_pickle("/tmp/wave/_insider_rich.pkl")
print(f"rich panel {len(P)} rows  t={time.time()-t0:.0f}s", flush=True)

uni = set()
for f in ("sp500_universe.txt","xuniverse_sp400.txt","xuniverse_ndx.txt"):
    txt = open(f"/tmp/wave/{f}").read()
    uni |= set(txt.split()) if " " in txt else set(l.strip() for l in txt.splitlines() if l.strip())
uni = sorted(t for t in uni if t and t.isalpha())
import os
if os.path.exists("/tmp/wave/_ins_px.pkl"):
    me = pd.read_pickle("/tmp/wave/_ins_px.pkl"); names = [c for c in me.columns if c not in ("SPY","QQQ")]
else:
    px = yf.download(uni+["SPY","QQQ"], start="2009-06-01", auto_adjust=True, progress=False)["Close"]
    names = [t for t in uni if t in px.columns and px[t].notna().sum()>800]
    me = px[names+["SPY","QQQ"]].resample("ME").last(); me.index = me.index.to_period("M").to_timestamp()
    me.to_pickle("/tmp/wave/_ins_px.pkl")
mret = me.pct_change()
Pn = P[P.tk.isin(names)]
def pan(col): return Pn.pivot_table(index="ym",columns="tk",values=col,aggfunc="sum").reindex(index=me.index,columns=names).fillna(0)
buy,sell,nbuyers,offb,ceob = pan("buy"),pan("sell"),pan("nbuyers"),pan("off_buy"),pan("ceo_buy")
net3 = (buy-sell).rolling(3,min_periods=1).sum()
nb3 = nbuyers.rolling(3,min_periods=1).sum()
SIG = {
  "net-buyer (base)": (net3>0),
  "cluster>=2": (nb3>=2),
  "cluster>=3": (nb3>=3),
  "officer buy": (offb.rolling(3,min_periods=1).sum()>0),
  "CEO/CFO buy": (ceob.rolling(3,min_periods=1).sum()>0),
}
print(f"priced {len(names)} names  t={time.time()-t0:.0f}s", flush=True)
def port(flagdf, cost=0.002, rand=False, seed=0):
    rng=np.random.default_rng(seed); prev=set(); rets=[]
    for i in range(3,len(me.index)-1):
        d=me.index[i]; nxt=me.index[i+1]
        avail=[t for t in names if np.isfinite(mret.loc[nxt,t])]
        if rand:
            k=max(5,int(flagdf.loc[d,avail].sum())); sel=list(rng.choice(avail,min(k,len(avail)),replace=False))
        else:
            sel=[t for t in avail if flagdf.loc[d,t]]
        if len(sel)<5: continue
        turn=1.0-len(prev&set(sel))/max(len(sel),1)
        rets.append((nxt, mret.loc[nxt,sel].mean()-turn*cost)); prev=set(sel)
    return pd.Series(dict(rets)).dropna()
def st(s,lo=None,hi=None):
    if lo: s=s[(s.index>=lo)&(s.index<hi)]
    s=s.dropna(); eq=(1+s).cumprod(); yrs=len(s)/12
    return eq.iloc[-1]**(1/yrs)-1, s.mean()/(s.std()+1e-12)*np.sqrt(12), float((eq/eq.cummax()-1).min())
print("\nRefined insider signals — long-only NET 20bps (CAGR/Sharpe/maxDD; avg names):", flush=True)
for nm,flag in SIG.items():
    s=port(flag); c,sh,dd=st(s); avgn=flag.loc[me.index[3:]].sum(axis=1).mean()
    r=port(flag,rand=True); rc,_,_=st(r)
    print(f"  {nm:18s} CAGR {c*100:5.1f}% Sh {sh:.2f} DD {dd*100:4.0f}%  | random {rc*100:5.1f}%  | "
          f"edge {(c-rc)*100:+4.1f}pp | ~{avgn:.0f} names  [2010-17 {st(s,'2010-01-01','2018-01-01')[0]*100:.0f}% vs r{st(r,'2010-01-01','2018-01-01')[0]*100:.0f}% | 2018-25 {st(s,'2018-01-01','2025-07-01')[0]*100:.0f}% vs r{st(r,'2018-01-01','2025-07-01')[0]*100:.0f}%]", flush=True)
print(f"  {'QQQ':18s} CAGR {st(mret['QQQ'].reindex(s.index).dropna())[0]*100:5.1f}% Sh {st(mret['QQQ'].reindex(s.index).dropna())[1]:.2f}", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
