"""Exp 49 — CLEAN (survivorship-corrected) insider test on Tiingo delisting-
inclusive prices. Insider-officer-buy stock-months vs rest, forward 3m return,
INCLUDING delisted names (their forward return uses last-traded price = realized
loss up to delisting). Compare to survivor-ONLY version to measure how much
survivorship inflated the earlier (yfinance) result. Preliminary: ~half the
delisted insider names so far."""
import warnings, glob, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0=time.time()
# load Tiingo daily adjClose (all chunks), monthly
ac={}
for f in sorted(glob.glob("/home/user/bonds/dca/research/data/tiingo/prices/ac_*.parquet")):
    d=pd.read_parquet(f)
    for c in d.columns: ac[c]=d[c]
AC=pd.DataFrame(ac); AC.index=pd.to_datetime(AC.index)
me=AC.resample("ME").last(); me.index=me.index.to_period("M").to_timestamp()
print(f"Tiingo prices: {me.shape[1]} tickers, {me.index.min().date()}->{me.index.max().date()} t={time.time()-t0:.0f}s",flush=True)
# universe membership / delisted
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
uni["endDate"]=pd.to_datetime(uni.endDate,errors="coerce")
deli=set(uni[(uni.assetType=="Stock")&(uni.endDate<"2025-01-01")].ticker)
# insider officer-buy panel
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P=P[P.off_buy>0]
P["ym"]=pd.to_datetime(P.ym)
off=P.pivot_table(index="ym",columns="tk",values="off_buy",aggfunc="sum")
names=[c for c in me.columns if c in off.columns]
print(f"insider names with Tiingo prices: {len(names)} (delisted: {len([n for n in names if n in deli])})",flush=True)
off=off.reindex(index=me.index,columns=names).fillna(0)
buy3=(off.rolling(3,min_periods=1).sum()>0)
# forward 3m rel return, delisting-aware: carry last price up to 3m (realized at last trade)
mef=me[names].ffill(limit=3)
def fwd(h): return mef.shift(-h)/me[names]-1
qqq=yf.download("QQQ",start="2009-01-01",auto_adjust=True,progress=False)["Close"]
if hasattr(qqq,'columns'): qqq=qqq.iloc[:,0]
qm=qqq.resample("ME").last(); qm.index=qm.index.to_period("M").to_timestamp()
def run(universe_filter, tag):
    f=fwd(3); fq=qm.shift(-3)/qm-1
    byr=[]; rest=[]
    for d in me.index:
        if not (pd.Timestamp("2011-01-01")<=d<pd.Timestamp("2025-01-01")): continue
        q=fq.get(d,np.nan)
        for t in names:
            if not universe_filter(t): continue
            y=f.loc[d,t]
            if np.isfinite(y) and np.isfinite(q):
                (byr if buy3.loc[d,t] else rest).append(y-q)
    b=np.array(byr); r=np.array(rest)
    print(f"  {tag}: insider-buy {b.mean()*100:+.2f}% (n={len(b)}) | rest {r.mean()*100:+.2f}% (n={len(r)}) | diff {(b.mean()-r.mean())*100:+.2f}pp",flush=True)
print("\nForward-3m return vs QQQ (insider-officer-buy months vs rest):",flush=True)
run(lambda t: True, "ALL incl delisted (survivorship-CORRECTED)")
run(lambda t: t not in deli, "survivors ONLY (survivorship-BIASED, like yfinance)")
print(f"\nDONE t={time.time()-t0:.0f}s",flush=True)
