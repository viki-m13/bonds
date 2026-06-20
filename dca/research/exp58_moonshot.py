import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
stocks=set(uni[uni.assetType=="Stock"].ticker)
cols=[c for c in me.columns if c in stocks]
px=me[cols].astype(float)
p(f"universe {len(cols)} stocks, {me.index[0].date()}..{me.index[-1].date()}  t={time.time()-t0:.0f}s")

# ---------- PART A: case studies — actual pre-rally technical signature ----------
def biggest_run(s, win=24):
    s=s.dropna()
    if len(s)<win+3: return None
    # find the start month maximizing forward `win`-month return
    fwd=s.shift(-win)/s-1
    if fwd.dropna().empty: return None
    st=fwd.idxmax(); ret=fwd.max()
    return st,ret
p("\n=== CASE STUDIES: signature in the 3 months BEFORE the biggest 24-mo run ===")
p(f"{'tk':6} {'launch':>10} {'fwd24m':>7} | {'mom12':>6} {'mom6':>6} {'mom3':>6} {'%of12mHigh':>10} {'vol6':>6} {'px':>7}")
for tk in ["NVDA","AVGO","AMD","GOOGL","AAPL","LLY","ARM","WDC","SNDK","MSFT","TSLA","META","NFLX","ANET","SMCI","VRT"]:
    if tk not in px.columns: p(f"{tk:6} (not in data)"); continue
    s=px[tk]; r=biggest_run(s,24)
    if r is None: p(f"{tk:6} (insufficient history)"); continue
    st,ret=r
    # features measured AT launch month (backward looking)
    hi12=s.rolling(12).max(); ret_m=(s/s.shift(1)-1)
    try:
        mom12=s[st]/s.loc[:st].shift(12).loc[st]-1
    except Exception: mom12=np.nan
    i=s.index.get_loc(st)
    def back(k):
        return s.iloc[i]/s.iloc[i-k]-1 if i-k>=0 else np.nan
    mom12=back(12); mom6=back(6); mom3=back(3)
    pctHigh=s.iloc[i]/hi12.iloc[i] if not np.isnan(hi12.iloc[i]) else np.nan
    vol6=ret_m.iloc[max(0,i-6):i].std()
    p(f"{tk:6} {str(st.date()):>10} {ret:>7.0%} | {mom12:>6.0%} {mom6:>6.0%} {mom3:>6.0%} {pctHigh:>10.0%} {vol6:>6.1%} {s.iloc[i]:>7.0f}")

# ---------- PART B: systematic — what features mark a future moonshot ----------
p(f"\n=== SYSTEMATIC: features of moonshots (fwd-12m top 2% cross-sectionally) ===  t={time.time()-t0:.0f}s")
ret_m=(px/px.shift(1)-1)
mom12=px/px.shift(12)-1; mom6=px/px.shift(6)-1; mom3=px/px.shift(3)-1; mom1=ret_m
hi12=px.rolling(12).max(); lo12=px.rolling(12).min()
pctHigh=px/hi12; pctLow=px/lo12-1
vol6=ret_m.rolling(6).std()
accel=mom3-(mom12-mom3)            # recent 3m vs prior 9m pace
fwd12=(px.shift(-12)/px-1)
liq=(px>=3.0)
fok=fwd12.where(liq)
# moonshot threshold: top 2% of fwd12 each month
thr=fok.quantile(0.98,axis=1)
moon=fok.gt(thr,axis=0)
midx=px.index[(px.index>=pd.Timestamp("2011-01-01"))&(px.index<=pd.Timestamp("2024-06-30"))]
feats={"mom12":mom12,"mom6":mom6,"mom3":mom3,"mom1":mom1,"%of12mHigh":pctHigh,
       "%above12mLow":pctLow,"vol6":vol6,"accel":accel,"price":px}
moonM=moon.loc[midx]; liqM=liq.loc[midx]
p(f"{'feature':>14} {'moonshot avg-rank':>18} {'rest avg-rank':>14}  (cross-sec pctile 0-1)")
for nm,f in feats.items():
    fr=f.where(liq).rank(axis=1,pct=True).loc[midx]
    mv=fr.where(moonM).stack().mean(); rv=fr.where((~moonM)&liqM).stack().mean()
    p(f"{nm:>14} {mv:>18.2f} {rv:>14.2f}")
# IC of each feature with fwd12 (rank corr), avg over months
p(f"\n{'feature':>14} {'rankIC vs fwd12':>16}")
for nm,f in feats.items():
    fr=f.where(liq); ics=[]
    sub=fok[mask]
    for dt in sub.index[::3]:
        a=fr.loc[dt]; b=fok.loc[dt]; d=pd.concat([a,b],axis=1).dropna()
        if len(d)>50: ics.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    p(f"{nm:>14} {np.nanmean(ics):>16.3f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
