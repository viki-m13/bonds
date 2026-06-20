import glob, numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
acs,vos=[],[]
for f in sorted(glob.glob("/home/user/bonds/dca/research/data/tiingo/prices/ac_*.parquet")):
    d=pd.read_parquet(f);
    if d.shape[1]: d.index=pd.to_datetime(d.index); acs.append(d)
for f in sorted(glob.glob("/home/user/bonds/dca/research/data/tiingo/prices/vol_*.parquet")):
    d=pd.read_parquet(f)
    if d.shape[1]: d.index=pd.to_datetime(d.index); vos.append(d)
C=pd.concat(acs,axis=1).astype("float32"); C=C.loc[:,~C.columns.duplicated()].sort_index()
V=pd.concat(vos,axis=1).astype("float32"); V=V.loc[:,~V.columns.duplicated()].sort_index()
C=C[C.index>=pd.Timestamp("2009-06-01")]; V=V.reindex(C.index)
common=[c for c in C.columns if c in V.columns]; C=C[common]; V=V[common]
# liquidity filter
dv=C*V; adv60=dv.rolling(60,min_periods=30).mean()
liqcols=[c for c in C.columns if adv60[c].max()>=5e6]
C=C[liqcols]; V=V[liqcols]; dv=dv[liqcols]; adv60=adv60[liqcols]
p(f"daily C{C.shape} liquid {len(liqcols)} t={time.time()-t0:.0f}s")
ME=C.index.to_period("M").to_timestamp("M")   # for month-end sampling
mfirst=pd.Series(C.index,index=C.index).groupby(C.index.to_period("M")).last()  # last trading day each month
msel=mfirst.values
def monthly(daily):
    return daily.reindex(pd.DatetimeIndex(msel))
rc=C.pct_change()
sma20=C.rolling(20,min_periods=20).mean(); sma50=C.rolling(50,min_periods=50).mean(); sma200=C.rolling(200,min_periods=150).mean()
adv20=dv.rolling(20,min_periods=10).mean()
hi252=C.rolling(252,min_periods=120).max(); lo252=C.rolling(252,min_periods=120).min()
FEAT2={}
# --- volume / accumulation (pre-run smart-money) ---
FEAT2["vol_dryup"]=monthly(adv20/adv60)                              # <1 = quiet base
up=(rc>0); dn=(rc<0)
gv=(V.where(up,0)).rolling(20,min_periods=10).sum(); rv=(V.where(dn,0)).rolling(20,min_periods=10).sum()
FEAT2["accum20"]=monthly(gv/(rv+1e-9))                                # >1 institutional accumulation
# OBV slope (normalized): cumulative signed volume, 20d slope / adv
obv=(np.sign(rc)*V).cumsum()
FEAT2["obv_slope20"]=monthly((obv-obv.shift(20))/(adv20*20+1e-9))
FEAT2["vol_surge"]=monthly(V/adv20)                                   # today vs avg (breakout vol)
# --- volatility compression / coiled spring ---
std20=rc.rolling(20,min_periods=15).std()
std120=rc.rolling(120,min_periods=60).std()
FEAT2["bb_squeeze"]=monthly(std20/(std120+1e-9))                      # <1 = compressing
rng20=(C.rolling(20).max()-C.rolling(20).min())/C
rng120=(C.rolling(120).max()-C.rolling(120).min())/C
FEAT2["range_comp"]=monthly(rng20/(rng120+1e-9))                      # tightness
# --- base position / breakout proximity ---
FEAT2["dist_52w_high"]=monthly(C/hi252-1)                             # near 0 = at highs
FEAT2["pct_above_200"]=monthly(C/sma200-1)
FEAT2["pct_off_low"]=monthly(C/lo252-1)
# days since 252d high (recency of breakout)
ishigh=(C>=hi252*0.999)
dsh=pd.DataFrame(np.nan,index=C.index,columns=C.columns)
last=np.full(C.shape[1],np.nan)
arr=ishigh.values
out=np.empty(C.shape,dtype="float32")
cnt=np.full(C.shape[1],9999.0)
for i in range(C.shape[0]):
    cnt=cnt+1; cnt[arr[i]]=0; out[i]=cnt
FEAT2["days_since_high"]=monthly(pd.DataFrame(out,index=C.index,columns=C.columns))
# RSI14
delta=C.diff(); ug=delta.clip(lower=0).rolling(14,min_periods=14).mean(); dg=(-delta.clip(upper=0)).rolling(14,min_periods=14).mean()
FEAT2["rsi14"]=monthly(100-100/(1+ug/(dg+1e-9)))
# MA stack / trend quality
FEAT2["above_all_ma"]=monthly(((C>sma20)&(sma20>sma50)&(sma50>sma200)).astype("float32"))
FEAT2["sma50_slope"]=monthly((sma50/sma50.shift(20)-1))
# liquidity level
FEAT2["log_advdollar"]=monthly(np.log(adv60.clip(lower=1)))
for k in FEAT2: FEAT2[k].index=pd.DatetimeIndex(FEAT2[k].index).to_period("M").to_timestamp()
pd.to_pickle(FEAT2,"/tmp/wave/_techfeat.pkl")
p(f"saved {len(FEAT2)} technical/volume features: {list(FEAT2)}")
p(f"DONE t={time.time()-t0:.0f}s")
