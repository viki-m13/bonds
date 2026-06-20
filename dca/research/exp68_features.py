import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
me=pd.read_pickle("/tmp/wave/_tiingo_me.pkl"); me=me.loc[:,~me.columns.duplicated()]; me.index=pd.to_datetime(me.index)
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
stocks=set(uni[uni.assetType=="Stock"].ticker)
cols=[c for c in me.columns if c in stocks]; me=me[cols]
M=me.index
def qmap(df):  # quarterly(quarter-end idx) -> monthly avail-lagged, aligned to cols
    df=df.reindex(columns=cols)
    av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]
    return d2.reindex(M,method="ffill",limit=6)
# ---------- revenue ----------
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
qe=pd.PeriodIndex([q[2:] for q in rev.index],freq="Q").to_timestamp(how="end").normalize(); rev.index=qe
rev_yoy=rev/rev.shift(4)-1
rev_accel=((rev_yoy.diff()>0)&(rev_yoy.diff().shift(1)>0)&(rev_yoy>0)).astype(float)
rev_surprise=rev_yoy-rev_yoy.rolling(4,min_periods=2).mean()
# ---------- fundamentals ----------
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
def g(k):
    d=F.get(k)
    if d is None: return None
    d=d.copy(); d.index=pd.PeriodIndex([q[2:] for q in d.index],freq="Q").to_timestamp(how="end").normalize(); return d
GP,OI,NI,RD=g("GrossProfit"),g("OperatingIncomeLoss"),g("NetIncomeLoss"),g("ResearchAndDevelopmentExpense")
AST,EQ,CASH,STI,LTD=g("Assets"),g("StockholdersEquity"),g("CashAndCashEquivalentsAtCarryingValue"),g("ShortTermInvestments"),g("LongTermDebt")
SH=g("EntityCommonStockSharesOutstanding")
def align(d): return d.reindex(index=rev.index,columns=rev.columns) if d is not None else None
GP,OI,NI,RD,AST,EQ,CASH,STI,LTD,SH=[align(x) for x in [GP,OI,NI,RD,AST,EQ,CASH,STI,LTD,SH]]
gross_margin=GP/rev; op_margin=OI/rev; net_margin=NI/rev
gm_delta=gross_margin-gross_margin.shift(4); om_delta=op_margin-op_margin.shift(4)
rps=rev/SH; rps_yoy=rps/rps.shift(4)-1
share_chg=SH/SH.shift(4)-1
rnd_int=RD/rev; oi_yoy=OI/OI.shift(4).abs()-np.sign(OI.shift(4))  # rough
op_leverage=(OI/OI.shift(4)-1)-(rev/rev.shift(4)-1)
roe=(NI*4)/EQ; roa=(NI*4)/AST
rule40=rev_yoy+op_margin
ni_inflect=((NI>0)&(NI.shift(1)<=0)).astype(float)        # turned profitable
# market cap proxy & net cash
# monthly fundamentals
FEAT={}
for nm,d in [("rev_yoy",rev_yoy),("rev_accel",rev_accel),("rev_surprise",rev_surprise),
             ("gross_margin",gross_margin),("op_margin",op_margin),("net_margin",net_margin),
             ("gm_delta",gm_delta),("om_delta",om_delta),("rps_yoy",rps_yoy),("share_chg",share_chg),
             ("rnd_int",rnd_int),("op_leverage",op_leverage),("roe",roe),("roa",roa),
             ("rule40",rule40),("ni_inflect",ni_inflect)]:
    FEAT[nm]=qmap(d)
SHm=qmap(SH); mcap=me*SHm; netcash=qmap((CASH.fillna(0)+STI.fillna(0)-LTD.fillna(0)))/mcap
FEAT["netcash_mcap"]=netcash; FEAT["log_mcap"]=np.log(mcap)
p(f"fundamentals features {len(FEAT)} t={time.time()-t0:.0f}s")
# ---------- insider (rich) ----------
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P["ym"]=pd.to_datetime(P.ym)
P=P[P.tk.isin(set(cols))]
def ip(col): return P.pivot_table(index="ym",columns="tk",values=col,aggfunc="sum").reindex(index=M,columns=cols).fillna(0)
buy,sell,nb,offb,ceob=ip("buy"),ip("sell"),ip("nbuyers"),ip("off_buy"),ip("ceo_buy")
buy3=buy.rolling(3,min_periods=1).sum(); nb3=nb.rolling(3,min_periods=1).sum()
off3=offb.rolling(3,min_periods=1).sum(); ceo3=ceob.rolling(3,min_periods=1).sum()
net3=(buy-sell).rolling(3,min_periods=1).sum()
mom6=me/me.shift(6)-1
FEAT["ins_clustern"]=nb3
FEAT["ins_buy$"]=buy3
FEAT["ins_buy$_mcap"]=buy3/mcap
FEAT["ins_ceo"]=(ceo3>0).astype(float)
FEAT["ins_net"]=net3
FEAT["ins_opportunistic"]=((buy3>0)&(mom6<0)).astype(float)        # buying after a fall
FEAT["ins_x_revaccel"]=((nb3>=2)&(qmap(rev_accel)>0.5)).astype(float)  # NOVEL: cluster buy + rev inflection
FEAT["ins_x_marginexp"]=((buy3>0)&(qmap(gm_delta)>0)).astype(float)    # NOVEL: insider + margin expansion
# ---------- technical ----------
ma10=me.rolling(10,min_periods=10).mean(); vol6=me.pct_change().rolling(6,min_periods=4).std()
hi12=me.rolling(12,min_periods=6).max()
FEAT["mom3"]=me/me.shift(3)-1; FEAT["mom6"]=mom6; FEAT["mom12"]=me/me.shift(12)-1
FEAT["trend"]=(me>ma10).astype(float); FEAT["distHigh"]=me/hi12-1
FEAT["vol6"]=vol6; FEAT["vol_contraction"]=(vol6.rank(axis=1,pct=True)<0.3).astype(float)
FEAT["price_accel"]=(me/me.shift(3)-1)-(me/me.shift(12)-1)
# NOVEL composite: quiet compounder = rev accel + margin exp + low vol + insider, not yet moved
FEAT["quiet_compounder"]=((qmap(rev_accel)>0.5)&(qmap(gm_delta)>0)&(vol6.rank(axis=1,pct=True)<0.4)&(buy3>0)&(mom6<0.3)).astype(float)
FEAT["triple_confirm"]=((nb3>=2)&(qmap(rev_accel)>0.5)&(qmap(gm_delta)>0)).astype(float)
p(f"all features {len(FEAT)} t={time.time()-t0:.0f}s")
# ---------- fwd returns + IC ----------
liq=(me.shift(1)>=3.0).fillna(False)
fwd12=(me.shift(-12)/me-1).clip(-0.95,5.0)
fok=fwd12.where(liq)
pd.to_pickle({"FEAT":FEAT,"fok":fok,"liq":liq,"me":me,"cols":cols},"/tmp/wave/_featmat.pkl")
idxA=M[(M>=pd.Timestamp("2012-07-01"))&(M<=pd.Timestamp("2024-06-30"))]
def ic(f,m):
    fr=f.where(liq); ics=[]
    for dt in m[::2]:
        a=fr.loc[dt]; b=fok.loc[dt]; d=pd.concat([a,b],axis=1).dropna()
        if len(d)>40 and d.iloc[:,0].std()>0: ics.append(d.iloc[:,0].corr(d.iloc[:,1],method="spearman"))
    return np.nanmean(ics) if ics else np.nan
eras=[("ALL",idxA),("2012-16",M[(M>=pd.Timestamp("2012-07-01"))&(M<=pd.Timestamp("2016-12-31"))]),
      ("2017-20",M[(M>=pd.Timestamp("2017-01-01"))&(M<=pd.Timestamp("2020-12-31"))]),
      ("2021-24",M[(M>=pd.Timestamp("2021-01-01"))&(M<=pd.Timestamp("2024-06-30"))])]
p(f"\n{'feature':20} "+" ".join(f"{e[0]:>9}" for e in eras))
rows=[]
for nm,f in FEAT.items():
    vals=[ic(f,e[1]) for e in eras]; rows.append((nm,vals[0]))
    p(f"{nm:20} "+" ".join(f"{v:>9.3f}" for v in vals))
p("\nTop features by |IC| (ALL):")
for nm,v in sorted(rows,key=lambda x:-abs(x[1]) if not np.isnan(x[1]) else 0)[:12]:
    p(f"  {nm:20} {v:+.3f}")
p(f"\nsaved /tmp/wave/_featmat.pkl  DONE t={time.time()-t0:.0f}s")
