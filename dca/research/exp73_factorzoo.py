import numpy as np, pandas as pd, time
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def qidx(df):
    df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=qidx(rev)
def gq(k):
    d=F.get(k);
    if d is None: return None
    return qidx(d).reindex(columns=rev.columns)
GP,OI,NI,AST,EQ,CASH,STI,LTD,SH=[gq(k) for k in ["GrossProfit","OperatingIncomeLoss","NetIncomeLoss","Assets","StockholdersEquity","CashAndCashEquivalentsAtCarryingValue","ShortTermInvestments","LongTermDebt","EntityCommonStockSharesOutstanding"]]
def qmap(df):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=6)
SHm=qmap(SH); mcap=me*SHm
# ---- new factors (sign so HIGHER = predicted-higher return) ----
NEW={}
NEW["BM"]=qmap(EQ)/mcap                                  # value: book/market
NEW["EP"]=qmap(NI*4)/mcap                                 # earnings yield
NEW["GP_A"]=qmap(GP*4)/qmap(AST)                          # Novy-Marx gross profitability
NEW["assetgrowth_neg"]=-qmap(AST/AST.shift(4)-1)         # investment anomaly (low growth good)
NEW["sales_to_assets"]=qmap(rev*4)/qmap(AST)            # asset turnover
# Sloan accruals = -(dNI proxy): use -(change in non-cash assets)/assets
nca=AST-(CASH.fillna(0)+STI.fillna(0))
NEW["accruals_neg"]=-qmap(nca.diff()/AST.shift(1))      # low accruals good
NEW["leverage_neg"]=-qmap(LTD.fillna(0)/AST)            # low leverage good
# Piotroski-lite (0-5): ROA>0, dROA>0, dGM>0, dShares<=0, dLeverage<=0
roa=NI*4/AST
pz=((roa>0).astype(float)+(roa.diff()>0).astype(float)
    +((GP/rev).diff()>0).astype(float)+(SH.diff()<=0).astype(float)
    +((LTD.fillna(0)/AST).diff()<=0).astype(float))
NEW["piotroski"]=qmap(pz)
# pull reusable ones from featmat (already monthly-aligned)
USE={"ROA":FEAT["roa"],"ROE":FEAT["roe"],"lowvol":-FEAT["vol6"],"mom12":FEAT["mom12"],
     "near52high":FEAT["distHigh"],"buyback":-FEAT["share_chg"],"opleverage":FEAT["op_leverage"],
     "revaccel":FEAT["rev_accel"],"insider":FEAT["ins_clustern"]}
FAC={**NEW,**USE}
ret=(me/me.shift(1)-1).clip(-0.9,2.0); fwd1=ret.shift(-1)
LIQ=(liq&(me>=3.0))
idx=M[(M>=pd.Timestamp("2012-07-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def ls_series(fac,q=0.2):
    fr=fac.where(LIQ); rk=fr.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    sw=(rk<=q).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)
    lr=(lw*fwd1).sum(axis=1); sr=(sw*fwd1).sum(axis=1)
    return (lr-sr).reindex(idx), (lr).reindex(idx)   # market-neutral L/S, and long-only-leg
def ann(r):
    r=r.dropna();
    if len(r)<12 or r.std()==0: return (np.nan,np.nan)
    return r.mean()*12, r.mean()/r.std()*np.sqrt(12)
p(f"{'factor':18} {'LS_ann%':>8} {'LS_Sharpe':>10} {'corrQQQ':>8} {'longSharpe':>11}")
LS={}
for nm,fac in FAC.items():
    ls,lo=ls_series(fac); LS[nm]=ls
    a,s=ann(ls); _,ls_corr=0,ls.corr(qret); _,lsh=ann(lo)
    p(f"{nm:18} {a*100:>8.1f} {s:>10.2f} {ls.corr(qret):>8.2f} {lsh:>11.2f}")
# ---- combined multi-factor market-neutral (avg z of all factors) ----
def z(fac):
    fr=fac.where(LIQ); return fr.sub(fr.mean(axis=1),axis=0).div(fr.std(axis=1).replace(0,np.nan),axis=0)
combo=sum(z(f) for f in FAC.values())
clo_ls,clo_long=ls_series(combo,0.1)
a,s=ann(clo_ls); p(f"\nCOMBINED L/S (decile, {len(FAC)} factors): ann {a*100:.1f}% Sharpe {s:.2f} corrQQQ {clo_ls.corr(qret):.2f}")
clo_ls2,_=ls_series(combo,0.2); a2,s2=ann(clo_ls2); p(f"COMBINED L/S (quintile): ann {a2*100:.1f}% Sharpe {s2:.2f} corrQQQ {clo_ls2.corr(qret):.2f}")
# quality-value-momentum subset combo (drop weak/insider)
strong=["GP_A","ROA","lowvol","near52high","mom12","buyback","assetgrowth_neg","EP","piotroski","accruals_neg"]
combo2=sum(z(FAC[k]) for k in strong)
s_ls,_=ls_series(combo2,0.1); a,s=ann(s_ls); p(f"\nSTRONG-10 combo L/S decile: ann {a*100:.1f}% Sharpe {s:.2f} corrQQQ {s_ls.corr(qret):.2f}")
# sub-periods of strong combo L/S
p("Strong-combo L/S sub-period Sharpe:")
for lo2,hi2 in [("2012","2016"),("2017","2020"),("2021","2025")]:
    m=(idx>=pd.Timestamp(lo2))&(idx<=pd.Timestamp(hi2+"-12-31")); a,s=ann(s_ls[m]); p(f"  {lo2}-{hi2}: ann {a*100:>5.1f}% Sharpe {s:.2f}")
pd.to_pickle({"LS":LS,"combo_ls":s_ls,"idx":idx},"/tmp/wave/_ls.pkl")
p(f"\nsaved _ls.pkl DONE t={time.time()-t0:.0f}s")
