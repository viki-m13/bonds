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
    d=F.get(k); return qidx(d).reindex(columns=rev.columns) if d is not None else None
GP,NI,AST,EQ,SH=[gq(k) for k in ["GrossProfit","NetIncomeLoss","Assets","StockholdersEquity","EntityCommonStockSharesOutstanding"]]
def qmap(df):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=6)
mcap=me*qmap(SH)
FAC={"BM":qmap(EQ)/mcap,"GP_A":qmap(GP*4)/qmap(AST),"sales_to_assets":qmap(rev*4)/qmap(AST),
     "ROA":FEAT["roa"],"ROE":FEAT["roe"],"mom12":FEAT["mom12"],"near52high":FEAT["distHigh"],
     "buyback":-FEAT["share_chg"],"revaccel":FEAT["rev_accel"],"insider":FEAT["ins_clustern"],"EP":qmap(NI*4)/mcap}
ret=(me/me.shift(1)-1).clip(-0.9,2.0); fwd1=ret.shift(-1)
LIQ=(liq&(me>=3.0)); LIQ5=(liq&(me>=5.0))           # shorts need higher liquidity
idx=M[(M>=pd.Timestamp("2012-07-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def ann(r):
    r=r.dropna(); a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
def legs(fac,q=0.2):
    fr=fac.where(LIQ); rk=fr.rank(axis=1,pct=True)
    lw=(rk>=1-q).astype(float); lw=lw.div(lw.sum(axis=1).replace(0,np.nan),axis=0)
    frs=fac.where(LIQ5); rks=frs.rank(axis=1,pct=True)
    sw=(rks<=q).astype(float); sw=sw.div(sw.sum(axis=1).replace(0,np.nan),axis=0)   # short only liquid>=$5
    lr=(lw*fwd1).sum(axis=1).reindex(idx); sr=(sw*fwd1).sum(axis=1).reindex(idx)
    return lr,sr,lw
# combined via return-series of good factors
good=list(FAC.keys())
longs={}; ls_gross={}; ls_net={}
borrow=0.06/12   # 6%/yr small-cap borrow cost on short notional
for k in good:
    lr,sr,lw=legs(FAC[k]); longs[k]=lr; ls_gross[k]=lr-sr; ls_net[k]=lr-sr-borrow
L=pd.DataFrame(longs).reindex(idx); G=pd.DataFrame(ls_gross).reindex(idx); Nn=pd.DataFrame(ls_net).reindex(idx)
alpha_gross=G.mean(axis=1); alpha_net=Nn.mean(axis=1); longonly_factor=L.mean(axis=1)
p(f"{'strategy':42} {'ann%':>7} {'Sharpe':>7} {'maxDD':>7} {'corrQQQ':>8}")
for nm,r in [("QQQ",qret),
             ("multi-factor L/S GROSS",alpha_gross),
             ("multi-factor L/S NET (6%/yr borrow, $5 short)",alpha_net),
             ("LONG-ONLY factor tilt (top-quintile, no short/lev)",longonly_factor)]:
    a,s,d=ann(r); p(f"{nm:42} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {r.corr(qret):>8.2f}")
# deployable, no-leverage combos
p(f"\n--- DEPLOYABLE (no leverage, no shorting) ---")
for nm,r in [("50 QQQ / 50 long-only-factor",0.5*qret+0.5*longonly_factor),
             ("long-only factor tilt alone",longonly_factor)]:
    a,s,d=ann(r); p(f"{nm:42} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {r.corr(qret):>8.2f}")
# modest-leverage 130/30-style: long-only + 0.3x net alpha overlay
p(f"\n--- MODEST LEVERAGE (net of borrow) ---")
for k in [0.3,0.5,1.0]:
    r=qret+k*alpha_net; a,s,d=ann(r); p(f"{'QQQ + '+str(k)+'x NET alpha':42} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {r.corr(qret):>8.2f}")
r=0.5*qret+0.5*longonly_factor+0.5*alpha_net; a,s,d=ann(r); p(f"{'50QQQ/50LOfac + 0.5x NET alpha':42} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {r.corr(qret):>8.2f}")
# factor timing: weight by trailing-12m sharpe of each net L/S
p(f"\n--- FACTOR TIMING (weight by trailing 12m return) ---")
tw=Nn.rolling(12,min_periods=6).mean().clip(lower=0); tw=tw.div(tw.sum(axis=1),axis=0)
timed=(tw.shift(1)*Nn).sum(axis=1)
a,s,d=ann(timed); p(f"{'timed multi-factor L/S (net)':42} {a*100:>7.1f} {s:>7.2f} {d:>7.1%} {timed.corr(qret):>8.2f}")
# net-of-cost sub-periods
p(f"\nNET alpha sub-period Sharpe:")
for lo,hi in [("2012","2016"),("2017","2020"),("2021","2025")]:
    m=(idx>=pd.Timestamp(lo))&(idx<=pd.Timestamp(hi+"-12-31")); a,s,_=ann(alpha_net[m]); p(f"  {lo}-{hi}: ann {a*100:>5.1f}% Sharpe {s:.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
