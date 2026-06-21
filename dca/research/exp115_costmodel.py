import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; ret=(me/me.shift(1)-1).clip(-0.9,2.0)
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(M)
LIQ=(liq&(me>=3.0)).fillna(False); SHORT=(liq&(me>=10.0)).fillna(False)
# --- market cap (me x shares from SEC) ---
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def qidx(df):
    df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=qidx(rev)
SH=qidx(F["EntityCommonStockSharesOutstanding"]).reindex(columns=rev.columns)
def qmap(df):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=6)
mcap=(me*qmap(SH))                       # $ market cap
mcap=mcap.where(mcap>0)
lmrank=np.log(mcap).rank(axis=1,pct=True)
# --- dollar ADV proxy: mcap x daily turnover (calibrated 0.5%/day, conservative) ---
ADV=mcap*0.005
# --- daily vol proxy for impact (annualized vol6 -> daily) ---
volA=FEAT["vol6"]                         # treat as ~annualized; daily = /sqrt(252)
sig_d=(volA/np.sqrt(252)).clip(0.005,0.20)
# --- tiered half-spread (bps) by mcap rank: bigger=tighter ---
spread=pd.DataFrame(40.0,index=M,columns=cols)        # default micro
spread=spread.where(lmrank<0.5,20.0).where(lmrank<0.8,8.0).where(lmrank<0.95,4.0)
# --- tiered borrow (annual) ---
borrow_rate=pd.DataFrame(0.06,index=M,columns=cols).where(lmrank<0.8,0.02).where(lmrank<0.95,0.01)
# trailing-12m beta
q=qret.reindex(M); qm=q.rolling(12,min_periods=8).mean(); qv=q.rolling(12,min_periods=8).var()
rq=ret.mul(q,axis=0); cov=rq.rolling(12,min_periods=8).mean().sub(ret.rolling(12,min_periods=8).mean().mul(qm,axis=0),axis=0)
BETA=cov.div(qv,axis=0).clip(-3,3).fillna(1.0)
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(index=M,columns=cols)
DEV=("2015-01-01","2022-12-31"); HOLD=("2023-01-01","2025-12-31"); FULL=("2015-01-01","2025-12-31")
def ann(r,sub=None):
    r=r.dropna()
    if sub: r=r[(r.index>=pd.Timestamp(sub[0]))&(r.index<=pd.Timestamp(sub[1]))]
    if len(r)<6: return (np.nan,)*3
    a=r.mean()*12; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return a,s,d
def build():
    rebal,buffer,qq=3,2.0,0.1
    rkl=PROB.where(LIQ).rank(axis=1,pct=True); rks=PROB.where(SHORT).rank(axis=1,pct=True)
    Wl=pd.DataFrame(0.0,index=M,columns=cols); Ws=pd.DataFrame(0.0,index=M,columns=cols)
    hl=set(); hs=set(); cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
    for k,dt in enumerate(M):
        if k%rebal==0:
            rl=rkl.loc[dt]; rs=rks.loc[dt]
            sel_l=[t for t in hl if rl.get(t,0)>=1-qq*buffer]; nt=int((rl>=1-qq).sum())
            for t in rl[rl>=1-qq].sort_values(ascending=False).index:
                if len(sel_l)>=nt: break
                if t not in sel_l: sel_l.append(t)
            sel_s=[t for t in hs if rs.get(t,1)<=qq*buffer]; st=int((rs<=qq).sum())
            for t in rs[rs<=qq].sort_values().index:
                if len(sel_s)>=st: break
                if t not in sel_s: sel_s.append(t)
            hl=set(sel_l); hs=set(sel_s)
            cl=pd.Series(0.0,index=cols); cs=pd.Series(0.0,index=cols)
            if sel_l: cl[sel_l]=1.0/len(sel_l)
            if sel_s: cs[sel_s]=1.0/len(sel_s)
        b=BETA.loc[dt]; bl=(cl*b).sum(); bs=(cs*b).sum(); css=cs*(bl/bs) if bs>0.05 else cs
        Wl.loc[dt]=cl; Ws.loc[dt]=css
    return Wl,Ws
Wl,Ws=build()
gross=((Wl.shift(1)*ret).sum(axis=1)-(Ws.shift(1)*ret).sum(axis=1)).reindex(idx)
dWl=Wl.diff().abs(); dWs=Ws.diff().abs(); traded=(dWl+dWs)            # |Δweight| per name
# cost components (as return drag, since traded$/AUM = Δweight)
spread_cost=(traded*spread/1e4).sum(axis=1).reindex(idx)
def impact_cost(AUM,exec_days=3,K=0.8):
    part=(traded*AUM)/(ADV*exec_days)                                # participation per name
    imp_bps=K*sig_d*np.sqrt(part.clip(lower=0))*1e4
    return (traded*imp_bps/1e4).sum(axis=1).reindex(idx)
borrow_cost=(Ws.shift(1)*borrow_rate/12).sum(axis=1).reindex(idx)
fin_cost=((Wl.shift(1).sum(axis=1)+Ws.shift(1).sum(axis=1))*0.005/12).reindex(idx)  # 50bps/yr financing on gross
p(f"=== v3 COST WATERFALL (beta-neutral; avg gross {(Wl.sum(axis=1)+Ws.sum(axis=1)).reindex(idx).mean():.2f}, avg turnover {traded.sum(axis=1).reindex(idx).mean():.0%}/mo) ===")
def show(nm,r): a,s,d=ann(r,FULL); _,sD,_=ann(r,DEV); _,sH,_=ann(r,HOLD); p(f"{nm:38}{a*100:>6.1f}{s:>6.2f}{d:>7.1%} |{sD:>6.2f}{sH:>6.2f}")
p(f"{'stage':38}{'ann%':>6}{'Sh':>6}{'maxDD':>7} |{'devSh':>6}{'hldSh':>6}")
show("gross (no costs)",gross)
n=gross-spread_cost; show("- tiered spread",n)
n=n-borrow_cost; show("- tiered borrow",n)
n=n-fin_cost; show("- financing 50bps/yr",n)
for aum in [10e6,50e6,100e6]:
    nn=n-impact_cost(aum); show(f"- impact @ ${aum/1e6:.0f}M AUM (NET)",nn)
# delisting stress: held names that vanish next month get penalized (long -30%, short -20% merger premium)
nextvanish=(me.notna())&(~me.shift(-1).notna())     # last valid month
delist_long=(Wl.shift(1)*nextvanish.shift(1)*(-0.30)).sum(axis=1).reindex(idx)   # extra loss on delisted longs
delist_short=(Ws.shift(1)*nextvanish.shift(1)*(-0.20)).sum(axis=1).reindex(idx)  # shorts lose on buyout premium
net100=n-impact_cost(100e6)
show("NET @100M + delisting stress",net100+delist_long-delist_short)
# CAPACITY curve
p(f"\n=== CAPACITY (NET Sharpe & CAGR vs AUM; full sample) ===")
base=gross-spread_cost-borrow_cost-fin_cost
for aum in [1e6,10e6,50e6,100e6,250e6,500e6,1e9]:
    r=base-impact_cost(aum); a,s,d=ann(r,FULL)
    medpart=((traded*aum)/(ADV*3)).replace(0,np.nan).reindex(idx).stack().median()
    p(f"  ${aum/1e6:>5.0f}M: NET ann {a*100:>5.1f}%  Sharpe {s:.2f}  maxDD {d:.1%}  median participation {medpart*100:.1f}%/3d")
pd.to_pickle({"Wl":Wl,"Ws":Ws,"base":base,"impact_cost":None},"/tmp/wave/_costmodel.pkl")
p(f"\nDONE t={time.time()-t0:.0f}s")
