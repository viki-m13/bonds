import urllib.request, ssl, os, io, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
K=os.environ.get("TIINGO_KEY","2897486ab136799678eda8c333ea343811bb0af4")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M); ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1
qpx=pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"]
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
# ---------- SLEEVE 1: WAVE (ML champion, ride+gates) ----------
PROB=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
def ride_sim(score_mask_fn,N=12,exit_dd=-0.30,trend=True,durab=None):
    rank,elig=score_mask_fn(); pos={}; cash=1.0; out=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx); ex=False
            if exit_dd is not None and cpx/e["peak"]-1<=exit_dd: ex=True
            if trend and cpx<ma10.loc[dt].get(tk,np.nan): ex=True
            if durab is not None and bool(durab.loc[dt].get(tk,False)): ex=True
            if ex: cash+=e["val"]; pos.pop(tk)
        rk=rank.loc[dt]; el=elig.loc[dt]
        cands=[t for t in rk[rk<=N*4].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan)) and bool(el.get(t,False))]
        for tk in cands:
            if len(pos)>=N: break
            if cash>1e-9: sl=cash/max(1,(N-len(pos))); pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":sl}; cash-=sl
        eq0=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx):
            for tk in pos:
                r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx): out.append((didx[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    return pd.Series(dict(out))
def wave_fn():
    a=PROB-PROB.shift(2); el=(liq&(me>=3.0)&(me>ma10)&(mom3>0)&(a>0)).fillna(False).astype(bool)
    return PROB.where(el).rank(axis=1,ascending=False), el
wave=ride_sim(wave_fn,N=12); p(f"WAVE done t={time.time()-t0:.0f}s")
# ---------- SLEEVE 2: COMPOUNDER (durable-survivor composite) ----------
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl"); rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def qi(df): df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=qi(rev)
def gq(k):
    d=F.get(k); return qi(d).reindex(columns=rev.columns) if d is not None else None
OI,NI,AST,CASH,STI,SH=[gq(k) for k in ["OperatingIncomeLoss","NetIncomeLoss","Assets","CashAndCashEquivalentsAtCarryingValue","ShortTermInvestments","EntityCommonStockSharesOutstanding"]]
def qm(df):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=6)
roic_m=qm((OI*4)/(AST-CASH.fillna(0)-STI.fillna(0)).clip(lower=1)); ryoy_m=qm(rev/rev.shift(4)-1); mcap=me*qm(SH); ep=qm(NI*4)/mcap
def z(x):
    fr=x.where(liq); return (fr.sub(fr.mean(axis=1),axis=0)).div(fr.std(axis=1).replace(0,np.nan),axis=0).fillna(0)
SC=(z(ep)+z(-FEAT["share_chg"])+z(roic_m)+z(-FEAT["vol6"])+z(-FEAT["log_mcap"])).where(liq)
durb=((roic_m.rank(axis=1,pct=True)<0.40).rolling(6,min_periods=4).min()>0.5)|(ryoy_m< -0.15)
def comp_fn(): return SC.rank(axis=1,ascending=False),(liq&(me>=3.0)).fillna(False)
comp=ride_sim(comp_fn,N=20,exit_dd=None,trend=False,durab=durb); p(f"COMPOUNDER done t={time.time()-t0:.0f}s")
# ---------- SLEEVE 3: CROSS-ASSET TREND ----------
def fetch(tk):
    u=f"https://api.tiingo.com/tiingo/daily/{tk}/prices?startDate=2008-01-01&token={K}&format=csv&resampleFreq=daily"
    d=urllib.request.urlopen(urllib.request.Request(u),timeout=60,context=ctx).read().decode()
    df=pd.read_csv(io.StringIO(d)); df["date"]=pd.to_datetime(df.date); return df.set_index("date")["adjClose"].astype(float)
ET=["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ","BIL"]; S={}
for t in ET:
    try: S[t]=fetch(t)
    except Exception as e: p(f"fail {t}")
ep_=pd.DataFrame(S).sort_index(); met=ep_.resample("ME").last(); met.index=met.index.to_period("M").to_timestamp()
Rt={t:met[t].pct_change() for t in met.columns}; momt=met/met.shift(12)-1; ma10t=met.rolling(10,min_periods=10).mean()
risk=["SPY","QQQ","IWM","EFA","EEM","GLD","DBC","VNQ","TLT","LQD","HYG"]; csh="BIL"
tr_out=[]
for i,dt in enumerate(met.index):
    if i+1>=len(met.index): continue
    elg=[t for t in risk if met[t].loc[dt]>ma10t[t].loc[dt] and momt[t].loc[dt]>0]
    elg=sorted(elg,key=lambda t:-momt[t].loc[dt])[:4]; nx=met.index[i+1]
    tr_out.append((nx, np.mean([Rt[t].loc[nx] for t in elg]) if elg else Rt[csh].loc[nx]))
trend=pd.Series(dict(tr_out)); p(f"TREND done t={time.time()-t0:.0f}s")
# ---------- ASSEMBLE ----------
idx=M[(M>=pd.Timestamp("2015-01-01"))&(M<=pd.Timestamp("2025-12-31"))]
qret=qpx.pct_change().reindex(idx)
S=pd.DataFrame({"WAVE":wave,"Compounder":comp,"Trend":trend,"QQQ":qret}).reindex(idx).dropna(how="all").fillna(0.0)
p(f"\nSleeve stats (2015-2025):")
p(f"{'sleeve':12}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}")
for c in S.columns:
    cc,ss,dd=stats(S[c]); p(f"{c:12}{cc:>7.1%}{ss:>7.2f}{dd:>7.1%}")
p(f"\ncorrelation matrix:"); cm=S.corr(); p(cm.round(2).to_string())
# risk-parity (inverse-vol, rolling 12m, monthly)
iv=1.0/S.rolling(12,min_periods=6).std(); w=iv.div(iv.sum(axis=1),axis=0)
rp_ret=(w.shift(1)*S).sum(axis=1)
# static equal-risk (full-sample inverse vol) for headline weights
sv=1.0/S.std(); wstat=sv/sv.sum()
p(f"\nstatic risk-parity weights: "+", ".join(f"{c} {wstat[c]:.0%}" for c in S.columns))
p(f"\n{'portfolio':34}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'corrQQQ':>8}")
c,s,d=stats(qret); p(f"{'QQQ (benchmark)':34}{c:>7.1%}{s:>7.2f}{d:>7.1%}{1.0:>8.2f}")
ports={"risk-parity (rolling inv-vol)":rp_ret,
       "static inv-vol":(wstat*S).sum(axis=1),
       "equal-weight 4 sleeves":S.mean(axis=1),
       "no-QQQ rp (WAVE+Comp+Trend)":None}
ivx=1.0/S[["WAVE","Compounder","Trend"]].rolling(12,min_periods=6).std(); wx=ivx.div(ivx.sum(axis=1),axis=0)
ports["no-QQQ rp (WAVE+Comp+Trend)"]=(wx.shift(1)*S[["WAVE","Compounder","Trend"]]).sum(axis=1)
best=None
for nm,r in ports.items():
    r=r.reindex(idx).dropna(); c,s,d=stats(r); p(f"{nm:34}{c:>7.1%}{s:>7.2f}{d:>7.1%}{r.corr(qret.reindex(r.index)):>8.2f}")
    if best is None or s>best: best=s;bnm=nm;bser=r
# vol-target the best to 14%
tgt=0.14; rv=bser.rolling(12,min_periods=6).std()*np.sqrt(12); lev=(tgt/rv).clip(0.3,2.0).shift(1); vt=(lev*bser).dropna()
c,s,d=stats(vt); p(f"{'best + vol-target 14% (lev<=2)':34}{c:>7.1%}{s:>7.2f}{d:>7.1%}{vt.corr(qret.reindex(vt.index)):>8.2f}")
p(f"\nbest blend: {bnm} (Sharpe {best:.2f})")
# DCA $1k/mo: portfolio vs QQQ
def dcaS(r,cc=1000.0):
    v=0.0; o=[]
    for x in r.fillna(0): v=v*(1+x)+cc; o.append(v)
    return pd.Series(o,index=r.index)
bd=dcaS(bser); qd=dcaS(qret.reindex(bser.index)); contrib=np.arange(1,len(bser)+1)*1000
p(f"\nDCA $1k/mo: portfolio ${bd.iloc[-1]:,.0f} vs QQQ ${qd.iloc[-1]:,.0f} (contributed ${contrib[-1]:,.0f})")
# figure
fig,ax=plt.subplots(1,2,figsize=(15,6))
for c,co,lw in [("WAVE","#1f77b4",1),("Compounder","#2ca02c",1),("Trend","#9467bd",1),("QQQ","#888",1.6)]:
    g=(1+S[c]).cumprod(); ax[0].plot(g.index,g,label=c,lw=lw,color=co,alpha=0.8)
gb=(1+bser).cumprod(); ax[0].plot(gb.index,gb,label=f"BLEND ({bnm.split('(')[0].strip()})",lw=2.6,color="#d62728")
cc,ss,dd=stats(bser); qc,qs,qd2=stats(qret)
ax[0].set_yscale("log"); ax[0].set_title(f"Sleeves + diversified blend vs QQQ\nblend CAGR {cc:.0%}/Sh {ss:.2f}/DD {dd:.0%} vs QQQ {qc:.0%}/{qs:.2f}/{qd2:.0%}")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(bd.index,bd/1e3,label=f"Diversified blend DCA → ${bd.iloc[-1]/1e3:.0f}k",lw=2.4,color="#d62728")
ax[1].plot(qd.index,qd/1e3,label=f"QQQ DCA → ${qd.iloc[-1]/1e3:.0f}k",lw=2,color="#888")
ax[1].plot(bser.index,contrib/1e3,label=f"Contributed → ${contrib[-1]/1e3:.0f}k",lw=1.2,ls="--",color="#bbb")
ax[1].set_title("$1,000/mo DCA — account value ($k)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
fig.suptitle("Diversified multi-sleeve portfolio (WAVE + Compounder + Trend + QQQ), risk-parity, 2015-2025",fontsize=12)
fig.tight_layout(); fig.savefig("/home/user/portfolio_blend.png",dpi=110)
S.to_pickle("/tmp/wave/_sleeves.pkl")
p(f"\nsaved portfolio_blend.png  DONE t={time.time()-t0:.0f}s")
