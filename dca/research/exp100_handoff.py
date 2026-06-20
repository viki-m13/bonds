import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,liq,me,cols=D["FEAT"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M)
F=pd.read_pickle("/home/user/bonds/dca/research/data/sec/sec_fundamentals.pkl")
rev=pd.read_parquet("/home/user/bonds/dca/research/data/sec/sec_revenue_quarterly.parquet")
def qidx(df):
    df=df.copy(); df.index=pd.PeriodIndex([q[2:] for q in df.index],freq="Q").to_timestamp(how="end").normalize(); return df
rev=qidx(rev)
def gq(k):
    d=F.get(k); return qidx(d).reindex(columns=rev.columns) if d is not None else None
OI,NI,AST,CASH,STI,SH=[gq(k) for k in ["OperatingIncomeLoss","NetIncomeLoss","Assets","CashAndCashEquivalentsAtCarryingValue","ShortTermInvestments","EntityCommonStockSharesOutstanding"]]
def qm(df,lim=6):
    df=df.reindex(columns=cols); av=(df.index+pd.DateOffset(days=80)).to_period("M").to_timestamp()
    d2=df.copy(); d2.index=av; d2=d2[~d2.index.duplicated(keep="last")]; return d2.reindex(M,method="ffill",limit=lim)
# --- growth deceleration (past peak rate, still positive) ---
ryoy=rev/rev.shift(4)-1
decel=((ryoy>0.0)&(ryoy<ryoy.shift(4)))            # growing but rate falling vs yr ago
ryoy_m=qm(ryoy); decel_m=qm(decel.astype(float))>0.5
# --- ROIC proxy = operating income / invested capital (assets - cash - ST inv) ---
inv_cap=(AST-CASH.fillna(0)-STI.fillna(0)).clip(lower=1)
roic=(OI*4)/inv_cap
roic_stable=(roic.rolling(6,min_periods=4).min()>0.05)        # consistently >5% (durable)
roic_m=qm(roic); roic_stable_m=qm(roic_stable.astype(float))>0.5
roic_rank=roic_m.rank(axis=1,pct=True)
# --- valuation: P/S, not re-rated (current <= trailing 18mo median) ---
mcap=me*qm(SH); ps=mcap/qm(rev*4).clip(lower=1)
ps_med=ps.rolling(18,min_periods=9).median(); not_rerated=(ps<=ps_med*1.05)
# --- momentum decay (growth investors rotating out): 12m mom positive history but now fading ---
mom12=me/me.shift(12)-1; mom6=me/me.shift(6)-1
mom_decay=((mom12.shift(6)>0.10)&(mom6<0.05))                 # was strong 6m ago, now flat/soft
liqf=(me.shift(1)>=3.0).fillna(False)
# === HANDOFF-GAP entry mask ===
HANDOFF=( decel_m & roic_stable_m & (roic_m.rank(axis=1,pct=True)>0.6) & not_rerated & mom_decay & liqf ).fillna(False)
# control cohorts
ACCEL=( (ryoy_m>0)&(ryoy_m>ryoy_m.shift(4)) & (roic_m.rank(axis=1,pct=True)>0.6) & liqf ).fillna(False)  # accelerating growth + quality
idx=M[(M>=pd.Timestamp("2012-01-01"))&(M<=pd.Timestamp("2024-06-30"))]
fwd12=(me.shift(-12)/me-1).clip(-0.95,5.0); fwd24=(me.shift(-24)/me-1).clip(-0.95,9.0)
univ=fwd12.where(liqf)
p(f"=== ENTRY-ALPHA (cross-sectional, no exit logic) ===")
p(f"avg handoff names/mo: {HANDOFF.loc[idx].sum(axis=1).mean():.0f}   accel names/mo: {ACCEL.loc[idx].sum(axis=1).mean():.0f}")
def coh(mask,fwd):
    vals=fwd.where(mask).loc[idx].stack(); return vals.mean(),vals.median(),(vals>0).mean(),(vals>1.0).mean(),(vals>2.0).mean()
for nm,mask in [("HANDOFF-GAP",HANDOFF),("accel-growth+quality",ACCEL),("universe",liqf)]:
    mean12,med12,hit,mb,mb2=coh(mask,fwd12)
    m24=fwd24.where(mask).loc[idx].stack().mean()
    p(f"  {nm:22} fwd12 mean {mean12:+.1%} med {med12:+.1%} hit {hit:.0%} >100% {mb:.1%} >200% {mb2:.1%} | fwd24 mean {m24:+.1%}")
# === PORTFOLIO SIM with 3 exit rules — separate entry vs hold-through-drawdown alpha ===
ret=(me/me.shift(1)-1).clip(-0.9,3.0)
qret=(pd.read_pickle("/tmp/wave/_tiingo_me.pkl")["QQQ"].pct_change()).reindex(idx)
def stats(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); d=(eq/eq.cummax()-1).min(); return c,s,d
def sim(exit_rule,maxhold=60,N=20):
    pos={}; cash=1.0; out=[]; trades=[]; dd_held=[]
    for k,dt in enumerate(didx):
        px=me.loc[dt]
        for tk in list(pos.keys()):
            e=pos[tk]; cpx=px.get(tk,np.nan)
            if not np.isfinite(cpx): trades.append(-0.7); pos.pop(tk); continue
            e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1; e["mdd"]=min(e["mdd"],cpx/e["peak"]-1); age=k-e["i"]; ex=False
            if exit_rule=="durability":
                # exit only when ROIC durability breaks (roic rank<0.4 or roic<0) or revenue turns down hard
                rr=roic_rank.loc[dt].get(tk,0.5); rc=roic_m.loc[dt].get(tk,0.1); ry=ryoy_m.loc[dt].get(tk,0.0)
                if (rr<0.4) or (rc<0.0) or (ry< -0.10): ex=True
            elif exit_rule=="price_stop":
                if cpx/e["peak"]-1<=-0.20: ex=True
            elif exit_rule=="fixed":
                pass
            if age>=maxhold: ex=True
            if ex: trades.append(rs); dd_held.append(e["mdd"]); cash+=e["val"]; pos.pop(tk)
        ent=HANDOFF.loc[dt]; cands=[t for t in ent.index[ent.values] if t not in pos and np.isfinite(px.get(t,np.nan))]
        cands=sorted(cands,key=lambda t:-(roic_m.loc[dt].get(t,0)))
        for tk in cands:
            if len(pos)>=N: break
            if cash>1e-9: sl=cash/max(1,(N-len(pos))); pos[tk]={"i":k,"px":px[tk],"peak":px[tk],"val":sl,"mdd":0.0}; cash-=sl
        eq0=cash+sum(e["val"] for e in pos.values())
        if k+1<len(didx):
            for tk in pos:
                r1=ret.iloc[k+1].get(tk,np.nan); pos[tk]["val"]*=(1+(r1 if np.isfinite(r1) else -0.5))
        eq1=cash+sum(e["val"] for e in pos.values())
        if dt>=idx[0] and dt<=idx[-1] and k+1<len(didx): out.append((didx[k+1],eq1/eq0-1 if eq0>0 else 0.0))
    s=pd.Series(dict(out)).reindex(idx).fillna(0.0); tr=np.array(trades) if trades else np.array([0.0])
    return s,tr,np.array(dd_held) if dd_held else np.array([0.0])
p(f"\n=== EXIT-RULE DECOMPOSITION (handoff entries, same names, different exits) ===")
p(f"{'exit rule':16}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'win%':>6}{'avgTr':>7}{'>100%':>7}{'medDDheld':>10}")
c,s,d=stats(qret); p(f"{'QQQ':16}{c:>7.1%}{s:>7.2f}{d:>7.1%}")
for er in ["price_stop","fixed","durability"]:
    sr,tr,dh=sim(er); c,s,d=stats(sr)
    p(f"{er:16}{c:>7.1%}{s:>7.2f}{d:>7.1%}{(tr>0).mean():>6.0%}{tr.mean():>+7.1%}{(tr>1.0).mean():>7.1%}{np.median(dh):>+10.0%}")
p(f"\n(hold-through-drawdown alpha = durability minus price_stop CAGR/Sharpe; medDDheld shows the drawdowns winners rode through)")
p(f"DONE t={time.time()-t0:.0f}s")
