import pandas as pd, numpy as np, warnings, time, os, urllib.request, json; warnings.filterwarnings("ignore")
t0=time.time()
cur=pd.read_pickle("/tmp/ndx_daily.pkl"); rem=pd.read_pickle("/tmp/ndx_removed_daily.pkl"); data={**cur,**rem}
tk=[t for t,df in data.items() if len(df)>250]
dates=sorted(set().union(*[set(data[t].index) for t in tk])); dates=[d for d in dates if d>=pd.Timestamp("2001-01-01")]
def panel(f): return pd.DataFrame({t:data[t][f] for t in tk}).reindex(dates)
O,H,L,C,V=[panel(x) for x in ["o","h","l","c","v"]]
prevC=C.shift(1); prevH=H.shift(1)
tr=pd.concat([H-L,(H-prevC).abs(),(L-prevC).abs()]).groupby(level=0).max()
atr=tr.rolling(5,min_periods=5).mean(); atrp=atr/C
ema200=C.ewm(span=200,min_periods=200).mean(); vol20=V.rolling(20,min_periods=20).mean()
# RSI(2) Wilder, IBS
d=C.diff(); ru=d.clip(lower=0).ewm(alpha=1/2,min_periods=2).mean(); rd=(-d).clip(lower=0).ewm(alpha=1/2,min_periods=2).mean()
rsi2=100-100/(1+ru/rd.replace(0,np.nan)); ibs=(C-L)/(H-L).replace(0,np.nan)
base_uni=(C>5)&(vol20>100000)&(C>ema200)&(atrp>0.03)
trig3=(C<prevC*0.97)                                 # 3% down close
# QQQ regime
regime=pd.Series(True,index=dates)
Cv,Ov,Hv,Lv,PHv,ATRv,ATRPv=C.values,O.values,H.values,L.values,prevH.values,atr.values,atrp.values
col={t:j for j,t in enumerate(tk)}; regv=regime.values
def run(entry_mask, use_regime=False, voltgt=None, pos_frac=0.20, maxpos=10, comm=0.005, E0=1e5):
    em=entry_mask.values
    cash=E0; pos={}; eq=[]; trades=[]; pend=[]
    # portfolio trailing vol for vol-target
    rets=[]
    for i,dd in enumerate(dates):
        for t in list(pos.keys()):
            p=pos[t]; j=col[t]; hi,lo,cl,ph=Hv[i,j],Lv[i,j],Cv[i,j],PHv[i,j]
            if not np.isfinite(cl): continue
            ex=px=None
            if np.isfinite(hi) and hi>=p["tgt"]: ex,px="t",p["tgt"]
            elif np.isfinite(ph) and cl>ph: ex,px="c",cl
            elif p["bars"]>=9: ex,px="9",cl
            if ex: cash+=p["sh"]*px-p["sh"]*comm; trades.append(px/p["ep"]-1); pos.pop(t); continue
            p["bars"]+=1
        allow = (not use_regime) or bool(regv[i])
        pf=pos_frac
        if voltgt is not None and len(eq)>20:
            rv=np.std(rets[-20:])*np.sqrt(252) if len(rets)>=20 else voltgt
            pf=pos_frac*min(2.0, voltgt/max(rv,1e-3))
        if allow:
            for t,lim,tgt in pend:
                if t in pos or len(pos)>=maxpos: continue
                j=col[t]; op,lo=Ov[i,j],Lv[i,j]
                if not np.isfinite(lo) or lo>lim: continue
                fill=min(op,lim) if np.isfinite(op) else lim
                eqty=cash+sum(pp["sh"]*Cv[i,col[tt]] for tt,pp in pos.items() if np.isfinite(Cv[i,col[tt]]))
                sh=int((pf*eqty)//fill)
                if sh<=0: continue
                cash-=sh*fill+sh*comm; pos[t]={"sh":sh,"ep":fill,"tgt":tgt,"bars":0}
        mv=sum(pp["sh"]*Cv[i,col[tt]] for tt,pp in pos.items() if np.isfinite(Cv[i,col[tt]]))
        tot=cash+mv; eq.append(tot)
        if len(eq)>1: rets.append(eq[-1]/eq[-2]-1)
        cand=sorted([(ATRPv[i,col[t]],t) for t in tk if em[i,col[t]] and np.isfinite(ATRPv[i,col[t]])],reverse=True)
        pend=[(t,Cv[i,col[t]]-0.9*ATRv[i,col[t]],Cv[i,col[t]]+0.5*ATRv[i,col[t]]) for _,t in cand[:20]]
    eq=pd.Series(eq,index=dates); r=eq.pct_change().dropna(); yrs=(eq.index[-1]-eq.index[0]).days/365.25
    tr=np.array(trades)
    return eq
import pandas as pd, numpy as np
# --- MR sleeve (improved: RSI2<15, unit book pf20) daily equity -> monthly returns ---
mr_eq=run(base_uni&trig3&(rsi2<15), use_regime=False, voltgt=None, pos_frac=0.20, maxpos=10)
mr_m=mr_eq.resample("ME").last().pct_change().dropna()
# --- Momentum/trend sleeve: 12-1 top-20 EW monthly on S&P daily names ---
sp=pd.read_pickle("/tmp/sp_daily.pkl"); stk=[t for t,df in sp.items() if len(df)>300]
Cm=pd.DataFrame({t:sp[t]["c"] for t in stk}).resample("ME").last()
mom=Cm.shift(1)/Cm.shift(12)-1; retm=Cm.pct_change()
rk=mom.rank(axis=1,ascending=False); w=(rk<=20).astype(float); w=w.div(w.sum(axis=1),axis=0)
momo=(w.shift(1)*retm).sum(axis=1).dropna()
# --- Buy-the-dip sleeve: RSI5<20 & >200DMA, 5d hold, EW active, daily->monthly (S&P) ---
def rsi(px,n):
    d=px.diff(); ru=d.clip(lower=0).ewm(alpha=1/n,min_periods=n).mean(); rd=(-d).clip(lower=0).ewm(alpha=1/n,min_periods=n).mean()
    return 100-100/(1+ru/rd.replace(0,np.nan))
Cd=pd.DataFrame({t:sp[t]["c"] for t in stk}); R5=rsi(Cd,5); MA2=Cd.rolling(200,min_periods=200).mean()
ent=((R5<20)&(Cd>MA2)).fillna(False); f5=Cd.shift(-5)/Cd-1
# daily portfolio return proxy: each day, avg fwd-1d of names entered in last 5d (approx) -> use monthly avg of entry fwd5/5
btd_daily=(f5.where(ent).mean(axis=1)/5).fillna(0)   # spread the 5d return over ~holding
btd_m=(1+btd_daily).resample("ME").prod()-1; btd_m=btd_m.reindex(mr_m.index).fillna(0)
# align
idx=mr_m.index.intersection(momo.index)
S=pd.DataFrame({"MeanRev(improved)":mr_m.reindex(idx),"Momentum(12-1)":momo.reindex(idx),"BuyDip":btd_m.reindex(idx)}).dropna()
def stat(r): 
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12); eq=(1+r).cumprod(); return c,s,(eq/eq.cummax()-1).min()
print("correlation matrix:\n",S.corr().round(2).to_string())
print(f"\n{'sleeve / combo':30}{'CAGR':>7}{'Sharpe':>8}{'maxDD':>7}")
for c in S.columns:
    a,b,d=stat(S[c]); print(f"  {c:28}{a*100:>6.1f}%{b:>8.2f}{d*100:>6.0f}%")
# equal-vol (risk-parity) combine
iv=1/S.std(); wv=iv/iv.sum(); combo=(S*wv).sum(axis=1)
a,b,d=stat(combo); print(f"  {'EQUAL-VOL COMBO':28}{a*100:>6.1f}%{b:>8.2f}{d*100:>6.0f}%")
# 60% improved-MR / 40% momentum
c2=0.6*S["MeanRev(improved)"]+0.4*S["Momentum(12-1)"]; a,b,d=stat(c2); print(f"  {'60% MeanRev / 40% Momentum':28}{a*100:>6.1f}%{b:>8.2f}{d*100:>6.0f}%")
# ---- iteration 4: holdout validation + equity curve ----
import os,urllib.request,json,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def stat2(r):
    r=r.dropna(); c=(1+r).prod()**(12/len(r))-1; s=r.mean()/r.std()*np.sqrt(12); eq=(1+r).cumprod(); return c,s,(eq/eq.cummax()-1).min()
iv=1/S.std(); wv=iv/iv.sum(); combo=(S*wv).sum(axis=1)
dev=S.index<pd.Timestamp("2016-01-01"); hold=~dev
print("\n=== DEV (pre-2016) / HOLDOUT (2016+) Sharpe — is the combo robust OOS? ===")
print(f"{'':28}{'dev Sh':>8}{'hold Sh':>9}")
for c in S.columns: print(f"  {c:26}{stat2(S[c][dev])[1]:>8.2f}{stat2(S[c][hold])[1]:>9.2f}")
print(f"  {'EQUAL-VOL COMBO':26}{stat2(combo[dev])[1]:>8.2f}{stat2(combo[hold])[1]:>9.2f}")
# equity curves vs QQQ
try:
    q=pd.Series({pd.Timestamp(x['date'][:10]):x['adjClose'] for x in json.loads(urllib.request.urlopen('https://api.tiingo.com/tiingo/daily/QQQ/prices?startDate=2001-01-01&token='+os.environ['TIINGO']+'&format=json',timeout=60).read())}).resample("ME").last().pct_change().reindex(S.index).fillna(0)
except Exception: q=pd.Series(0,index=S.index)
fig,ax=plt.subplots(figsize=(11,5.5))
for c,col_,w in [("MeanRev(improved)","#15803d",1.3),("Momentum(12-1)","#b91c1c",1.3),("BuyDip","#a855f7",1.3)]:
    g=(1+S[c]).cumprod(); ax.plot(g.index,g,color=col_,lw=1.2,alpha=.7,label=f"{c} (Sh {stat2(S[c])[1]:.2f})")
g=(1+combo).cumprod(); ax.plot(g.index,g,color="#111",lw=2.6,label=f"PROPRIETARY equal-vol combo (Sh {stat2(combo)[1]:.2f}, CAGR {stat2(combo)[0]*100:.0f}%, DD {stat2(combo)[2]*100:.0f}%)")
gq=(1+q).cumprod(); ax.plot(gq.index,gq,color="#2563eb",lw=1.6,ls="--",label=f"QQQ (Sh {stat2(q)[1]:.2f})")
ax.set_yscale("log"); ax.grid(alpha=.3,which="both"); ax.legend(loc="upper left",fontsize=8)
ax.set_title("Proprietary combined system: risk-parity blend of uncorrelated sleeves (monthly, 2001-2026)")
fig.tight_layout(); fig.savefig("/home/user/combo.png",dpi=120); print("\nsaved /home/user/combo.png")
