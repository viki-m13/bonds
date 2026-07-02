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
q=pd.Series({pd.Timestamp(x['date'][:10]):x['adjClose'] for x in json.loads(urllib.request.urlopen('https://api.tiingo.com/tiingo/daily/QQQ/prices?startDate=2000-06-01&token='+os.environ['TIINGO']+'&format=json',timeout=40).read())})
qd=q.reindex(dates).ffill(); regime=(qd>qd.rolling(200,min_periods=200).mean())   # market above 200DMA
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
    return (eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1, r.mean()/r.std()*np.sqrt(252), (eq/eq.cummax()-1).min(), (tr>0).mean(), len(tr)
E={"base (ever-member)": (base_uni&trig3, False, None),
   "+ market regime (QQQ>200DMA)": (base_uni&trig3, True, None),
   "+ RSI2<10 dip-quality": (base_uni&trig3&(rsi2<10), False, None),
   "+ IBS<0.3 (weak close)": (base_uni&trig3&(ibs<0.3), False, None),
   "+ regime + RSI2<10": (base_uni&trig3&(rsi2<10), True, None),
   "+ regime + RSI2 + IBS<0.3": (base_uni&trig3&(rsi2<10)&(ibs<0.3), True, None),
   "+ regime + RSI2 + vol-target 20%": (base_uni&trig3&(rsi2<10), True, 0.20)}
print(f"{'variant':38}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'winR':>6}{'nTrd':>7}")
for nm,(mask,reg,vt) in E.items():
    c,s,dd,wr,n=run(mask,use_regime=reg,voltgt=vt); print(f"  {nm:36}{c*100:>6.1f}%{s:>7.2f}{dd*100:>6.0f}%{wr*100:>5.0f}%{n:>7}",flush=True)
print(f"DONE t={time.time()-t0:.0f}s")
