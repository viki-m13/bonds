import pandas as pd, numpy as np, warnings, time; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time()
cur=pd.read_pickle("/tmp/ndx_daily.pkl"); rem=pd.read_pickle("/tmp/ndx_removed_daily.pkl")
def build(data):
    tk=[t for t,df in data.items() if len(df)>250]
    dates=sorted(set().union(*[set(data[t].index) for t in tk])); dates=[d for d in dates if d>=pd.Timestamp("2001-01-01")]
    def panel(f): return pd.DataFrame({t:data[t][f] for t in tk}).reindex(dates)
    O,H,L,C,V=[panel(x) for x in ["o","h","l","c","v"]]
    prevC=C.shift(1); prevH=H.shift(1)
    tr=pd.concat([H-L,(H-prevC).abs(),(L-prevC).abs()]).groupby(level=0).max()
    atr=tr.rolling(5,min_periods=5).mean(); atrp=atr/C
    ema200=C.ewm(span=200,min_periods=200).mean(); vol20=V.rolling(20,min_periods=20).mean()
    uni=(C>5)&(vol20>100000)&(C>ema200)&(atrp>0.03); trig=(C<prevC*0.97)&uni
    return dict(tk=tk,dates=dates,O=O.values,H=H.values,L=L.values,C=C.values,PH=prevH.values,ATR=atr.values,ATRP=atrp.values,TRIG=trig.values,col={t:j for j,t in enumerate(tk)})
def run(P,pos_frac=0.20,maxpos=10,comm=0.005,E0=100000.0):
    tk,dates,O,H,L,C,PH,ATR,ATRP,TRIG,col=P["tk"],P["dates"],P["O"],P["H"],P["L"],P["C"],P["PH"],P["ATR"],P["ATRP"],P["TRIG"],P["col"]
    cash=E0; pos={}; eq=[]; trades=[]; pend=[]
    for i,d in enumerate(dates):
        for t in list(pos.keys()):
            p=pos[t]; j=col[t]; hi,lo,cl,ph=H[i,j],L[i,j],C[i,j],PH[i,j]
            if not np.isfinite(cl): continue
            ex=px=None
            if np.isfinite(hi) and hi>=p["tgt"]: ex,px="t",p["tgt"]
            elif np.isfinite(ph) and cl>ph: ex,px="c",cl
            elif p["bars"]>=9: ex,px="9",cl
            if ex: cash+=p["sh"]*px-p["sh"]*comm; trades.append(px/p["ep"]-1); pos.pop(t); continue
            p["bars"]+=1
        for t,lim,tgt in pend:
            if t in pos or len(pos)>=maxpos: continue
            j=col[t]; op,lo=O[i,j],L[i,j]
            if not np.isfinite(lo) or lo>lim: continue
            fill=min(op,lim) if np.isfinite(op) else lim
            eqty=cash+sum(pp["sh"]*C[i,col[tt]] for tt,pp in pos.items() if np.isfinite(C[i,col[tt]]))
            sh=int((pos_frac*eqty)//fill)
            if sh<=0: continue
            cash-=sh*fill+sh*comm; pos[t]={"sh":sh,"ep":fill,"tgt":tgt,"bars":0}
        mv=sum(pp["sh"]*C[i,col[tt]] for tt,pp in pos.items() if np.isfinite(C[i,col[tt]]))
        eq.append(cash+mv)
        cand=sorted([(ATRP[i,col[t]],t) for t in tk if TRIG[i,col[t]] and np.isfinite(ATRP[i,col[t]])],reverse=True)
        pend=[(t,C[i,col[t]]-0.9*ATR[i,col[t]],C[i,col[t]]+0.5*ATR[i,col[t]]) for _,t in cand[:20]]
    eq=pd.Series(eq,index=dates); tr=np.array(trades)
    r=eq.pct_change().dropna(); yrs=(eq.index[-1]-eq.index[0]).days/365.25
    return eq,((eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1, r.mean()/r.std()*np.sqrt(252),(eq/eq.cummax()-1).min(),(tr>0).mean(),len(tr))
ever={**cur,**rem}
Pc=build(cur); Pe=build(ever)
eqc,sc=run(Pc); eqe,se=run(Pe)
print(f"{'universe':32}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'winR':>6}{'nTrd':>7}")
print(f"  {'current NDX-100 (survivor-biased)':32}{sc[0]*100:>6.1f}%{sc[1]:>7.2f}{sc[2]*100:>6.0f}%{sc[3]*100:>5.0f}%{sc[4]:>7}")
print(f"  {'ever-member (+94 removed/delisted)':32}{se[0]*100:>6.1f}%{se[1]:>7.2f}{se[2]*100:>6.0f}%{se[3]*100:>5.0f}%{se[4]:>7}")
print(f"  survivorship inflation on CAGR: {(sc[0]-se[0])*100:+.1f} pts")
# chart
import os,urllib.request,json
q=pd.Series({pd.Timestamp(x['date'][:10]):x['adjClose'] for x in json.loads(urllib.request.urlopen('https://api.tiingo.com/tiingo/daily/QQQ/prices?startDate=2001-01-01&token='+os.environ['TIINGO']+'&format=json',timeout=40).read())}).reindex(eqc.index).ffill()
qbh=100000*q/q.iloc[0]
fig,ax=plt.subplots(figsize=(11,5.5))
ax.plot(eqc.index,eqc,color="#b91c1c",lw=2.0,label=f"Current NDX-100 (survivorship-biased): CAGR {sc[0]*100:.1f}%, DD {sc[2]*100:.0f}%")
ax.plot(eqe.index,eqe,color="#111",lw=2.2,label=f"Ever-member (+delisted, honest-er): CAGR {se[0]*100:.1f}%, DD {se[2]*100:.0f}%")
ax.plot(qbh.index,qbh,color="#2563eb",lw=1.6,ls="--",label=f"QQQ buy&hold: CAGR {((qbh.iloc[-1]/qbh.iloc[0])**(365.25/(qbh.index[-1]-qbh.index[0]).days)-1)*100:.1f}%")
ax.set_yscale("log"); ax.grid(alpha=.3,which="both"); ax.legend(loc="upper left",fontsize=9)
ax.set_title("NDX-100 mean-reversion: survivorship-biased vs ever-member universe (2001-2026)")
fig.tight_layout(); fig.savefig("/home/user/ndx_surv.png",dpi=120)
print(f"saved chart  DONE t={time.time()-t0:.0f}s")
