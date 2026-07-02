import pandas as pd, numpy as np, warnings, time; warnings.filterwarnings("ignore")
t0=time.time()
data=pd.read_pickle("/tmp/ndx_daily.pkl")
tk=[t for t,df in data.items() if len(df)>250]
dates=sorted(set().union(*[set(data[t].index) for t in tk]))
dates=[d for d in dates if d>=pd.Timestamp("2001-01-01")]
def panel(f): return pd.DataFrame({t:data[t][f] for t in tk}).reindex(dates)
O,H,L,C,V=[panel(x) for x in ["o","h","l","c","v"]]
prevC=C.shift(1); prevH=H.shift(1)
tr=pd.concat([H-L,(H-prevC).abs(),(L-prevC).abs()]).groupby(level=0).max()
atr=tr.rolling(5,min_periods=5).mean(); atrp=atr/C
ema200=C.ewm(span=200,min_periods=200).mean(); vol20=V.rolling(20,min_periods=20).mean()
# universe filter (known at signal close t): C>5, vol20>100k, C>ema200, atr%>3%
uni=(C>5)&(vol20>100000)&(C>ema200)&(atrp>0.03)
trig=(C<prevC*0.97)&uni                       # closed >3% below prior close
Ci={d:i for i,d in enumerate(dates)}
def run(pos_frac=0.20, maxpos=10, leverage=True, comm=0.005, E0=100000.0):
    cash=E0; pos={}; eq=[]; trades=[]; pend=[]
    Cv,Ov,Hv,Lv,PHv=C.values,O.values,H.values,L.values,prevH.values
    atrv=atr.values; trigv=trig.values; colidx={t:j for j,t in enumerate(tk)}
    for i,d in enumerate(dates):
        # 1) manage existing positions on today's bar (positions opened before today)
        for t in list(pos.keys()):
            p=pos[t]; j=colidx[t]
            hi,lo,cl,ph=Hv[i,j],Lv[i,j],Cv[i,j],PHv[i,j]
            if not np.isfinite(cl): continue
            ex=None; px=None
            if np.isfinite(hi) and hi>=p["tgt"]: ex="target"; px=p["tgt"]           # profit-target limit
            elif np.isfinite(ph) and cl>ph: ex="close>prevH"; px=cl
            elif p["bars"]>=9: ex="9bars"; px=cl
            if ex:
                cash+=p["sh"]*px - p["sh"]*comm; trades.append((t,p["ep"],px,ex,p["bars"],(px/p["ep"]-1)))
                pos.pop(t); continue
            p["bars"]+=1
        # 2) fill pending limit orders from yesterday using today's O/L
        for t,lim,tgt in pend:
            if t in pos or len(pos)>=maxpos: continue
            j=colidx[t]; op,lo=Ov[i,j],Lv[i,j]
            if not np.isfinite(lo) or lo>lim: continue                              # limit not reached
            fill=min(op,lim) if np.isfinite(op) else lim
            equity=cash+sum(pp["sh"]*Cv[i,colidx[tt]] for tt,pp in pos.items() if np.isfinite(Cv[i,colidx[tt]]))
            sh=int((pos_frac*equity)//fill)
            if sh<=0: continue
            cost=sh*fill+sh*comm
            if not leverage and cost>cash: 
                sh=int((cash-1)//(fill+comm)); cost=sh*fill+sh*comm
                if sh<=0: continue
            cash-=cost; pos[t]={"sh":sh,"ep":fill,"tgt":tgt,"bars":0}
        # 3) mark-to-market equity
        mv=sum(pp["sh"]*Cv[i,colidx[tt]] for tt,pp in pos.items() if np.isfinite(Cv[i,colidx[tt]]))
        eq.append(cash+mv)
        # 4) generate today's signals -> limit orders for tomorrow, ranked by atr% desc
        cand=[(atrp.values[i,colidx[t]],t) for t in tk if trigv[i,colidx[t]]]
        cand=[c for c in cand if np.isfinite(c[0])]; cand.sort(reverse=True)
        pend=[(t, Cv[i,colidx[t]]-0.9*atrv[i,colidx[t]], Cv[i,colidx[t]]+0.5*atrv[i,colidx[t]]) for _,t in cand[:20]]
    return pd.Series(eq,index=dates),pd.DataFrame(trades,columns=["tk","entry","exit","reason","bars","ret"])
def stats(eq,tr):
    r=eq.pct_change().dropna(); n=len(r); yrs=(eq.index[-1]-eq.index[0]).days/365.25
    cagr=(eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1; sh=r.mean()/r.std()*np.sqrt(252) if r.std()>0 else np.nan
    dd=(eq/eq.cummax()-1).min(); expo=(tr.shape[0]) 
    wr=(tr["ret"]>0).mean() if len(tr) else np.nan; avg=tr["ret"].mean() if len(tr) else np.nan
    return cagr,sh,dd,wr,avg,len(tr),tr["bars"].mean() if len(tr) else np.nan
print(f"data ready {len(tk)} tickers, {dates[0].date()}..{dates[-1].date()} t={time.time()-t0:.0f}s")
print(f"{'config':30}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'winR':>6}{'avgTrd':>7}{'nTrd':>7}{'avgBars':>8}")
for nm,lev in [("literal 20%x10 (implied lev)",True),("no-leverage (cash-capped)",False)]:
    eq,tr=run(leverage=lev); c,s,dd,wr,avg,ntr,ab=stats(eq,tr)
    print(f"  {nm:28}{c*100:>6.1f}%{s:>7.2f}{dd*100:>6.0f}%{wr*100:>5.0f}%{avg*100:>6.2f}%{ntr:>7}{ab:>8.1f}")
    if lev: pd.to_pickle({"eq":eq,"tr":tr},"/tmp/ndx_mr_result.pkl")
print(f"DONE t={time.time()-t0:.0f}s")
