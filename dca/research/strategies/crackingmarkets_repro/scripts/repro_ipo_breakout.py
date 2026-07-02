import pandas as pd, numpy as np, warnings, time, os, urllib.request, json; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t0=time.time(); B=pd.read_pickle("/tmp/ipo_daily.pkl"); data=B["data"]
entries={}; paths={}
for t,df in data.items():
    if "rc" not in df.columns or "rv" not in df.columns: continue
    df=df[~df.index.duplicated()].sort_index()
    if len(df)<5: continue
    if not (df["rc"].iloc[0]*df["rv"].iloc[0] > 10e6): continue
    c=df["c"].values; n=len(c); sig=None; run=c[0]
    for i in range(1,min(101,n)):
        if c[i]>run: sig=i; break
        run=max(run,c[i])
    if sig is None or sig+1>=n: continue
    ep=df["o"].iloc[sig+1]
    if not np.isfinite(ep) or ep<=0: continue
    entries.setdefault(df.index[sig+1],[]).append((t,ep))
    paths[t]=df.iloc[sig+1:][["o","h","l","c"]]
alldates=sorted(set().union(*[set(p.index) for p in paths.values()])); alldates=[d for d in alldates if d>=pd.Timestamp("2003-01-01")]
lut={t:{d:(r.o,r.h,r.l,r.c) for d,r in p.iterrows()} for t,p in paths.items()}
E0=100000.0; cash=E0; pos={}; eq=[]; trades=[]; expo=[]
for d in alldates:
    for t in list(pos.keys()):
        p=pos[t]; row=lut[t].get(d)
        if row is None:
            lastc=paths[t]["c"].iloc[-1]; px=lastc if np.isfinite(lastc) else p["ep"]*0.5
            cash+=p["sh"]*px; trades.append((px/p["ep"]-1,p["bars"],"delist")); pos.pop(t); continue
        o,h,l,c=row; stop=p["ep"]*0.80; tgt=p["ep"]*1.30; ex=px=None
        if np.isfinite(o) and o<=stop: ex,px="stop",o
        elif np.isfinite(o) and o>=tgt: ex,px="target",o
        elif np.isfinite(l) and l<=stop: ex,px="stop",stop
        elif np.isfinite(h) and h>=tgt: ex,px="target",tgt
        if ex: cash+=p["sh"]*px; trades.append((px/p["ep"]-1,p["bars"],ex)); pos.pop(t); continue
        p["bars"]+=1
    equity=cash+np.nansum([pp["sh"]*(lut[tt].get(d,(np.nan,np.nan,np.nan,pp["ep"]))[3]) for tt,pp in pos.items()]) if pos else cash
    for t,ep in entries.get(d,[]):
        if len(pos)>=5 or t in pos: continue
        sh=int((0.20*equity)//ep)
        if sh<=0 or sh*ep>cash: continue
        cash-=sh*ep; pos[t]={"sh":sh,"ep":ep,"bars":0}
    tot=cash+np.nansum([pp["sh"]*(lut[tt].get(d,(np.nan,np.nan,np.nan,pp["ep"]))[3]) for tt,pp in pos.items()])
    eq.append((d,tot)); expo.append(len(pos))
eq=pd.Series(dict(eq)); tr=pd.DataFrame(trades,columns=["ret","bars","reason"])
r=eq.pct_change().dropna(); yrs=(eq.index[-1]-eq.index[0]).days/365.25
cagr=(eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1; sh=r.mean()/r.std()*np.sqrt(252); dd=(eq/eq.cummax()-1).min()
print(f"IPO strategy FULL survivorship-clean universe ({len(paths)} IPOs with signals of {len(data)} fetched):")
print(f"  CAGR {cagr*100:.2f}%  Sharpe {sh:.2f}  maxDD {dd*100:.1f}%  | trades {len(tr)}  winR {(tr.ret>0).mean()*100:.0f}%  avgTrade {tr.ret.mean()*100:+.2f}%  avgHold {tr.bars.mean():.0f}d  avgExposure {np.mean(expo):.1f}/5")
print(f"  exits: {tr.reason.value_counts().to_dict()}")
print(f"  PUBLISHED (2003-2025): CAGR 20.53%  Sharpe 0.95  maxDD -31.57%  avgTrade +6.72%  avgHold 67d")
q=pd.Series({pd.Timestamp(x['date'][:10]):x['adjClose'] for x in json.loads(urllib.request.urlopen('https://api.tiingo.com/tiingo/daily/QQQ/prices?startDate=2003-01-01&token='+os.environ['TIINGO']+'&format=json',timeout=40).read())}).reindex(eq.index).ffill(); qbh=100000*q/q.iloc[0]
fig,ax=plt.subplots(figsize=(11,5.5))
ax.plot(eq.index,eq,color="#111",lw=2.2,label=f"IPO strategy (survivorship-clean, full universe): CAGR {cagr*100:.1f}%, DD {dd*100:.0f}%, Sharpe {sh:.2f}")
ax.plot(qbh.index,qbh,color="#2563eb",lw=1.7,ls="--",label=f"QQQ buy&hold: CAGR {((qbh.iloc[-1]/qbh.iloc[0])**(365.25/(qbh.index[-1]-qbh.index[0]).days)-1)*100:.1f}%")
ax.set_yscale("log"); ax.grid(alpha=.3,which="both"); ax.legend(loc="upper left",fontsize=9)
ax.set_title(f"IPO all-time-high breakout strategy reproduced — full survivorship-clean universe ({len(data)} IPOs, 2003-2026)")
fig.tight_layout(); fig.savefig("/home/user/ipo_full.png",dpi=120)
print(f"saved chart  DONE t={time.time()-t0:.0f}s")
