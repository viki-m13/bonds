import pandas as pd, numpy as np, json, time, warnings; warnings.filterwarnings("ignore")
def p(*a): print(*a,flush=True)
t0=time.time()
D=pd.read_pickle("/tmp/wave/_featmat.pkl"); FEAT,fok,liq,me,cols=D["FEAT"],D["fok"],D["liq"],D["me"],D["cols"]
M=me.index; didx=list(M); fnames=list(FEAT.keys())
ret=(me/me.shift(1)-1).clip(-0.9,3.0); ma10=me.rolling(10,min_periods=10).mean(); mom3=me/me.shift(3)-1; mom6=FEAT["mom6"]
uni=pd.read_parquet("/home/user/bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet")
nm=uni.drop_duplicates("ticker").set_index("ticker")
def nameof(t):
    try: return str(nm.loc[t,"ticker"])
    except: return t
# ---------- A) HISTORICAL WAVE picks: ride-sim trade ledger from saved ML (2015-2025) ----------
PROBh=pd.read_pickle("/tmp/wave/_mlprob.pkl").reindex(M)
a=PROBh-PROBh.shift(2); elh=(liq&(me>=3.0)&(me>ma10)&(mom3>0)&(a>0)).fillna(False).astype(bool)
rankh=PROBh.where(elh).rank(axis=1,ascending=False)
pos={}; trades=[]
for k,dt in enumerate(didx):
    px=me.loc[dt]
    for tk in list(pos.keys()):
        e=pos[tk]; cpx=px.get(tk,np.nan)
        if not np.isfinite(cpx): trades.append((tk,e["dt"],k-e["i"],-0.6,False)); pos.pop(tk); continue
        e["peak"]=max(e["peak"],cpx); rs=cpx/e["px"]-1
        if cpx/e["peak"]-1<=-0.30 or cpx<ma10.loc[dt].get(tk,np.nan):
            trades.append((tk,e["dt"],k-e["i"],rs,False)); pos.pop(tk)
    if dt in PROBh.index:
        rk=rankh.loc[dt]; cands=[t for t in rk[rk<=48].sort_values().index if t not in pos and np.isfinite(px.get(t,np.nan))]
        for tk in cands:
            if len(pos)>=12: break
            pos[tk]={"i":k,"dt":dt.strftime("%Y-%m"),"px":px[tk],"peak":px[tk]}
    if k+1<len(didx):
        for tk in pos:
            r1=ret.iloc[k+1].get(tk,np.nan);
# open positions at end (last sim month)
endk=len(didx)-1
for tk,e in pos.items():
    cpx=me.iloc[endk].get(tk,np.nan); rs=(cpx/e["px"]-1) if np.isfinite(cpx) else np.nan
    trades.append((tk,e["dt"],endk-e["i"],rs,True))
tr=pd.DataFrame(trades,columns=["tk","entry","held","ret","open"])
top=tr.sort_values("ret",ascending=False).head(15)
hist=[{"t":nameof(r.tk),"entry":r.entry,"held":int(r.held),"ret":round(float(r.ret*100),0)} for r in top.itertuples()]
p(f"hist winners built ({len(tr)} trades) t={time.time()-t0:.0f}s")
# ---------- B) CURRENT picks: train GBT on all labeled data, predict latest months ----------
Z={k:FEAT[k].where(liq).rank(axis=1,pct=True) for k in fnames}
recs=[]
for dt in M[(M>=pd.Timestamp("2011-06-01"))&(M<=pd.Timestamp("2026-03-31"))]:
    fv=fok.loc[dt].dropna()
    if len(fv)<60: continue
    q1,q2=fv.quantile(1/3),fv.quantile(2/3); y=pd.Series(np.where(fv>=q2,1,np.where(fv<=q1,0,np.nan)),index=fv.index)
    X=np.column_stack([Z[k].loc[dt].reindex(fv.index).values for k in fnames])
    for i,tk in enumerate(fv.index): recs.append((dt,tk,*X[i],y.iloc[i]))
DF=pd.DataFrame.from_records(recs,columns=["date","tk"]+fnames+["y"]).dropna(subset=["y"])
from sklearn.ensemble import HistGradientBoostingClassifier
clf=HistGradientBoostingClassifier(max_iter=250,max_depth=4,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=200,random_state=0)
clf.fit(DF[fnames].values,DF["y"].astype(int).values)
# predict the most recent months for acceleration
predm=M[(M>=pd.Timestamp("2026-02-01"))&(M<=pd.Timestamp("2026-06-30"))]
PB=pd.DataFrame(index=predm,columns=cols,dtype=float)
for dt in predm:
    fl=liq.loc[dt]; tks=[c for c in cols if bool(fl.get(c,False))]
    if not tks: continue
    X=np.column_stack([Z[k].loc[dt].reindex(tks).values for k in fnames])
    PB.loc[dt,tks]=clf.predict_proba(X)[:,1]
last=predm[-1]; acc=PB.loc[last]-(PB.loc[predm[-3]] if len(predm)>=3 else PB.loc[predm[0]])
# real-common-stock filter: must have SEC fundamentals (ROA not NaN) AND no preferred/unit/warrant ticker
import re
realco=pd.Series({c:(("-" not in c) and (not re.search(r'(U|UN|WS|W|RT|R)$',c[3:] if len(c)>3 else "")) and bool(FEAT["roa"].loc[last].notna().get(c,False))) for c in cols})
elig=(liq.loc[last]&(me.loc[last]>=3.0)&(me.loc[last]>ma10.loc[last])&(mom3.loc[last]>0)&(acc>0)&realco).fillna(False)
sc=PB.loc[last].where(elig); waveN=sc.sort_values(ascending=False).head(12)
wave_now=[{"t":nameof(t),"score":int(round(PB.loc[last].rank(pct=True)[t]*100)),"mom6":round(float(mom6.loc[last].get(t,np.nan))*100,0),"px":round(float(me.loc[last].get(t,np.nan)),0)} for t in waveN.index]
# SUMMIT current longs/shorts (decile by score among liquid REAL companies)
sl=PB.loc[last].where(liq.loc[last]&realco); r=sl.rank(pct=True)
longs=sl[r>=0.9].sort_values(ascending=False).head(10); shorts=sl[r<=0.1].sort_values().head(10)
sum_long=[nameof(t) for t in longs.index]; sum_short=[nameof(t) for t in shorts.index]
p(f"current picks: {last.date()}  wave {len(wave_now)} | longs {len(sum_long)} shorts {len(sum_short)} t={time.time()-t0:.0f}s")
PICKS={"asof":last.strftime("%Y-%m"),"hist_asof":didx[endk].strftime("%Y-%m"),
       "wave_now":wave_now,"wave_hist":hist,"summit_long":sum_long,"summit_short":sum_short}
json.dump(PICKS,open("/home/user/_picks.json","w"))
p("WAVE NOW:",", ".join(x["t"] for x in wave_now))
p("SUMMIT LONG:",", ".join(sum_long))
p("SUMMIT SHORT:",", ".join(sum_short))
p("TOP HIST:",", ".join(f'{x["t"]}+{x["ret"]:.0f}%' for x in hist[:8]))
p(f"saved _picks.json t={time.time()-t0:.0f}s")
