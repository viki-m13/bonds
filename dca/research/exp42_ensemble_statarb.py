"""Exp 42 — does the market-neutral StatArb sleeve (0.13 corr, Sharpe ~0.34 @2bps)
lift the ensemble? Per exp40 the ceiling came from EQUITY-correlated sleeves; a
genuinely uncorrelated positive-Sharpe stream should help even at modest Sharpe.
Build StatArb (2bps) + QQQ + MeanRev + Insider monthly, combine, measure."""
import warnings, time, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
def to_m(s):
    g = (1+s).groupby(s.index.to_period("M")).prod()-1; g.index = g.index.to_timestamp(); return g
# StatArb (PCA residual reversal, 2bps/side)
names = sorted(set(open("/tmp/wave/sp500_universe.txt").read().split()))
raw = yf.download(names+["SPY","QQQ"], start="2009-01-01", auto_adjust=True, progress=False)["Close"]
names = [t for t in names if t in raw.columns and raw[t].notna().sum()>1500]
C = raw[names]; ret = C.pct_change(); wk = C.index[::5]; rows=[]; pL=pS=set(); COST=0.0002
for i in range(13,len(wk)-1):
    d=wk[i]; nxt=wk[i+1]; win=ret.loc[:d].iloc[-60:]
    valid=[t for t in names if win[t].notna().all() and np.isfinite(C.loc[nxt,t]) and np.isfinite(C.loc[d,t])]
    if len(valid)<50: continue
    R=win[valid].values; Rz=(R-R.mean(0))/(R.std(0)+1e-9)
    U,Sv,Vt=np.linalg.svd(Rz,full_matrices=False); F=U[:,:5]; resid=Rz-F@(F.T@Rz)
    z=pd.Series(resid[-5:].sum(0),index=valid); z=(z-z.mean())/(z.std()+1e-12)
    longs=list(z[z<=z.quantile(0.2)].index); shorts=list(z[z>=z.quantile(0.8)].index)
    fr=(C.loc[nxt]/C.loc[d]-1); turn=(1-len(pL&set(longs))/max(len(longs),1))+(1-len(pS&set(shorts))/max(len(shorts),1))
    rows.append((nxt,(fr[longs].mean()-fr[shorts].mean())-turn*COST)); pL,pS=set(longs),set(shorts)
sa=pd.Series(dict(rows)).dropna(); statarb_m=to_m(sa); statarb_m.to_pickle("/tmp/wave/_statarb_m.pkl")
print(f"StatArb built t={time.time()-t0:.0f}s", flush=True)
# QQQ + IBS mean-reversion
qr=yf.download(["QQQ"],start="1999-03-01",auto_adjust=True,progress=False)
c,h,l=qr["Close"]["QQQ"].dropna(),qr["High"]["QQQ"],qr["Low"]["QQQ"]; h,l=h.reindex(c.index),l.reindex(c.index); hl=h-l
entry=(c<h.rolling(10).max()-2.5*hl.rolling(25).mean())&((c-l)/hl.replace(0,np.nan)<0.3); exi=c>h.shift(1)
pos=np.zeros(len(c)); ip=False
for i in range(len(c)): ip=True if (not ip and entry.iloc[i]) else (False if ip and exi.iloc[i] else ip); pos[i]=ip
posl=pd.Series(pos,index=c.index).shift(1).fillna(0)
qqq_m=to_m(c.pct_change()); mr_m=to_m(posl*c.pct_change()-0.0005*posl.diff().abs().fillna(0))
# insider
me=pd.read_pickle("/tmp/wave/_ins_px.pkl"); inames=[x for x in me.columns if x not in ("SPY","QQQ")]; mret=me.pct_change()
P=pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P=P[P.tk.isin(inames)]
off=(P.pivot_table(index="ym",columns="tk",values="off_buy",aggfunc="sum").reindex(index=me.index,columns=inames).fillna(0).rolling(3,min_periods=1).sum()>0)
prev=set(); rr=[]
for i in range(3,len(me.index)-1):
    d=me.index[i]; nxt=me.index[i+1]; sel=[t for t in inames if off.loc[d,t] and np.isfinite(mret.loc[nxt,t])]
    if len(sel)<5: continue
    turn=1-len(prev&set(sel))/max(len(sel),1); rr.append((nxt,mret.loc[nxt,sel].mean()-turn*0.002)); prev=set(sel)
ins_m=pd.Series(dict(rr)).dropna()
D=pd.DataFrame({"QQQ":qqq_m,"MeanRev":mr_m,"Insider":ins_m,"StatArb":statarb_m}).dropna()
print(f"\npanel {len(D)} months\ncorrelations:\n{D.corr().round(2).to_string()}", flush=True)
def st(s,lo=None,hi=None):
    if lo: s=s[(s.index>=lo)&(s.index<hi)]
    s=s.dropna(); eq=(1+s).cumprod(); yrs=len(s)/12
    return eq.iloc[-1]**(1/yrs)-1, s.mean()/(s.std()+1e-12)*np.sqrt(12), float((eq/eq.cummax()-1).min())
iv=(1/D.std())/(1/D.std()).sum()
ports={"100% QQQ":D.QQQ,"3-sleeve (no StatArb)":0.6*D.QQQ+0.2*D.MeanRev+0.2*D.Insider,
       "4-sleeve inverse-vol":(D*iv).sum(axis=1),
       "50 QQQ/17 each":0.5*D.QQQ+(D.MeanRev+D.Insider+D.StatArb)/6}
print("\nEnsembles:", flush=True)
for nm,s in ports.items():
    a=st(s); a2=st(s,"2018-01-01","2026-12-31")
    print(f"  {nm:24s} CAGR {a[0]*100:5.1f}%  Sharpe {a[1]:.2f}  maxDD {a[2]*100:4.0f}%  [18-25 Sh {a2[1]:.2f}]", flush=True)
print(f"  inverse-vol weights: {dict((iv*100).round(0).astype(int))}", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
