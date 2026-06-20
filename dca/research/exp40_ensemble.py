"""Exp 40 — extend the ensemble with 2 new low-corr sleeves and re-measure.
NEW: (4) multi-asset TREND-following / managed-futures (long-or-cash, vol-target
~10%, the canonical equity diversifier; Hurst-Ooi-Pedersen). (5) cross-sectional
SHORT-TERM REVERSAL (long prior-1m losers, monthly). Combine with QQQ + IBS
mean-reversion + insider. Report correlations + ensemble Sharpe (inverse-vol),
OOS sub-periods. Keep only sleeves that genuinely diversify."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
def to_m(s):
    g = (1 + s).groupby(s.index.to_period("M")).prod() - 1; g.index = g.index.to_timestamp(); return g

# sleeve 1+2: QQQ + IBS mean-reversion
raw = yf.download(["QQQ"], start="1999-03-01", auto_adjust=True, progress=False)
c, h, l = raw["Close"]["QQQ"].dropna(), raw["High"]["QQQ"], raw["Low"]["QQQ"]
h, l = h.reindex(c.index), l.reindex(c.index); hl = h - l
entry = (c < h.rolling(10).max() - 2.5*hl.rolling(25).mean()) & ((c-l)/hl.replace(0,np.nan) < 0.3)
exi = c > h.shift(1); pos = np.zeros(len(c)); ip = False
for i in range(len(c)):
    ip = True if (not ip and entry.iloc[i]) else (False if ip and exi.iloc[i] else ip); pos[i] = ip
posl = pd.Series(pos, index=c.index).shift(1).fillna(0)
qqq_m = to_m(c.pct_change()); mr_m = to_m(posl*c.pct_change() - 0.0005*posl.diff().abs().fillna(0))

# sleeve 4: multi-asset trend-following (long-or-cash, vol-targeted ~10%)
U = ["SPY","QQQ","EFA","EEM","TLT","IEF","GLD","DBC","VNQ"]
tp = yf.download(U, start="2006-01-01", auto_adjust=True, progress=False)["Close"].dropna()
tm = tp.resample("ME").last(); tm.index = tm.index.to_period("M").to_timestamp()
tr = tm.pct_change(); above = tm > tm.rolling(10).mean()                 # 10-month trend
w = above.div(above.sum(axis=1).clip(lower=1), axis=0)                    # eq-wt assets in uptrend, rest cash
trend_raw = (w.shift(1) * tr).sum(axis=1)
rv = trend_raw.rolling(6).std()*np.sqrt(12); scale = (0.10/rv).clip(upper=1.0).shift(1).fillna(1.0)
trend_m = trend_raw*scale - 0.001*w.diff().abs().sum(axis=1)              # net cost

# sleeve 3 + 5: insider tilt + cross-sectional short-term reversal (cached universe)
me = pd.read_pickle("/tmp/wave/_ins_px.pkl"); names=[x for x in me.columns if x not in ("SPY","QQQ")]
mret = me.pct_change()
P = pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P=P[P.tk.isin(names)]
off = (P.pivot_table(index="ym",columns="tk",values="off_buy",aggfunc="sum").reindex(index=me.index,columns=names).fillna(0).rolling(3,min_periods=1).sum()>0)
ret1 = me[names].pct_change()
def sleeve(selfn, cost):
    prev=set(); rr=[]
    for i in range(3,len(me.index)-1):
        d=me.index[i]; nxt=me.index[i+1]; sel=selfn(d,nxt)
        if len(sel)<5: continue
        turn=1-len(prev&set(sel))/max(len(sel),1); rr.append((nxt, mret.loc[nxt,sel].mean()-turn*cost)); prev=set(sel)
    return pd.Series(dict(rr)).dropna()
ins_m = sleeve(lambda d,nxt:[t for t in names if off.loc[d,t] and np.isfinite(mret.loc[nxt,t])], 0.002)
def losers(d,nxt):
    r = ret1.loc[d].dropna(); r = r[[t for t in r.index if np.isfinite(mret.loc[nxt,t])]]
    return list(r[r<=r.quantile(0.2)].index)
xsr_m = sleeve(losers, 0.002)

D = pd.DataFrame({"QQQ":qqq_m,"MeanRev":mr_m,"Trend":trend_m,"Insider":ins_m,"XSReversal":xsr_m}).dropna()
print(f"ensemble panel {len(D)} months {D.index[0].date()}->{D.index[-1].date()}  t={time.time()-t0:.0f}s", flush=True)
print("\ncorrelations:\n", D.corr().round(2).to_string(), flush=True)
def st(s,lo=None,hi=None):
    if lo: s=s[(s.index>=lo)&(s.index<hi)]
    s=s.dropna(); eq=(1+s).cumprod(); yrs=len(s)/12
    return eq.iloc[-1]**(1/yrs)-1, s.mean()/(s.std()+1e-12)*np.sqrt(12), float((eq/eq.cummax()-1).min())
print("\nStandalone sleeves (CAGR/Sharpe/maxDD):", flush=True)
for col in D.columns:
    a=st(D[col]); print(f"  {col:12s} {a[0]*100:5.1f}% / {a[1]:.2f} / {a[2]*100:4.0f}%", flush=True)
iv = (1/D.std())/(1/D.std()).sum()
ports = {"100% QQQ":D.QQQ, "3-sleeve (QQQ+MR+Ins)":0.6*D.QQQ+0.2*D.MeanRev+0.2*D.Insider,
         "5-sleeve inverse-vol":(D*iv).sum(axis=1),
         "5-sleeve eq-weight":D.mean(axis=1)}
print("\nEnsembles:", flush=True)
for nm,s in ports.items():
    a=st(s); a1=st(s,"2010-01-01","2018-01-01"); a2=st(s,"2018-01-01","2026-12-31")
    print(f"  {nm:24s} CAGR {a[0]*100:5.1f}%  Sharpe {a[1]:.2f}  maxDD {a[2]*100:4.0f}%  [Sh 10-17 {a1[1]:.2f} | 18-25 {a2[1]:.2f}]", flush=True)
print(f"  inverse-vol weights: {dict((iv*100).round(0).astype(int))}", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
