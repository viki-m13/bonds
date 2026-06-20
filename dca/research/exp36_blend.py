"""Exp 36 — DEPLOYABLE: QQQ core + insider-officer-buy tilt sleeve vs plain QQQ.
Uses cached _ins_px.pkl + _insider_rich.pkl. Monthly rebal, net 20bps.
Result: 70/30 QQQ/insider -> Sharpe 1.17 vs QQQ 1.12, maxDD -28% vs -33%.
Sharpe/DD gain is diversification (robust); CAGR uplift partly survivorship-aided.
"""
import numpy as np, pandas as pd
me = pd.read_pickle("/tmp/wave/_ins_px.pkl"); names=[c for c in me.columns if c not in ("SPY","QQQ")]
P = pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P=P[P.tk.isin(names)]
mret = me.pct_change()
def pan(c): return P.pivot_table(index="ym",columns="tk",values=c,aggfunc="sum").reindex(index=me.index,columns=names).fillna(0)
offflag = (pan("off_buy").rolling(3,min_periods=1).sum()>0)
prev=set(); rets=[]
for i in range(3,len(me.index)-1):
    d=me.index[i]; nxt=me.index[i+1]
    sel=[t for t in names if offflag.loc[d,t] and np.isfinite(mret.loc[nxt,t])]
    if len(sel)<5: continue
    turn=1.0-len(prev&set(sel))/max(len(sel),1)
    rets.append((nxt, mret.loc[nxt,sel].mean()-turn*0.002)); prev=set(sel)
sl=pd.Series(dict(rets)).dropna(); q=mret["QQQ"].reindex(sl.index)
for w in (1.0,0.85,0.70,0.0):
    b=(w*q+(1-w)*sl).dropna(); eq=(1+b).cumprod()
    print(w, eq.iloc[-1]**(12/len(b))-1, b.mean()/b.std()*np.sqrt(12), (eq/eq.cummax()-1).min())
