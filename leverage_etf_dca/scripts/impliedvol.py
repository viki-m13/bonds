"""Implied-vol dial for VOLT — cross-pollination test from the SUMMIT research
(where inverse-implied-vol sizing beat realized-vol sizing for the market-neutral book).

Variants, all keeping the shipped reversal dial (rev6-MA) and identical harness:
  BASE      = shipped VOLT (realized 63d + 20d acceleration overlay)         [control]
  IV        = pure implied: vol* = 3 x (VIX/100) x trailing-252d QQQ/SPY vol ratio (PIT)
  MAXRV_IV  = max(realized-with-accel, implied)  -> de-lever on the worse signal
  TERM      = realized 63d, acceleration trigger replaced by VIX/VIX3M backwardation
              (vs *= (VIX/VIX3M)^2 when >1; 2009+ only, base accel before that)
No new free parameters beyond stated priors (3x, ratio, ^2 mirrors shipped accel_k=2).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl"); close = P["close"].sort_index()
retd = close.pct_change()
mgrid = pd.DatetimeIndex(sorted(close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1]).values))
mret = close.reindex(mgrid).pct_change()
VX = pd.read_pickle(f"{HERE}/_vix.pkl"); vix = VX["vix"]; v3 = VX["vix3m"]

def av(col, w):
    return (retd[col].rolling(w, min_periods=int(w*0.7)).std()*np.sqrt(252)).reindex(mgrid, method="ffill")

def rev_boost(rev_k=6.0, rev_cap=2.5, os_win=50, sec_win=200):
    q = close["QQQ"]; px = q.reindex(mgrid, method="ffill")
    sec = (q > q.rolling(sec_win, min_periods=120).mean()).reindex(mgrid, method="ffill")
    maN = q.rolling(os_win, min_periods=os_win//2).mean().reindex(mgrid, method="ffill")
    oversold = (maN/px - 1.0).clip(lower=0.0)
    return (1.0 + rev_k*oversold.where(sec.fillna(False), 0.0)).clip(1.0, rev_cap)

def w_variant(kind, target=0.30):
    vs_r = av("TQQQ", 63)
    accel = (av("TQQQ",20)/vs_r).clip(lower=1.0)**2.0
    vix_m = vix.reindex(mgrid, method="ffill")
    ratio = (retd["QQQ"].rolling(252,min_periods=150).std()/retd["SPY"].rolling(252,min_periods=150).std()).reindex(mgrid,method="ffill").fillna(1.2)
    vs_iv = 3.0*(vix_m/100.0)*ratio
    if kind=="BASE":      vs = vs_r*accel
    elif kind=="IV":      vs = vs_iv
    elif kind=="MAXRV_IV":vs = np.maximum(vs_r*accel, vs_iv)
    elif kind=="TERM":
        v3_m = v3.reindex(mgrid, method="ffill")
        bw = (vix_m/v3_m).clip(lower=1.0)**2.0
        vs = np.where(v3_m.notna(), vs_r*bw, vs_r*accel); vs = pd.Series(vs, index=mgrid)
    w = (target/vs).clip(0,1)
    return (w*rev_boost()).clip(0,1)

def strat_ret(w, start, end, defense=("GLD","TLT"), cost=0.001):
    rr=[]; prevw=0.0
    for dt in mgrid[(mgrid>=pd.Timestamp(start))&(mgrid<=pd.Timestamp(end))][1:]:
        i=mgrid.get_loc(dt); wt=w.iloc[i-1]
        if not np.isfinite(wt): continue
        rt=mret["TQQQ"].loc[dt]; rd=np.nanmean([mret[d].loc[dt] for d in defense])
        if not (np.isfinite(rt) and np.isfinite(rd)): continue
        rr.append((dt, wt*rt+(1-wt)*rd-abs(wt-prevw)*2*cost)); prevw=wt
    return pd.Series(dict(rr))

def dca(r, c=1000.0):
    v=0.0
    for x in r.values: v=(v+c)*(1+x)
    return v
def lump(r):
    yrs=len(r)/12; cagr=(1+r).prod()**(1/yrs)-1; sh=r.mean()/r.std()*np.sqrt(12)
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); return cagr,sh,dd

ERAS=[("2006-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-12"),("2010-01","2026-12"),("2006-01","2026-12")]
print(f"{'variant':12}"+ "".join(f"{a[:7]:>9}" for a,_ in ERAS) + f"{'CAGR':>7}{'Shrp':>6}{'maxDD':>7}")
base_ratios={}
for kind in ["BASE","IV","MAXRV_IV","TERM"]:
    w=w_variant(kind); cells=[]
    for s,e in ERAS:
        r=strat_ret(w,s,e); q=mret["QQQ"].loc[r.index]
        cells.append(dca(r)/dca(q))
    r=strat_ret(w,"2006-01","2026-12"); c_,s_,d_=lump(r)
    print(f"{kind:12}"+ "".join(f"{x:>9.2f}" for x in cells) + f"{c_*100:>6.1f}%{s_:>6.2f}{d_*100:>6.0f}%")
# phase-robustness quick check: shift rebalance to mid-month grid
mgrid2 = pd.DatetimeIndex(sorted(close.groupby(close.index.to_period('M')).apply(lambda x: x.index[min(9,len(x.index)-1)]).values))
print("\n(phase check d10, full 2006-26 ratio):")
mg_save=mgrid
for kind in ["BASE","IV","MAXRV_IV","TERM"]:
    globals()['mgrid']=mgrid2; globals()['mret']=close.reindex(mgrid2).pct_change()
    w=w_variant(kind); r=strat_ret(w,"2006-01","2026-12"); q=mret["QQQ"].loc[r.index]
    print(f"  {kind:10} {dca(r)/dca(q):.2f}")
    globals()['mgrid']=mg_save; globals()['mret']=close.reindex(mg_save).pct_change()
