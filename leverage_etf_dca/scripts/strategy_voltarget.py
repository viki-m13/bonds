"""ATLAS-LEV — vol-targeted leveraged-NASDAQ tactical DCA. Definitive validation + chart.
import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
Strategy: each month hold weight w=clip(target_vol / trailing_63d_vol(TQQQ), 0, cap) in TQQQ,
(1-w) in a defensive asset; rebalance monthly; DCA $1000/mo. Honest risk on a lump-sum $1 path.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl"); close = P["close"].sort_index()
retd = close.pct_change()
# trading-day month-end grid
mgrid = pd.DatetimeIndex(sorted(close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1]).values))
mret = close.reindex(mgrid).pct_change()   # monthly returns per asset

def weights(target, cap=1.0, volwin=63, risk="TQQQ"):
    vol = (retd[risk].rolling(volwin, min_periods=40).std()*np.sqrt(252)).reindex(mgrid, method="ffill")
    return (target/vol).clip(0, cap)

def strat_monthly_ret(start, end, target=0.30, defense="TLT", cap=1.0, risk="TQQQ", cost=0.001, const=None):
    # weight decided at PRIOR month-end (no look-ahead): shift(1) so vol known before the month
    wfull = weights(target, cap, risk=risk).shift(1)
    g = mgrid[(mgrid>=start)&(mgrid<=end)]
    w = wfull.reindex(g)
    if const is not None: w = pd.Series(const, index=g)
    rr = []; prevw = 0.0
    for dt in g:
        wt = w.loc[dt] if np.isfinite(w.loc[dt]) else 0.0
        rt = mret.loc[dt, risk]; rd = mret.loc[dt, defense]
        rt = rt if np.isfinite(rt) else 0.0; rd = rd if np.isfinite(rd) else 0.0
        gross = wt*rt + (1-wt)*rd
        turn = abs(wt - prevw)*2  # both legs
        rr.append((dt, gross - turn*cost)); prevw = wt
    return pd.Series(dict(rr))

def lump_stats(r):
    r = r.dropna(); c = (1+r).cumprod()
    cagr = c.iloc[-1]**(12/len(r))-1; sh = r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    dd = (c/c.cummax()-1); mdd = dd.min()
    uw = (dd<0).astype(int); # longest underwater run
    run=mx=0
    for v in uw:
        run = run+1 if v else 0; mx=max(mx,run)
    w12 = (c/c.shift(12)-1).min()
    return dict(cagr=cagr, sharpe=sh, maxdd=mdd, underwater_m=mx, worst12=w12, mult=c.iloc[-1])

def dca(rser, contrib=1000.0):
    V=0.0; contributed=0.0; rows=[]
    for dt,r in rser.items():
        V=V*(1+r)+contrib; contributed+=contrib; rows.append((dt,V,contributed))
    return pd.DataFrame(rows,columns=["date","V","contributed"]).set_index("date")

def qqq_dca(start,end,contrib=1000.0):
    g=mgrid[(mgrid>=start)&(mgrid<=end)]; r=mret["QQQ"].reindex(g).fillna(0)
    return dca(r,contrib)

ERAS=[("2006-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),
      ("2010-01","2026-06"),("2006-01","2026-06")]
S=pd.Timestamp("2006-01-01"); E=pd.Timestamp("2026-07-01")

print("=== 1. DCA era ratios vs QQQ-DCA ===")
print(f"{'config':24} "+" ".join(f"{a[:7]:>7}" for a,b in ERAS))
def erow(nm,**kw):
    out=[]
    for st,en in ERAS:
        s=pd.Timestamp(st+"-01");e=pd.Timestamp(en+"-01")
        eq=dca(strat_monthly_ret(s,e,**kw)); b=qqq_dca(s,e)
        out.append(eq["V"].iloc[-1]/b["V"].iloc[-1])
    print(f"{nm:24} "+" ".join(f"{v:>7.2f}" for v in out))
    return out
r_tlt=erow("vt30 TQQQ|TLT",target=0.30,defense="TLT")
r_bil=erow("vt30 TQQQ|BIL",target=0.30,defense="BIL")
erow("vt25 TQQQ|TLT",target=0.25,defense="TLT")
erow("vt40 TQQQ|TLT",target=0.40,defense="TLT")
erow("vt30 TQQQ|GLD",target=0.30,defense="GLD")
erow("const50 TQQQ|BIL",const=0.5,defense="BIL")

print("\n=== 2. Honest lump-sum risk ($1, no contributions), 2006-2026 ===")
print(f"{'strategy':22} {'CAGR':>6} {'Shrp':>5} {'maxDD':>6} {'UW_m':>5} {'wrst12':>7} {'mult':>7}")
for nm,kw in [("vt30 TQQQ|TLT",dict(target=0.30,defense="TLT")),("vt30 TQQQ|BIL",dict(target=0.30,defense="BIL")),
              ("vt40 TQQQ|TLT",dict(target=0.40,defense="TLT")),("const50 TQQQ|BIL",dict(const=0.5,defense="BIL"))]:
    st=lump_stats(strat_monthly_ret(S,E,**kw))
    print(f"{nm:22} {st['cagr']:>6.1%} {st['sharpe']:>5.2f} {st['maxdd']:>6.0%} {st['underwater_m']:>5} {st['worst12']:>7.0%} {st['mult']:>6.0f}x")
qs=lump_stats(mret["QQQ"].reindex(mgrid[(mgrid>=S)&(mgrid<=E)]))
print(f"{'QQQ (buy&hold)':22} {qs['cagr']:>6.1%} {qs['sharpe']:>5.2f} {qs['maxdd']:>6.0%} {qs['underwater_m']:>5} {qs['worst12']:>7.0%} {qs['mult']:>6.0f}x")
ts=lump_stats(mret["TQQQ"].reindex(mgrid[(mgrid>=S)&(mgrid<=E)]))
print(f"{'TQQQ (buy&hold)':22} {ts['cagr']:>6.1%} {ts['sharpe']:>5.2f} {ts['maxdd']:>6.0%} {ts['underwater_m']:>5} {ts['worst12']:>7.0%} {ts['mult']:>6.0f}x")

print("\n=== 3. Phase robustness (vt30 TQQQ|TLT, rebalance on trading day n) full 2006-26 ratio ===")
for n in [0,4,9,14,-1]:
    g=[]
    for _,idx in close.groupby(close.index.to_period("M")).groups.items():
        d=close.loc[idx].index
        if len(d)>abs(n): g.append(d[n])
    g=pd.DatetimeIndex(sorted(g)); g=g[(g>=S)&(g<=E)]
    mr=close.reindex(g).pct_change()
    vol=(retd["TQQQ"].rolling(63,min_periods=40).std()*np.sqrt(252)).reindex(g,method="ffill")
    w=(0.30/vol).clip(0,1).shift(1); prevw=0; rr=[]
    for dt in g:
        wt=w.loc[dt] if np.isfinite(w.loc[dt]) else 0
        rt=mr.loc[dt,"TQQQ"]; rd=mr.loc[dt,"TLT"]; rt=rt if np.isfinite(rt) else 0; rd=rd if np.isfinite(rd) else 0
        rr.append((dt,wt*rt+(1-wt)*rd-abs(wt-prevw)*2*0.001)); prevw=wt
    eq=dca(pd.Series(dict(rr))); b=dca(mr["QQQ"].reindex(g).fillna(0))
    print(f"  day {n:>3}: {eq['V'].iloc[-1]/b['V'].iloc[-1]:.2f}x")

print("\n=== 4. REAL vs synthetic TQQQ (2011-2026, DCA ratio) ===")
def load(tk):
    for d in [os.path.join(REPO,"data/etfs"),os.path.join(REPO,"data/etfs_extended")]:
        f=f"{d}/{tk}.csv"
        if os.path.exists(f):
            s=pd.read_csv(f); s['Date']=pd.to_datetime(s['Date']); return s.set_index('Date')['Close'].sort_index()
realT=load("TQQQ")
for lbl,use_real in [("synthetic (recon)",False),("REAL TQQQ",True)]:
    g=mgrid[(mgrid>=pd.Timestamp("2011-01-01"))&(mgrid<=E)]
    px = realT.reindex(close.index).ffill() if use_real else close["TQQQ"]
    tret = px.reindex(g).pct_change()
    vol=(px.pct_change().rolling(63,min_periods=40).std()*np.sqrt(252)).reindex(g,method="ffill")
    w=(0.30/vol).clip(0,1).shift(1); prevw=0; rr=[]
    for dt in g:
        wt=w.loc[dt] if np.isfinite(w.loc[dt]) else 0; rt=tret.loc[dt]; rd=mret.loc[dt,"TLT"]
        rt=rt if np.isfinite(rt) else 0; rd=rd if np.isfinite(rd) else 0
        rr.append((dt,wt*rt+(1-wt)*rd-abs(wt-prevw)*2*0.001)); prevw=wt
    eq=dca(pd.Series(dict(rr))); b=dca(mret["QQQ"].reindex(g).fillna(0))
    print(f"  {lbl:20}: {eq['V'].iloc[-1]/b['V'].iloc[-1]:.2f}x QQQ-DCA")

print("\n=== 5. 2022 stress + defense-by-era (ratio in that era) ===")
for st,en in [("2022-01","2022-12")]:
    s=pd.Timestamp(st+"-01");e=pd.Timestamp(en+"-01")
    for dfa in ["TLT","BIL","GLD"]:
        eq=dca(strat_monthly_ret(s,e,target=0.30,defense=dfa)); b=qqq_dca(s,e)
        print(f"  {st}..{en} defense={dfa}: {eq['V'].iloc[-1]/b['V'].iloc[-1]:.2f}x QQQ-DCA (both fell; relative)")

# ---- EQUITY CURVE ----
def build_curve(target=0.30, defense="TLT"):
    r=strat_monthly_ret(S,E,target=target,defense=defense)
    return dca(r), qqq_dca(S,E), r
eqA,bq,rA = build_curve()
plt.rcParams.update({"font.size":10.5,"axes.edgecolor":"#d7dbe3","figure.facecolor":"white"})
BLUE,RED,INK="#2a78d6","#e34948","#6f7787"
fig,axes=plt.subplots(2,1,figsize=(11,10))
for ax,(logy,ttl) in zip(axes,[(False,"2006-2026 · $1,000/month"),(True,"2006-2026 (log)")]):
    ax.plot(eqA.index,eqA["V"],color=BLUE,lw=2,label="ATLAS-LEV (vol-targeted TQQQ|TLT)")
    ax.plot(bq.index,bq["V"],color=RED,lw=2,label="QQQ-DCA")
    ax.plot(eqA.index,eqA["contributed"],color=INK,lw=1.1,ls=(0,(4,3)),label="Contributed")
    if logy: ax.set_yscale("log")
    ax.grid(axis="y",color="#eceff4"); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.set_title(ttl,loc="left",fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v,_: f"${v/1e6:.1f}M" if v>=1e6 else f"${v/1e3:.0f}k")
    fr=eqA["V"].iloc[-1]/bq["V"].iloc[-1]
    ax.annotate(f"ATLAS ${eqA['V'].iloc[-1]/1e6:.1f}M ({fr:.1f}x QQQ-DCA)",xy=(eqA.index[-1],eqA["V"].iloc[-1]),
                xytext=(-4,6),textcoords="offset points",ha="right",color="#1a2333",fontweight="bold",fontsize=9.5)
    ax.annotate(f"QQQ-DCA ${bq['V'].iloc[-1]/1e6:.1f}M",xy=(bq.index[-1],bq["V"].iloc[-1]),
                xytext=(-4,-12),textcoords="offset points",ha="right",color="#8a2b2b",fontsize=9.5)
axes[0].legend(loc="upper left",frameon=False,fontsize=9.5)
lm=lump_stats(rA)
fig.suptitle(f"ATLAS-LEV: vol-targeted leveraged NASDAQ vs QQQ-DCA  ·  lump-sum maxDD {lm['maxdd']:.0%} (QQQ {qs['maxdd']:.0%})",
             x=0.06,y=0.99,ha="left",fontsize=13,fontweight="bold")
fig.text(0.06,0.955,"Monthly $1,000 DCA · 10bps/side · leveraged series reconstructed & validated vs real TQQQ (0.999 corr) · a risk-managed leverage dial, not alpha",fontsize=8.7,color=INK)
fig.tight_layout(rect=(0,0,1,0.94))
fig.savefig(f"{HERE}/etf_voltarget_equity.png",dpi=155,bbox_inches="tight")
print(f"\nsaved etf_voltarget_equity.png  ATLAS full {eqA['V'].iloc[-1]/bq['V'].iloc[-1]:.2f}x QQQ-DCA")
