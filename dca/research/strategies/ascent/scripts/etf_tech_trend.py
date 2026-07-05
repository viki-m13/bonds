"""Leveraged-tech trend core: DCA into leveraged tech (TQQQ/TECL/SOXL) while tech
is in a confirmed uptrend, rotate to defense (GLD/TLT/BIL) when it breaks down.
The honest 'risk dial with a trend stop' — more of QQQ's winning beta, out before crashes.
Tested vs QQQ-DCA, era-sliced + strict OOS + honest drawdowns.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
from etf_tactical import mclose, close, above_ma, mom3, mom6, mom12, kind, dca_bench, stats, ma200

def run_core(start, end, risk_asset="TQQQ", defense=("GLD","TLT","BIL"),
             gate_ma=200, contrib=1000.0, cost=0.0010, mom_gate=0.0):
    dates = mclose.index[(mclose.index>=start)&(mclose.index<=end)]
    pos={}; contributed=0.0; rows=[]; hl=[]
    for dt in dates:
        px=mclose.loc[dt]
        # regime: is QQQ (the tech proxy) in a confirmed uptrend?
        q_up = bool(above_ma.loc[dt,"QQQ"]) and (mom3.loc[dt,"QQQ"]>mom_gate) and (mom6.loc[dt,"QQQ"]>mom_gate)
        # sell everything not matching current regime target
        target = risk_asset if q_up else None
        if not q_up:
            # pick best-trending defense above its own 200d MA, else BIL
            dcand=[d for d in defense if d in above_ma.columns and bool(above_ma.loc[dt,d]) and np.isfinite(px.get(d,np.nan))]
            dcand=sorted(dcand, key=lambda d:-(mom6.loc[dt,d] if np.isfinite(mom6.loc[dt,d]) else -1))
            target = dcand[0] if dcand else "BIL"
        freed=0.0
        for tk in list(pos.keys()):
            if tk!=target:
                freed+=pos[tk]*px.get(tk,0)*(1-cost); pos.pop(tk)
        cashamt=contrib+freed; contributed+=contrib
        if target in px and np.isfinite(px[target]):
            pos[target]=pos.get(target,0)+cashamt/px[target]; cashamt=0.0
        hl.append((dt,target))
        V=sum(sh*px.get(t,0) for t,sh in pos.items())+cashamt
        rows.append((dt,V,contributed))
    eq=pd.DataFrame(rows,columns=["date","V","contributed"]).set_index("date")
    return eq, pd.DataFrame(hl,columns=["date","held"]).set_index("date")

ERAS=[("2006-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),
      ("2020-01","2026-06"),("2010-01","2026-06"),("2006-01","2026-06")]
print(f"{'config':38} "+" ".join(f"{a[:7]:>7}" for a,b in ERAS)+"   fullMOIC  fullDD")
def line(nm, **kw):
    out=[];
    for st,en in ERAS:
        s=pd.Timestamp(st+"-01");e=pd.Timestamp(en+"-01")
        eq,_=run_core(s,e,**kw); b=dca_bench("QQQ",s,e)
        out.append(eq["V"].iloc[-1]/b["V"].iloc[-1])
    s=pd.Timestamp("2010-01-01");e=pd.Timestamp("2026-06-01")
    eq,_=run_core(s,e,**kw); S=stats(eq)
    print(f"{nm:38} "+" ".join(f"{v:>7.2f}" for v in out)+f"   {S['moic']:6.1f}x {S['maxdd']:6.1%}")

line("TQQQ|GLD,TLT,BIL (200MA gate)", risk_asset="TQQQ")
line("TQQQ|BIL only", risk_asset="TQQQ", defense=("BIL",))
line("TQQQ|TLT,GLD,BIL mom-gate5%", risk_asset="TQQQ", mom_gate=0.0)
line("QLD(2x)|GLD,TLT,BIL", risk_asset="QLD")
line("SOXL(3x semis)|GLD,TLT,BIL", risk_asset="SOXL")
line("TECL(3x tech)|GLD,TLT,BIL", risk_asset="TECL")
line("QQQ(1x)|GLD,TLT,BIL (no leverage)", risk_asset="QQQ")

# headline detail + strict OOS
print("\n--- TQQQ|GLD,TLT,BIL detail ---")
for tag,(st,en) in [("DEV 2010-2017",("2010-01","2017-12")),("OOS 2018-2026",("2018-01","2026-06"))]:
    s=pd.Timestamp(st+"-01");e=pd.Timestamp(en+"-01")
    eq,hl=run_core(s,e,risk_asset="TQQQ"); b=dca_bench("QQQ",s,e)
    S=stats(eq);Sb=stats(b)
    print(f"{tag}: ratio {S['final']/b['V'].iloc[-1]:.2f}x  CAGR {S['cagr']:.1%} Sh {S['sharpe']:.2f} DD {S['maxdd']:.1%}"
          f" | QQQ CAGR {Sb['cagr']:.1%} DD {Sb['maxdd']:.1%}")
print("recent:", hl.tail(8).held.tolist())
