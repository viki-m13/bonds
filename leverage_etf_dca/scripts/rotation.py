"""ALL-LEVERAGED-ETF monthly rotation — buy the trending leveraged ETFs each month,
sell when they break trend. Tests whether rotating across the full leveraged menu
beats (a) QQQ-DCA and (b) the single vol-targeted TQQQ sleeve.

Universe (16 leveraged, reconstructed from real underlyings, validated vs real TQQQ):
  tech:   TQQQ TECL SOXL QLD      broad: UPRO SSO TNA
  sector: FAS ERX DRN LABU        intl:  EDC YINN
  cmdty:  UGL UCO                 bonds: TMF
Rules: hold top-K leveraged ETFs by blended momentum that are ABOVE their 200d MA AND
whose UNDERLYING is in an uptrend; equal-weight; SELL on 200d-MA break; when <K qualify,
put the rest in a GLD-TLT defensive blend. Monthly DCA, 10bps/side, no look-ahead.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl"); close = P["close"].sort_index(); kind = P["kind"]; LEV = P["lev"]
LEVS = [t for t in close.columns if kind[t] == "lev"]
UNDER = {name: u for name, (u, L, e, i) in LEV.items()}

def month_grid(nth=None):
    """month-end grid (nth=None) or nth-trading-day-of-month grid."""
    g = []
    for _, idx in close.groupby(close.index.to_period("M")).groups.items():
        d = close.loc[idx].index
        if nth is None: g.append(d[-1])
        elif len(d) > nth: g.append(d[nth])
    return pd.DatetimeIndex(sorted(g))

def build_signals(grid):
    px = close.reindex(grid)
    ma200 = close.rolling(200, min_periods=150).mean().reindex(grid, method="ffill")
    above = (px > ma200)
    mom = (px/px.shift(3)-1).rank(axis=1,pct=True) + (px/px.shift(6)-1).rank(axis=1,pct=True) + (px/px.shift(12)-1).rank(axis=1,pct=True)
    return px, above, mom

_CACHE = {}
def _prep(nth):
    if nth not in _CACHE:
        mg = month_grid(nth)
        px, above, mom = build_signals(mg)
        dgrid = close.reindex(mg)
        vol = (close.pct_change().rolling(63,min_periods=40).std()*np.sqrt(252)).reindex(mg,method="ffill")
        _CACHE[nth] = (mg, px, above, mom, dgrid, vol)
    return _CACHE[nth]

def rotate(start, end, K=3, nth=None, contrib=1000.0, cost=0.001, defense=("GLD","TLT"),
           vol_target=None):
    mg, px, above, mom, dgrid, volp = _prep(nth)
    grid = mg[(mg>=start)&(mg<=end)]
    # underlying uptrend check
    und_ok = {name: (above[u] if u in above.columns else above[name]) for name,u in UNDER.items()}  # cached
    pos = {}; contributed=0.0; rows=[]; holdlog=[]
    prev_grid = list(month_grid(nth))
    for dt in grid:
        row_above = above.loc[dt]; row_mom = mom.loc[dt]
        # eligible leveraged ETFs: above own 200MA + underlying uptrend + positive momentum
        elig = []
        for t in LEVS:
            if not np.isfinite(px.loc[dt,t]): continue
            uok = bool(und_ok.get(t, row_above).loc[dt]) if t in UNDER else True
            if bool(row_above.get(t,False)) and uok and np.isfinite(row_mom.get(t,np.nan)):
                elig.append(t)
        elig = sorted(elig, key=lambda t: -row_mom[t])
        # SELL holdings that dropped below 200MA or underlying broke
        freed=0.0
        for t in list(pos.keys()):
            if t in defense: continue
            broke = (not bool(row_above.get(t,False))) or (t in UNDER and not bool(und_ok[t].loc[dt]))
            if broke or not np.isfinite(px.loc[dt,t]):
                freed += pos[t]*px.loc[dt,t]*(1-cost) if np.isfinite(px.loc[dt,t]) else pos[t]*0; pos.pop(t)
        # liquidate defense each month (redeploy)
        for t in list(pos.keys()):
            if t in defense:
                freed += pos[t]*dgrid.loc[dt,t]*(1-cost); pos.pop(t)
        cash = contrib + freed; contributed += contrib
        # target: keep held-still-eligible, fill to K with best new
        held = [t for t in pos if t in elig]
        target = held[:]
        for t in elig:
            if len(target)>=K: break
            if t not in target: target.append(t)
        target = target[:K]
        holdlog.append((dt, tuple(target) if target else ("DEF",)))
        # allocate: K slots. filled slots -> leveraged; empty slots -> defense blend
        nfill = len(target)
        w_risk = (nfill/K) if K>0 else 0
        if vol_target is not None and nfill>0:
            # scale total leveraged exposure by target/portfolio-vol proxy (avg vol of picks)
            vol = volp
            pv = np.nanmean([vol.loc[dt,t] for t in target])
            w_risk = min(w_risk, float(np.clip(vol_target/pv,0,1)))
        # buy
        if nfill>0 and w_risk>0:
            per = cash*w_risk/nfill
            for t in target:
                pos[t] = pos.get(t,0) + per/px.loc[dt,t]
        # defense for the rest
        drem = cash*(1-w_risk)
        if drem>1e-9:
            
            for t in defense:
                if np.isfinite(dgrid.loc[dt,t]): pos[t]=pos.get(t,0)+drem/len(defense)/dgrid.loc[dt,t]
        
        V = sum(sh*dgrid.loc[dt,t] for t,sh in pos.items() if np.isfinite(dgrid.loc[dt,t]))
        rows.append((dt,V,contributed))
    eq = pd.DataFrame(rows,columns=["date","V","contributed"]).set_index("date")
    return eq, pd.DataFrame(holdlog,columns=["date","held"]).set_index("date")

def qqq_dca(start,end,nth=None,contrib=1000.0):
    g=month_grid(nth); g=g[(g>=start)&(g<=end)]
    r=close.reindex(month_grid(nth))["QQQ"].pct_change().reindex(g).fillna(0)
    V=0;c=0;rows=[]
    for dt,x in r.items(): V=V*(1+x)+contrib; c+=contrib; rows.append((dt,V,c))
    return pd.DataFrame(rows,columns=["date","V","contributed"]).set_index("date")

def lump(eq):
    V=eq["V"];C=eq["contributed"].diff().fillna(eq["contributed"]);r=((V-C)/V.shift(1)-1).dropna()
    cum=(1+r).cumprod();dd=(cum/cum.cummax()-1).min()
    return dict(maxdd=dd)

if __name__=="__main__":
    ERAS=[("2006-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),
          ("2010-01","2026-06"),("2006-01","2026-06")]
    print(f"{'config':30} "+" ".join(f"{a[:7]:>7}" for a,b in ERAS))
    def row(nm,**kw):
        out=[]
        for st,en in ERAS:
            s,e=pd.Timestamp(st+"-01"),pd.Timestamp(en+"-01")
            eq,_=rotate(s,e,**kw);b=qqq_dca(s,e,nth=kw.get("nth"))
            out.append(eq["V"].iloc[-1]/b["V"].iloc[-1])
        print(f"{nm:30} "+" ".join(f"{v:>7.2f}" for v in out));return out
    row("all-lev rotate K1",K=1)
    row("all-lev rotate K3",K=3)
    row("all-lev rotate K5",K=5)
    row("all-lev rotate K3 vol-tgt30",K=3,vol_target=0.30)
    row("all-lev rotate K3 vol-tgt40",K=3,vol_target=0.40)
    print("\nPHASE robustness (all-lev K3 vol-tgt30, full 2006-26 ratio):")
    for nth in [None,4,9,14]:
        s,e=pd.Timestamp("2006-01-01"),pd.Timestamp("2026-07-01")
        eq,_=rotate(s,e,K=3,vol_target=0.30,nth=nth);b=qqq_dca(s,e,nth=nth)
        print(f"  {'ME' if nth is None else 'day'+str(nth):5}: {eq['V'].iloc[-1]/b['V'].iloc[-1]:.2f}x  maxDD {lump(eq)['maxdd']:.0%}")
    s,e=pd.Timestamp("2006-01-01"),pd.Timestamp("2026-07-01")
    eq,hl=rotate(s,e,K=3,vol_target=0.30)
    print("recent holdings:",hl.tail(6).held.tolist())
