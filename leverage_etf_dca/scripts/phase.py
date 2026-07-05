"""Phase-robustness + honest trade-day RANGE for the shipped VOLT, using strategy.py's
exact accounting (weight decided at prior grid point; return = wt*rt+(1-wt)*defense
- |dw|*2*cost) but on nth-trading-day-of-month grids instead of only month-end.
Reports the DCA ratio vs QQQ-DCA across rebalance days -> the honest range to publish."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl"); close = P["close"].sort_index()
retd = close.pct_change()

def grid(nth):
    g = []
    for _, idx in close.groupby(close.index.to_period("M")).groups.items():
        d = close.loc[idx].index
        if nth is None: g.append(d[-1])
        elif len(d) > nth: g.append(d[nth])
    return pd.DatetimeIndex(sorted(g))

def weight(mg, target=0.30, cap=1.0, volwin=63, fastwin=20, accel_k=2.0,
           rev_k=6.0, rev_cap=2.5, os_win=50, sec_win=200):
    def av(w): return (retd["TQQQ"].rolling(w, min_periods=int(w*0.7)).std()*np.sqrt(252)).reindex(mg, method="ffill")
    vs = av(volwin)
    if accel_k: vs = vs*(av(fastwin)/vs).clip(lower=1.0)**accel_k
    w = (target/vs).clip(0, cap)
    if rev_k:
        q = close["QQQ"]; px = q.reindex(mg, method="ffill")
        sec = (q > q.rolling(sec_win, min_periods=120).mean()).reindex(mg, method="ffill")
        maN = q.rolling(os_win, min_periods=os_win//2).mean().reindex(mg, method="ffill")
        osv = (maN/px-1.0).clip(lower=0.0)
        w = (w*(1.0+rev_k*osv.where(sec.fillna(False), 0.0)).clip(1.0, rev_cap)).clip(0, cap)
    return w.shift(1)

def ratio(start, end, nth, rev_k, defense=("GLD","TLT"), cost=0.001, contrib=1000.0):
    mg = grid(nth); mret = close.reindex(mg).pct_change()
    g = mg[(mg >= start) & (mg <= end)]
    w = weight(mg, rev_k=rev_k)
    V=c=0.0; prevw=0.0
    for dt in g:
        wt = w.get(dt, 0.0); wt = wt if np.isfinite(wt) else 0.0
        rt = mret.loc[dt, "TQQQ"]; rt = rt if np.isfinite(rt) else 0.0
        rd = np.nanmean([mret.loc[dt, d] for d in defense]); rd = rd if np.isfinite(rd) else 0.0
        r = wt*rt+(1-wt)*rd - abs(wt-prevw)*2*cost; prevw = wt
        V = V*(1+r)+contrib; c += contrib
    # QQQ-DCA on same grid
    qg = close.reindex(grid(nth))["QQQ"].pct_change().reindex(g).fillna(0)
    Vq=0.0
    for x in qg: Vq = Vq*(1+x)+contrib
    return V/Vq

if __name__ == "__main__":
    for span,(st,en) in [("2006-26",("2006-01","2026-07")),("1999-26",("1999-03","2026-07"))]:
        s,e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
        print(f"=== {span}  DCA ratio vs QQQ-DCA by rebalance day (realistic cost) ===")
        for lbl,rk in [("VOLT baseline (rev off)",0.0),("VOLT + reversal dial",6.0)]:
            cells = {("ME" if n is None else f"d{n}"): ratio(s,e,n,rk) for n in [None,4,9,14]}
            vals = list(cells.values())
            print(f"  {lbl:26} " + "  ".join(f"{k}={v:.2f}" for k,v in cells.items())
                  + f"   [range {min(vals):.2f}-{max(vals):.2f}]")
