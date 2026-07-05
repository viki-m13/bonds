"""macrogate.py — Can a MACRO/CREDIT regime signal time the leverage dial of a
DCA'd leveraged-NASDAQ strategy MORE ROBUSTLY than trailing vol alone?

Reuses phase.py's EXACT accounting (weight decided at prior grid point;
return = wt*rt + (1-wt)*defense - |dw|*2*cost) on nth-trading-day grids
(ME/4/9/14) so every result is reported phase-robust. Every macro signal is
reindexed to the grid with ffill and then shifted one grid point (prior
month-end / prior trading day) -> no look-ahead.

Macro inputs (FRED, all shifted to prior grid point):
  T10Y3M  10y-3m curve slope   (from 2000)  -- inversion leads recessions
  T10Y2Y  10y-2y curve slope   (from 1976)  -- deeper history
  BAMLH0A0HYM2  HY OAS credit spread (from 2000) -- covers dot-com AND GFC
  BAMLC0A0CM    IG OAS credit spread (from 2000)

VOLT baseline (the thing to beat) = vol-targeted TQQQ + reversal dial, copied
verbatim from phase.py.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
FRED = "/home/user/bonds/data/fred"
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl"); close = P["close"].sort_index()
retd = close.pct_change()

# ---- FRED, aligned to the daily close index (business-day ffill) ----
def fred(name):
    s = pd.read_csv(f"{FRED}/{name}.csv", parse_dates=["Date"]).set_index("Date")[name]
    return s.sort_index()
_F = {n: fred(n) for n in ["T10Y3M","T10Y2Y","BAMLH0A0HYM2","BAMLC0A0CM"]}
def daily(name):
    """FRED series forward-filled onto the trading-day close index."""
    return _F[name].reindex(close.index, method="ffill")

# ---- grids identical to phase.py ----
def grid(nth):
    g = []
    for _, idx in close.groupby(close.index.to_period("M")).groups.items():
        d = close.loc[idx].index
        if nth is None: g.append(d[-1])
        elif len(d) > nth: g.append(d[nth])
    return pd.DatetimeIndex(sorted(g))

# ---- VOLT baseline weight (verbatim from phase.py), returned UNSHIFTED ----
def vol_weight_raw(mg, target=0.30, cap=1.0, volwin=63, fastwin=20, accel_k=2.0,
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
    return w

# ================= MACRO FACTORS (unshifted, on grid mg) =================
# each returns a multiplier series in [floor,1] to apply to the VOLT weight.

def curve_pen(mg, series="T10Y3M", scale=1.0, persist_m=0):
    """Penalty in [0,1] that grows as the curve inverts (slope<0).
    persist_m>0: keep the max penalty over the trailing persist_m grid points
    (recessions typically arrive AFTER the curve re-steepens)."""
    s = daily(series).reindex(mg, method="ffill")
    pen = (-s/scale).clip(0.0, 1.0)          # 0 when slope>=0, ->1 as it inverts
    if persist_m: pen = pen.rolling(persist_m, min_periods=1).max()
    return pen.fillna(0.0)

def credit_pen(mg, series="BAMLH0A0HYM2", mode="mom", scale=1.5, lookwin=126):
    """Penalty in [0,1] from credit-spread stress.
    mode='mom'  : trailing change in OAS over lookwin trading days (rising=stress)
    mode='ma'   : OAS above its own trailing MA (excess / scale)
    mode='lvl'  : OAS level vs trailing median (relative richness)"""
    s = daily(series)
    if mode == "mom":
        raw = s - s.shift(lookwin)
    elif mode == "ma":
        raw = s - s.rolling(lookwin, min_periods=lookwin//2).mean()
    elif mode == "lvl":
        raw = s / s.rolling(lookwin, min_periods=lookwin//2).median() - 1.0
    raw = raw.reindex(mg, method="ffill")
    return (raw/scale).clip(0.0, 1.0).fillna(0.0)

def macro_factor(mg, kind, floor=0.0, **kw):
    if kind == "none":
        return pd.Series(1.0, index=mg)
    if kind == "curve":
        pen = curve_pen(mg, series=kw.get("cseries","T10Y3M"),
                        scale=kw.get("cscale",1.0), persist_m=kw.get("persist",0))
        return (1.0 - kw.get("a",1.0)*pen).clip(floor, 1.0)
    if kind == "curve_bin":
        s = daily(kw.get("cseries","T10Y3M")).reindex(mg, method="ffill")
        inv = (s < kw.get("thr",0.0))
        if kw.get("persist",0): inv = inv.rolling(kw["persist"], min_periods=1).max() > 0
        return pd.Series(np.where(inv.fillna(False), floor, 1.0), index=mg)
    if kind == "credit":
        pen = credit_pen(mg, series=kw.get("kseries","BAMLH0A0HYM2"),
                         mode=kw.get("mode","mom"), scale=kw.get("kscale",1.5),
                         lookwin=kw.get("lookwin",126))
        return (1.0 - kw.get("b",1.0)*pen).clip(floor, 1.0)
    if kind == "combo":   # continuous curve + credit tilt
        cp = curve_pen(mg, series=kw.get("cseries","T10Y3M"),
                       scale=kw.get("cscale",1.0), persist_m=kw.get("persist",0))
        kp = credit_pen(mg, series=kw.get("kseries","BAMLH0A0HYM2"),
                        mode=kw.get("mode","mom"), scale=kw.get("kscale",1.5),
                        lookwin=kw.get("lookwin",126))
        return (1.0 - kw.get("a",0.7)*cp - kw.get("b",0.7)*kp).clip(floor, 1.0)
    raise ValueError(kind)

# ---- combined weight = VOLT * macro, then shifted to prior grid point ----
def weight(mg, target=0.30, rev_k=6.0, macro="none", **mkw):
    w = vol_weight_raw(mg, target=target, rev_k=rev_k)
    f = macro_factor(mg, macro, **mkw)
    return (w*f).shift(1)

# ---- accounting identical to phase.py.ratio ----
def series_ret(start, end, nth, target=0.30, rev_k=6.0, macro="none",
               defense=("GLD","TLT"), cost=0.001, **mkw):
    mg = grid(nth); mret = close.reindex(mg).pct_change()
    g = mg[(mg >= start) & (mg <= end)]
    w = weight(mg, target=target, rev_k=rev_k, macro=macro, **mkw)
    out = {}; prevw = 0.0
    for dt in g:
        wt = w.get(dt, 0.0); wt = wt if np.isfinite(wt) else 0.0
        rt = mret.loc[dt, "TQQQ"]; rt = rt if np.isfinite(rt) else 0.0
        rd = np.nanmean([mret.loc[dt, d] for d in defense]); rd = rd if np.isfinite(rd) else 0.0
        out[dt] = wt*rt+(1-wt)*rd - abs(wt-prevw)*2*cost; prevw = wt
    return pd.Series(out)

def dca_final(r, contrib=1000.0):
    V = 0.0
    for x in r: V = V*(1+x)+contrib
    return V

def qqq_ratio(start, end, nth, **kw):
    r = series_ret(start, end, nth, **kw)
    mg = grid(nth); g = mg[(mg >= start) & (mg <= end)]
    qg = close.reindex(mg)["QQQ"].pct_change().reindex(g).fillna(0)
    return dca_final(r)/dca_final(qg)

def maxdd(r):
    """lump-sum max drawdown of the monthly return stream."""
    cum = (1+r.dropna()).cumprod()
    return (cum/cum.cummax()-1).min()

ERAS = [("dotcom 00-03",("2000-01","2003-12")), ("00-10",("2000-01","2010-12")),
        ("gfc 06-09",("2006-01","2009-12")), ("10-14",("2010-01","2014-12")),
        ("15-19",("2015-01","2019-12")), ("20-26",("2020-01","2026-06")),
        ("full 06-26",("2006-01","2026-06")), ("full 99-26",("1999-03","2026-06"))]

def ts(x): return pd.Timestamp(x+"-01")

def era_row(nm, **kw):
    out = []
    for _,(st,en) in ERAS:
        out.append(qqq_ratio(ts(st), ts(en), None, **kw))
    print(f"{nm:30} " + " ".join(f"{v:>6.2f}" for v in out))
    return out

def phase_row(nm, span, **kw):
    st,en = span; s,e = ts(st), ts(en)
    cells = {("ME" if n is None else f"d{n}"): qqq_ratio(s,e,n,**kw) for n in [None,4,9,14]}
    vals=list(cells.values())
    print(f"  {nm:30} " + "  ".join(f"{k}={v:.2f}" for k,v in cells.items())
          + f"   [range {min(vals):.2f}-{max(vals):.2f}]  minDDcut")
    return cells

# the single best macro config: HY OAS above its own 252d MA cuts the dial
WIN = dict(macro="credit", mode="ma", kseries="BAMLH0A0HYM2", lookwin=252, kscale=1.5, b=1.0, floor=0.35)
DEF = dict(macro="credit", mode="ma", kseries="BAMLH0A0HYM2", lookwin=252, kscale=1.0, b=1.0, floor=0.20)  # more defensive

def phase_maxdd(nm, span, **kw):
    st,en = span; s,e = ts(st), ts(en)
    cells = {("ME" if n is None else f"d{n}"): qqq_ratio(s,e,n,**kw) for n in [None,4,9,14]}
    v = list(cells.values()); dd = maxdd(series_ret(s,e,None,**kw))
    print(f"  {nm:28} " + "  ".join(f"{k}={x:.2f}" for k,x in cells.items())
          + f"  [range {min(v):.2f}-{max(v):.2f}]  maxDD={dd:.0%}")

if __name__ == "__main__":
    hdr = " ".join(f"{a[:6]:>6}" for a,_ in ERAS)
    print("=== DCA final-wealth ratio vs QQQ-DCA (month-end) ===")
    print(f"{'config':30} {hdr}")
    era_row("VOLT baseline (macro off) *", macro="none")
    print("-- curve gates (FAIL: chronic de-lever thru 2022-25 tech bull) --")
    era_row("curve3m cont a=.6",          macro="curve", a=0.6, cscale=1.0, floor=0.3)
    era_row("curve3m BIN persist12",      macro="curve_bin", thr=0.0, persist=12, floor=0.3)
    print("-- credit gates (HY OAS, from 2000; covers dot-com AND GFC) --")
    era_row("credit HY mom126",           macro="credit", mode="mom", kscale=1.5, b=1.0, floor=0.3)
    era_row("credit HY above-MA252 *WIN", **WIN)
    era_row("credit HY above-MA252 defv", **DEF)
    print("-- combo curve+credit --")
    era_row("combo 3m+HYma",              macro="combo", a=0.5, b=0.5, cscale=1.0, mode="ma", kscale=1.0, floor=0.25)

    print("\n=== PHASE-ROBUSTNESS + maxDD (rebalance days ME/4/9/14) ===")
    for slbl,span in [("dotcom 00-03",("2000-01","2003-12")),("gfc 06-09",("2006-01","2009-12")),
                      ("full 06-26",("2006-01","2026-06")),("full 99-26",("1999-03","2026-06"))]:
        print(f"[{slbl}]")
        phase_maxdd("VOLT baseline", span, macro="none")
        phase_maxdd("credit HY above-MA252 WIN", span, **WIN)
        phase_maxdd("credit HY above-MA252 defv", span, **DEF)
