"""Regenerate atlas_data.json for the VOLT factsheet from the CURRENT strategy.py
(now including the reversal dial). Also emits the honest trade-day range. Keeps the
exact JSON schema the factsheet (docs/volt.html) already consumes.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as ST
import phase as PH

close, mgrid, mret = ST.close, ST.mgrid, ST.mret
S, E = ST.S, ST.E
DEF = ("GLD", "TLT")

w = ST.tqqq_weight(0.30)                       # shipped weight (reversal on)
def av(win): return (ST.retd["TQQQ"].rolling(win, min_periods=int(win*0.7)).std()*np.sqrt(252)).reindex(mgrid, method="ffill")
volest = av(63)

g = mgrid[(mgrid >= S) & (mgrid <= E)]
rA = ST.strat_ret(S, E, target=0.30, defense=DEF)
eqA = ST.dca(rA)
qser = mret["QQQ"].reindex(g).fillna(0)
eqQ = ST.dca(qser)

# monthly rows
cum = (1+rA).cumprod(); dd = (cum/cum.cummax()-1)
monthly = []
for dt in g:
    wt = float(w.get(dt, 0.0)); wt = wt if np.isfinite(wt) else 0.0
    tr = float(mret.loc[dt, "TQQQ"]) if np.isfinite(mret.loc[dt, "TQQQ"]) else 0.0
    dr = float(np.nanmean([mret.loc[dt, d] for d in DEF]))
    monthly.append(dict(
        date=str(dt.date()), tqqq_wt=round(wt, 4), def_wt=round(1-wt, 4),
        tqqq_vol=round(float(volest.get(dt, np.nan)), 4) if np.isfinite(volest.get(dt, np.nan)) else None,
        mret=round(float(rA.get(dt, 0.0)), 4), value=round(float(eqA.loc[dt, "V"]), 2),
        contributed=float(eqA.loc[dt, "contributed"]),
        tqqq_ret=round(tr, 4), def_ret=round(dr, 4),
        qqq_value=round(float(eqQ.loc[dt, "V"]), 2), drawdown=round(float(dd.get(dt, 0.0)), 4)))

# trades = weight-change log (>2 percentage points)
trades = []; prevw = 0.0
for dt in g:
    wt = float(w.get(dt, 0.0)); wt = wt if np.isfinite(wt) else 0.0
    if abs(wt-prevw) > 0.02:
        trades.append(dict(date=str(dt.date()),
            action=("INCREASE TQQQ" if wt > prevw else "REDUCE TQQQ"),
            from_wt=round(prevw, 3), to_wt=round(wt, 3),
            tqqq_vol=round(float(volest.get(dt, np.nan)), 3) if np.isfinite(volest.get(dt, np.nan)) else None,
            portfolio=round(float(eqA.loc[dt, "V"]), 2)))
    prevw = wt

# calendar-year returns (strategy vs qqq), compounded within year
def yearly(r):
    return (1+r).groupby(r.index.year).prod()-1
yA, yQ = yearly(rA), yearly(qser)
years = [dict(year=int(y), atlas=round(float(yA[y]), 4), qqq=round(float(yQ.get(y, np.nan)), 4))
         for y in yA.index]

# lump stats
sa, sq = ST.lump_stats(rA), ST.lump_stats(qser)
st = ST.lump_stats(mret["TQQQ"].reindex(g))
ERAS = [("2006-01","2009-12","2006–2009"),("2010-01","2014-12","2010–2014"),
        ("2015-01","2019-12","2015–2019"),("2020-01","2026-06","2020–2026"),
        ("2010-01","2026-06","2010–2026"),("2006-01","2026-06","2006–2026")]
eras = []
for a, b, lab in ERAS:
    s, e = pd.Timestamp(a+"-01"), pd.Timestamp(b+"-01")
    r = ST.strat_ret(s, e, target=0.30, defense=DEF)
    rat = ST.dca(r)["V"].iloc[-1]/ST.dca(mret["QQQ"].reindex(mgrid[(mgrid>=s)&(mgrid<=e)]).fillna(0))["V"].iloc[-1]
    eras.append(dict(era=lab, ratio=round(float(rat), 2)))

# honest trade-day range (realistic cost, from phase.py)
rng = {("ME" if n is None else f"d{n}"): round(PH.ratio(S, E, n, 6.0), 2) for n in [None, 4, 9, 14]}
vals = list(rng.values())

summary = dict(
    start=str(g[0].date()), end=str(g[-1].date()), months=len(g),
    contributed=float(eqA["contributed"].iloc[-1]),
    atlas_value=round(float(eqA["V"].iloc[-1]), 2), qqq_value=round(float(eqQ["V"].iloc[-1]), 2),
    atlas_moic=round(float(eqA["V"].iloc[-1]/eqA["contributed"].iloc[-1]), 2),
    qqq_moic=round(float(eqQ["V"].iloc[-1]/eqQ["contributed"].iloc[-1]), 2),
    wealth_ratio=round(float(eqA["V"].iloc[-1]/eqQ["V"].iloc[-1]), 2),
    atlas={k: round(float(v), 4) for k, v in sa.items()},
    qqq={k: round(float(v), 4) for k, v in sq.items()},
    tqqq={k: round(float(v), 4) for k, v in st.items()},
    eras=eras,
    trade_day_range=rng, ratio_low=min(vals), ratio_high=max(vals),
    current_tqqq_wt=round(float(w.get(g[-1], 0.0)), 3),
    current_def_wt=round(1-float(w.get(g[-1], 0.0)), 3),
    current_vol=round(float(volest.get(g[-1], np.nan)), 3),
    n_trades=len(trades))

out = dict(summary=summary, monthly=monthly, trades=trades, years=years)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atlas_data.json")
json.dump(out, open(path, "w"), separators=(",", ":"))
print("wrote atlas_data.json")
print("wealth_ratio (ME):", summary["wealth_ratio"], " honest trade-day range:", rng)
print("CAGR", summary["atlas"]["cagr"], "Sharpe", summary["atlas"]["sharpe"],
      "maxDD", summary["atlas"]["maxdd"], "trades", len(trades), "years", len(years))
print("eras:", eras)
