"""Generate aegis_data.json for the AEGIS factsheet — the all-weather leveraged
risk-parity companion to VOLT. AEGIS = momentum-tilted risk parity across 3x NASDAQ
(TQQQ), 3x long Treasury (TMF, reconstructed+validated), and 2x gold (UGL), inverse-vol
weighted with a continuous momentum tilt (k=3), vol-targeted to 22%. It BEATS QQQ-DCA in
every crisis (dot-com/GFC/2022) with ~half the drawdown, and honestly TRAILS in tech bulls.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import riskparity as RP

ASSETS = ["TQQQ", "TMF_R", "UGL_R"]
NICE = {"TQQQ": "TQQQ", "TMF_R": "TMF", "UGL_R": "UGL"}
CFG = dict(target=0.22, mode="momtilt", fixed={"tilt_k": 3.0})
S, E = pd.Timestamp("2000-01-01"), pd.Timestamp("2026-07-01")
close = RP.close

def monthly_curve(nth=None):
    mg = RP.month_grid(nth); grid = mg[(mg >= S) & (mg <= E)]
    W = RP.rp_weights(mg, ASSETS, **CFG)
    px = close.reindex(mg, method="ffill")
    pos = {}; contributed = 0.0; rows = []
    Vq = 0.0
    qm = close.reindex(RP.month_grid(nth))["QQQ"].pct_change()
    for dt in grid:
        cash = 0.0
        for t in list(pos.keys()):
            if t == "_CASH_": cash += pos.pop(t)
            elif np.isfinite(px.loc[dt, t]): cash += pos.pop(t)*px.loc[dt, t]*(1-0.001)
            else: pos.pop(t)
        contributed += 1000.0; cash += 1000.0
        wrow = W.loc[dt] if dt in W.index else pd.Series(0.0, index=ASSETS)
        wts = {}
        spent = 0.0
        for t in ASSETS:
            wt = float(wrow.get(t, 0.0))
            if wt > 0 and dt >= RP.valid_start[t] and np.isfinite(px.loc[dt, t]):
                pos[t] = wt*cash*(1-0.001)/px.loc[dt, t]; spent += wt*cash; wts[t] = wt
        if cash-spent > 0: pos["_CASH_"] = cash-spent
        V = sum((sh if t == "_CASH_" else sh*px.loc[dt, t])
                for t, sh in pos.items() if t == "_CASH_" or np.isfinite(px.loc[dt, t]))
        Vq = Vq*(1+(qm.get(dt, 0.0) if np.isfinite(qm.get(dt, 0.0)) else 0.0))+1000.0
        rows.append(dict(date=str(dt.date()), wts=wts,
                         cash_wt=round(max(0.0, 1-spent/cash), 4),
                         value=round(float(V), 2), contributed=contributed,
                         qqq_value=round(float(Vq), 2)))
    df = pd.DataFrame(rows)
    # drawdown of lump-sum $1 strat return
    r = ((df["value"]-df["contributed"].diff().fillna(df["contributed"]))/df["value"].shift(1)-1).fillna(0)
    cum = (1+r).cumprod(); df["drawdown"] = (cum/cum.cummax()-1).round(4)
    return df

df = monthly_curve()
monthly = []
for _, row in df.iterrows():
    monthly.append(dict(date=row["date"],
        w={NICE[k]: round(v, 3) for k, v in row["wts"].items()},
        cash=row["cash_wt"], value=row["value"], contributed=row["contributed"],
        qqq_value=row["qqq_value"], drawdown=float(row["drawdown"])))

# eras + crises
def era_ratio(a, b, nth=None):
    s, e = pd.Timestamp(a), pd.Timestamp(b)
    W = RP.rp_weights(RP.month_grid(nth), ASSETS, **CFG)
    eq = RP.run(s, e, W, nth=nth); bq = RP.qqq_dca(s, e, nth=nth)
    return float(eq["V"].iloc[-1]/bq["V"].iloc[-1]), float(RP.maxdd(eq)), float(RP.maxdd(bq))

CRISES = [("Dot-com 2000–03", "2000-01-01", "2003-12-01"),
          ("GFC 2007–09", "2007-01-01", "2009-12-01"),
          ("2018 Q4", "2018-01-01", "2018-12-31"),
          ("COVID 2020", "2020-01-01", "2020-12-31"),
          ("2022 bear", "2022-01-01", "2022-12-31")]
crises = []
for lab, a, b in CRISES:
    r, dds, ddq = era_ratio(a, b)
    crises.append(dict(label=lab, ratio=round(r, 2), aegis_dd=round(dds, 3), qqq_dd=round(ddq, 3)))

ERAS = [("2000–09", "2000-01-01", "2009-12-01"), ("2010–19", "2010-01-01", "2019-12-01"),
        ("2020–26", "2020-01-01", "2026-06-01"), ("Full 2000–26", "2000-01-01", "2026-06-01")]
eras = [dict(era=lab, ratio=round(era_ratio(a, b)[0], 2)) for lab, a, b in ERAS]

# phase-robustness range (full 2000-26)
phase = {}
for nth in [None, 4, 9, 14]:
    r, dd, _ = era_ratio("2000-01-01", "2026-06-01", nth=nth)
    phase[("ME" if nth is None else f"d{nth}")] = dict(ratio=round(r, 2), maxdd=round(dd, 3))
pv = [v["ratio"] for v in phase.values()]

def lump(nth=None):
    W = RP.rp_weights(RP.month_grid(nth), ASSETS, **CFG)
    eq = RP.run(S, E, W)
    r = ((eq["V"]-eq["contributed"].diff().fillna(eq["contributed"]))/eq["V"].shift(1)-1).dropna()
    cum = (1+r).cumprod(); cagr = cum.iloc[-1]**(12/len(r))-1
    return dict(cagr=round(float(cagr), 4), sharpe=round(float(r.mean()/r.std()*np.sqrt(12)), 2),
                maxdd=round(float((cum/cum.cummax()-1).min()), 4))
lm = lump()

last = df.iloc[-1]
summary = dict(
    start=monthly[0]["date"], end=monthly[-1]["date"], months=len(monthly),
    contributed=float(df["contributed"].iloc[-1]),
    aegis_value=float(df["value"].iloc[-1]), qqq_value=float(df["qqq_value"].iloc[-1]),
    wealth_ratio=round(float(df["value"].iloc[-1]/df["qqq_value"].iloc[-1]), 2),
    aegis_moic=round(float(df["value"].iloc[-1]/df["contributed"].iloc[-1]), 2),
    qqq_moic=round(float(df["qqq_value"].iloc[-1]/df["contributed"].iloc[-1]), 2),
    cagr=lm["cagr"], sharpe=lm["sharpe"], maxdd=lm["maxdd"],
    crises=crises, eras=eras, phase=phase,
    ratio_low=min(pv), ratio_high=max(pv),
    current=dict({NICE[k]: round(v, 3) for k, v in last["wts"].items()}, CASH=round(last["cash_wt"], 3)))

out = dict(summary=summary, monthly=monthly)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aegis_data.json")
json.dump(out, open(path, "w"), separators=(",", ":"))
print("wrote aegis_data.json:", len(monthly), "months")
print("full wealth ratio:", summary["wealth_ratio"], "CAGR", lm["cagr"], "Sharpe", lm["sharpe"], "maxDD", lm["maxdd"])
print("crises:", [(c["label"], c["ratio"], c["aegis_dd"], c["qqq_dd"]) for c in crises])
print("eras:", eras, "phase range:", summary["ratio_low"], "-", summary["ratio_high"])
print("current alloc:", summary["current"])
