"""Addendum evidence: QQQ vs SPY for the verdict page. Appends to /tmp/verdict_evidence.json.
- DCA final wealth, QQQ vs SPY, by era and full period (2000-2026, monthly, PIT panel)
- drawdown comparison (incl. dot-com: QQQ -81% vs SPY -55%)
- rolling 5y DCA win-rate QQQ vs SPY (is QQQ's edge just one era?)
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = f"{ROOT}/dca/research/strategies/ascent/scripts"
def _load(name):
    for p in (os.path.join(os.environ.get("ASCENT_WORK", "/tmp/ascent_work"), name), f"{A}/{name}"):
        if os.path.exists(p): return pd.read_pickle(p)
    raise FileNotFoundError(name)
ME = _load("_me_monthly.pkl")
q, s = ME["QQQ"].dropna(), ME["SPY"].dropna()
idx = q.index.intersection(s.index)
idx = idx[idx >= pd.Timestamp("1999-06-01")]
qr, sr = ME["QQQ"].reindex(idx).pct_change().fillna(0), ME["SPY"].reindex(idx).pct_change().fillna(0)

def dca(r):
    v = 0.0; out = []
    for x in r: v = (v+1000.0)*(1+x); out.append(v)
    return np.array(out)

OUT = json.load(open("/tmp/verdict_evidence.json")) if os.path.exists("/tmp/verdict_evidence.json") else {}
eras = [("2000-01","2004-12"),("2005-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),("2000-01","2026-06")]
rows = []
for a,b in eras:
    m = (idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b))
    dq, ds = dca(qr[m].values), dca(sr[m].values)
    rows.append({"era": f"{a[:4]}–{b[:4]}", "q": round(float(dq[-1])), "s": round(float(ds[-1])), "ratio": round(float(dq[-1]/ds[-1]),2)})
# full curves for chart
m = (idx>=pd.Timestamp("2000-01-01"))
dq, ds = dca(qr[m].values), dca(sr[m].values)
dts = [d.strftime("%Y-%m") for d in idx[m]]
# drawdowns of the DCA account value
def mdd(v):
    v=pd.Series(v); return float((v/v.cummax()-1).min())
# rolling 5y (60m) DCA start-anywhere win rate
wins=[]; L=60
rv_q, rv_s = qr[m].values, sr[m].values
for i0 in range(0, len(rv_q)-L):
    wq, ws = dca(rv_q[i0:i0+L])[-1], dca(rv_s[i0:i0+L])[-1]
    wins.append(wq>ws)
OUT["qqq_spy"] = {"eras": rows, "dates": dts,
    "q_curve": [round(float(x)) for x in dq], "s_curve": [round(float(x)) for x in ds],
    "q_mdd": round(mdd(dq)*100), "s_mdd": round(mdd(ds)*100),
    "roll5_winrate": round(float(np.mean(wins))*100)}
json.dump(OUT, open("/tmp/verdict_evidence.json","w"))
print("qqq_spy:", OUT["qqq_spy"]["eras"], "| 5y roll win", OUT["qqq_spy"]["roll5_winrate"], "% | mdd q/s", OUT["qqq_spy"]["q_mdd"], OUT["qqq_spy"]["s_mdd"])
