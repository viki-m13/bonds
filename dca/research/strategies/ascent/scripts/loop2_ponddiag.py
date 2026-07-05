"""LOOP iter 2a — the feasibility diagnostic: for EACH pond, run its static
top-5 leaders basket (same mandate mechanics, QQQ parking when <k eligible)
through every era. Question: does ANY pond's concentrated basket beat QQQ-DCA
in every era? If none wins 2003-09, all-era outperformance via pond
concentration is infeasible and the search must change direction.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import dca_benchmark, stats

t0 = time.time()
def p(*a): print(*a, flush=True)
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
qqq = bench["QQQ"].reindex(M)
mom12 = me / me.shift(12) - 1
vol6 = FEAT["vol6"]
dvr = dv.rank(axis=1, ascending=False)
top1000 = dvr <= 1000

PONDS = {
    "MEGA100":  (dvr <= 100),
    "NEXT100":  (dvr > 100) & (dvr <= 300),
    "MID":      (dvr > 300) & (dvr <= 800),
    "SMALL":    (dvr > 800) & (dvr <= 2000),
    "LOWVOL":   top1000 & (vol6.where(top1000).rank(axis=1, pct=True) <= 0.2),
    "HIMOM":    top1000 & (mom12.where(top1000).rank(axis=1, pct=True) >= 0.9),
}

def run_static(pond_mask, pick, start, end, k=5, contrib=1000.0, cost=0.002, trail=-0.30):
    dates = M[(M >= start) & (M <= end)]
    elig_all = (liq & pond_mask & (me >= 3.0) & (me > ma10))
    pos = {}; qqq_units = 0.0; qqq_entry = None; cash = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        prow = me.loc[dt]; qp = qqq.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= 0.75; cash += e["val"] * (1 - cost); pos.pop(tk)
        srow = pick.loc[dt]; erow = elig_all.loc[dt]; marow = ma10.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < 30: continue
            cp = e["last_px"]
            if (cp / e["peak_px"] - 1) <= trail or (np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk]):
                cash += e["val"] * (1 - cost); pos.pop(tk)
        cash += contrib; contributed += contrib
        cand = srow[erow.reindex(srow.index).fillna(False).astype(bool)].dropna()
        cand = cand[~cand.index.isin(pos)].sort_values(ascending=False)
        need = k - len(pos)
        if need > 0 and cash > 1e-9 and len(cand):
            if qqq_units > 0 and qqq_entry is not None and (dt - qqq_entry).days >= 30:
                cash += qqq_units * qp * (1 - 0.0005); qqq_units = 0.0; qqq_entry = None
            picks = list(cand.index[:need]); amt = cash / len(picks)
            for tk in picks:
                pos[tk] = {"val": amt * (1 - cost), "last_px": prow[tk], "peak_px": prow[tk], "entry_date": dt}
            cash = 0.0
        elif cash > 1e-9 and len(pos):
            hs = {tk: srow.get(tk, np.nan) for tk in pos}
            tp = sorted(hs, key=lambda t: -(hs[t] if np.isfinite(hs[t]) else -1))[:3]
            for tk in tp: pos[tk]["val"] += (cash / len(tp)) * (1 - cost)
            cash = 0.0
        if cash > 1e-9 and np.isfinite(qp):
            qqq_units += cash * (1 - 0.0005) / qp
            if qqq_entry is None: qqq_entry = dt
            cash = 0.0
        V = cash + sum(e["val"] for e in pos.values()) + (qqq_units * qp if np.isfinite(qp) else 0)
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date"), dates

ERAS = [("2000-01", "2002-12"), ("2003-01", "2009-12"), ("2010-01", "2014-12"),
        ("2015-01", "2019-12"), ("2020-01", "2026-06")]
p(f"{'pond top-5 ($vol pick)':22} " + " ".join(f"{a[:7]:>9}" for a, b in ERAS))
DVP = dv.rank(axis=1, pct=True)
results = {}
for nm, mask in PONDS.items():
    vals = []
    for st, en in ERAS:
        eq, dts = run_static(mask, DVP.where(mask), pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01"))
        b = dca_benchmark(bench["QQQ"], dts)
        vals.append(stats(eq)["final"] / b["V"].iloc[-1])
    results[nm] = vals
    p(f"{nm:22} " + " ".join(f"{v:>9.2f}" for v in vals))
# also equal-weight whole-pond (k=25 to approximate)
p(f"\n{'pond top-25':22} " + " ".join(f"{a[:7]:>9}" for a, b in ERAS))
for nm, mask in PONDS.items():
    vals = []
    for st, en in ERAS:
        eq, dts = run_static(mask, DVP.where(mask), pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01"), k=25)
        b = dca_benchmark(bench["QQQ"], dts)
        vals.append(stats(eq)["final"] / b["V"].iloc[-1])
    p(f"{nm:22} " + " ".join(f"{v:>9.2f}" for v in vals))
p(f"\nDONE t={time.time()-t0:.0f}s")
