"""LOOP iter 4 — the required-skill curve. How much cross-sectional predictive
skill (rank-IC vs forward 12m returns) would a stock-picking method need for a
top-5 mandate basket to beat QQQ-DCA in EVERY era?
  - ORACLE: score = true forward-12m return (IC=1). The information ceiling.
  - Synthetic skill: score = rank(fwd12) blended with noise, calibrated to
    rank-IC in {0.05, 0.10, 0.20, 0.30, 0.50}; 5 seeds each; era ratios.
  - Reference: the best real signal found anywhere in this program has
    IC ~= 0.075-0.10 (the honest ML).
DISCLOSURE: the synthetic signals are look-ahead constructions used ONLY to map
the skill->outcome curve; they are not tradeable.
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
dvr = dv.rank(axis=1, ascending=False)
top1500 = dvr <= 1500
fwd12 = (me.shift(-12) / me - 1)
FR = fwd12.where(liq & top1500).rank(axis=1, pct=True)

def run_static(pick, start, end, k=5, contrib=1000.0, cost=0.002, trail=-0.30, trend_gate=True):
    dates = M[(M >= start) & (M <= end)]
    elig_all = (liq & top1500 & (me >= 3.0) & ((me > ma10) if trend_gate else True))
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
        ("2015-01", "2019-12"), ("2020-01", "2025-06")]   # 2025-06 end: fwd12 must exist

def era_row(pick, label):
    vals = []
    for st, en in ERAS:
        eq, dts = run_static(pick, pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01"))
        b = dca_benchmark(bench["QQQ"], dts)
        vals.append(stats(eq)["final"] / b["V"].iloc[-1])
    p(f"{label:26} " + " ".join(f"{v:>9.2f}" for v in vals))
    return vals

p(f"{'signal (IC)':26} " + " ".join(f"{a[:7]:>9}" for a, b in ERAS))
era_row(FR, "ORACLE (IC=1.0)")

# calibrate noise blends to target ICs
rng = np.random.default_rng(42)
def synth(ic_target, seed):
    rngs = np.random.default_rng(seed)
    noise = pd.DataFrame(rngs.random(me.shape), index=M, columns=cols).where(FR.notna())
    # blend weights: rank-correlation of w*FR + (1-w)*noise with FR ~ tuned empirically
    for w in np.linspace(0.01, 0.9, 60):
        S = w * FR + (1 - w) * noise
        # sample IC on 20 dates
        ics = []
        for dt in M[140:440:15]:
            a = S.loc[dt].dropna(); b = FR.loc[dt].reindex(a.index)
            d = pd.concat([a, b], axis=1).dropna()
            if len(d) > 200: ics.append(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
        if ics and np.nanmean(ics) >= ic_target:
            return S, w, np.nanmean(ics)
    return S, w, np.nanmean(ics)

for ic in [0.05, 0.10, 0.20, 0.30, 0.50]:
    allv = []
    for seed in range(3):
        S, w, got = synth(ic, 1000 + seed)
        vals = []
        for st, en in ERAS:
            eq, dts = run_static(S, pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01"))
            b = dca_benchmark(bench["QQQ"], dts)
            vals.append(stats(eq)["final"] / b["V"].iloc[-1])
        allv.append(vals)
    mean_v = np.mean(allv, axis=0)
    min_v = np.min(allv, axis=0)
    p(f"{'IC~%.2f (3 seeds, mean)' % ic:26} " + " ".join(f"{v:>9.2f}" for v in mean_v))
    p(f"{'          (worst seed)':26} " + " ".join(f"{v:>9.2f}" for v in min_v))
p(f"\nDONE t={time.time()-t0:.0f}s")
