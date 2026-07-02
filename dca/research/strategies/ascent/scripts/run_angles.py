"""Remaining creative angles under the honest harness:
  A. Large-cap-constrained ML (ADV floors $20M/$50M) — fish in QQQ's pond.
  B. Explicit qualifier ensemble (exp62-67: rev-accel/hi-YoY & insider-cluster
     & uptrend), N=20, gentle exits, and a hold-through variant.
  C. Relative-target ML: y = terciles of (fwd12 - QQQ fwd12), embargo, all-names.
  D. fwd-6m target variant.
All monthly DCA, $1k, 20bps, min-30d hold, delist -25%.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)
from engine import dca_run, dca_benchmark, stats, twr

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, fok, liq, me, dv, bench, cols = D["FEAT"], D["fok"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M).reindex(columns=cols)
ma10 = me.rolling(10, min_periods=10).mean()
mom3 = me / me.shift(3) - 1
accel = PROB - PROB.shift(2)
qqq = bench["QQQ"]

START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-06-01")
dates = M[(M >= START) & (M <= END)]
bq = dca_benchmark(qqq, dates)

def show(nm, r):
    eq = r["equity"]; s = stats(eq)
    ratio = s["final"] / bq["V"].iloc[-1]
    rr = twr(eq); qr = twr(bq)
    subs = []
    for lo, hi in [("2015-01", "2021-12"), ("2022-01", "2026-06")]:
        a = rr[(rr.index >= lo) & (rr.index <= hi)]; b = qr[(qr.index >= lo) & (qr.index <= hi)]
        subs.append(f"{(1+a).prod()**(12/len(a))-1:+.1%}v{(1+b).prod()**(12/len(b))-1:+.1%}")
    p(f"{nm:52} IRR {s['irr']:6.1%} Sh {s['sharpe']:5.2f} DD {s['maxdd']:6.1%} vsQQQ {ratio:5.2f}x  [{subs[0]} | {subs[1]}]")

base = dict(dates=dates, N=12, trail=-0.30, ma=ma10, minhold_days=30,
            cost=0.0020, delist_ret=-0.25, cash_policy="add_top_held")
sq = stats(bq)
p(f"QQQ-DCA IRR {sq['irr']:.1%} Sh {sq['sharpe']:.2f} DD {sq['maxdd']:.1%}\n")

# ---- A. large-cap ADV floors ----
for advf in [2e7, 5e7]:
    EL = (liq & (me >= 3.0) & (dv >= advf) & (me > ma10) & (mom3 > 0))
    show(f"A: ML N12 ADV>=${advf/1e6:.0f}M", dca_run(me, PROB, EL, **base))
    show(f"A: ML N8  ADV>=${advf/1e6:.0f}M", dca_run(me, PROB, EL, **{**base, "N": 8}))

# ---- B. qualifier ensemble ----
rev_accel, rev_yoy, insn = FEAT["rev_accel"], FEAT["rev_yoy"], FEAT["ins_clustern"]
hi_yoy = rev_yoy.rank(axis=1, pct=True) >= 0.9
qual_mask = ((rev_accel > 0.5) | hi_yoy) & (insn >= 2) & (me > ma10)
QSCORE = FEAT["mom6"].where(qual_mask)
ELQ = (liq & (me >= 3.0) & (dv >= 2e6))
show("B: QUALIFIER N20 trail30+trend", dca_run(me, QSCORE, ELQ, **{**base, "N": 20}))
show("B: QUALIFIER N20 hold-through (trend exit only)", dca_run(me, QSCORE, ELQ, **{**base, "N": 20, "trail": -0.99}))
show("B: QUALIFIER N12 trail30+trend", dca_run(me, QSCORE, ELQ, **{**base, "N": 12}))

# ---- C/D. retrain with relative + fwd6 targets ----
fnames = list(FEAT.keys())
Z = {nm: FEAT[nm].where(liq).rank(axis=1, pct=True).astype(np.float32) for nm in fnames}
from sklearn.ensemble import HistGradientBoostingClassifier

def build_prob(fwd_target):
    recs = []
    for dt in M[(M >= pd.Timestamp("2011-06-01"))]:
        fv = fwd_target.loc[dt].dropna()
        if len(fv) < 60: continue
        q1, q2 = fv.quantile(1/3), fv.quantile(2/3)
        y = pd.Series(np.where(fv >= q2, 1, np.where(fv <= q1, 0, np.nan)), index=fv.index).dropna()
        X = np.column_stack([Z[nm].loc[dt].reindex(y.index).values for nm in fnames])
        for i, tk in enumerate(y.index):
            recs.append((dt, tk, *X[i], int(y.iloc[i])))
    DF = pd.DataFrame.from_records(recs, columns=["date", "tk"] + fnames + ["y"])
    pr = []
    for ytest in range(2015, 2027):
        tr = DF[DF.date < pd.Timestamp(f"{ytest-1}-01-01")]
        if len(tr) < 5000: continue
        clf = HistGradientBoostingClassifier(max_iter=250, max_depth=4, learning_rate=0.05,
                                             l2_regularization=1.0, min_samples_leaf=200)
        clf.fit(tr[fnames].values, tr["y"].values)
        te_m = M[(M >= pd.Timestamp(f"{ytest}-01-01")) & (M <= pd.Timestamp(f"{ytest}-12-31"))]
        for dt in te_m:
            ok = liq.loc[dt]; idxn = ok[ok].index
            X = np.column_stack([Z[nm].loc[dt].reindex(idxn).values for nm in fnames])
            good = ~np.isnan(X).all(axis=1)
            if good.sum() == 0: continue
            probs = clf.predict_proba(np.nan_to_num(X[good], nan=0.5))[:, 1]
            pr.append(pd.DataFrame({"date": dt, "tk": idxn[good], "p": probs}))
    return pd.concat(pr).pivot_table(index="date", columns="tk", values="p").reindex(M)

qm = qqq.reindex(M)
fwd12q = (qm.shift(-12) / qm - 1)
fwd12rel = ((me.shift(-12) / me - 1).sub(fwd12q, axis=0)).clip(-0.95, 5.0).where(liq)
PROB_REL = build_prob(fwd12rel)
p(f"\ntrained REL t={time.time()-t0:.0f}s")
pd.to_pickle(PROB_REL.astype(np.float32), f"{HERE}/_mlprob_rel.pkl")
fwd6 = ((me.shift(-6) / me - 1)).clip(-0.95, 5.0).where(liq)
PROB_F6 = build_prob(fwd6)
pd.to_pickle(PROB_F6.astype(np.float32), f"{HERE}/_mlprob_f6.pkl")
p(f"trained F6 t={time.time()-t0:.0f}s\n")

EL2 = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10) & (mom3 > 0))
EL50 = (liq & (me >= 3.0) & (dv >= 5e7) & (me > ma10) & (mom3 > 0))
for nm, PR in [("C: ML-REL", PROB_REL), ("D: ML-F6", PROB_F6)]:
    PR = PR.reindex(columns=cols)
    show(f"{nm} N12", dca_run(me, PR, EL2, **base))
    show(f"{nm} N12 ADV>=$50M", dca_run(me, PR, EL50, **base))
p(f"\nDONE t={time.time()-t0:.0f}s")
