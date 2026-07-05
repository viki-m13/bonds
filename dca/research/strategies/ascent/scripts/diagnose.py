"""Decompose the WAVE-published vs honest-rebuild gap.
Train 2 ML variants x 2 test-row conventions, then run the exp83-style
fixed-capital costless sim + my honest DCA sim on each:
  V0 = no embargo, test rows only where fwd12 label complete  (replicates exp78/83)
  V1 = no embargo, predict ALL liquid names                    (fixes test survivorship)
  V2 = embargo,    predict ALL liquid names                    (fixes both = my PROB)
Also report monthly IC vs fwd-3m for each.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)
t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, fok, liq, me, dv, bench, cols = D["FEAT"], D["fok"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
fnames = list(FEAT.keys())
Z = {nm: FEAT[nm].where(liq).rank(axis=1, pct=True).astype(np.float32) for nm in fnames}

recs = []
for dt in M[(M >= pd.Timestamp("2011-06-01"))]:
    fv = fok.loc[dt].dropna()
    if len(fv) < 60: continue
    q1, q2 = fv.quantile(1/3), fv.quantile(2/3)
    y = pd.Series(np.where(fv >= q2, 1, np.where(fv <= q1, 0, np.nan)), index=fv.index).dropna()
    X = np.column_stack([Z[nm].loc[dt].reindex(y.index).values for nm in fnames])
    for i, tk in enumerate(y.index):
        recs.append((dt, tk, *X[i], int(y.iloc[i])))
DF = pd.DataFrame.from_records(recs, columns=["date", "tk"] + fnames + ["y"])
p(f"samples {len(DF)} t={time.time()-t0:.0f}s")

from sklearn.ensemble import HistGradientBoostingClassifier

def train(embargo):
    pr = []
    for ytest in range(2015, 2027):
        cutoff = f"{ytest-1}-01-01" if embargo else f"{ytest}-01-01"
        tr = DF[DF.date < pd.Timestamp(cutoff)]
        if len(tr) < 5000: continue
        clf = HistGradientBoostingClassifier(max_iter=250, max_depth=4, learning_rate=0.05,
                                             l2_regularization=1.0, min_samples_leaf=200)
        clf.fit(tr[fnames].values, tr["y"].values)
        te_m = M[(M >= pd.Timestamp(f"{ytest}-01-01")) & (M <= pd.Timestamp(f"{ytest}-12-31"))]
        for dt in te_m:
            ok = liq.loc[dt]; idx = ok[ok].index
            X = np.column_stack([Z[nm].loc[dt].reindex(idx).values for nm in fnames])
            good = ~np.isnan(X).all(axis=1)
            if good.sum() == 0: continue
            probs = clf.predict_proba(np.nan_to_num(X[good], nan=0.5))[:, 1]
            pr.append(pd.DataFrame({"date": dt, "tk": idx[good], "p": probs}))
    return pd.concat(pr).pivot_table(index="date", columns="tk", values="p").reindex(M)

P_noemb = train(embargo=False)
p(f"trained no-embargo t={time.time()-t0:.0f}s")
P_emb = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M)   # embargo, all-names

surv_mask = fok.notna()
VARIANTS = {
    "V0 no-embargo, survivor-test-rows (replicates exp83)": P_noemb.where(surv_mask.reindex(columns=P_noemb.columns)),
    "V1 no-embargo, all-names": P_noemb,
    "V2 embargo, all-names (honest)": P_emb,
}

ma10 = me.rolling(10, min_periods=10).mean()
mom3 = me / me.shift(3) - 1
vol6 = FEAT["vol6"]
ret = (me / me.shift(1) - 1).clip(-0.9, 3.0)
idx = M[(M >= pd.Timestamp("2015-01-01")) & (M <= pd.Timestamp("2026-06-01"))]
qret = bench["QQQ"].pct_change().reindex(idx)
elig_orig = (liq & (me >= 3.0) & (me > ma10))   # exp83: no ADV filter

def stats(r):
    r = r.dropna()
    c = (1 + r).prod() ** (12 / len(r)) - 1
    s = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    eq = (1 + r).cumprod(); d = (eq / eq.cummax() - 1).min()
    return c, s, d

def sim83(PROB, N=12, trail=-0.30, cost=0.0, elig=elig_orig, accel_gate=False):
    """faithful exp83 fixed-capital sim (same-close exec, no min-hold, no costs by default)."""
    accel = PROB - PROB.shift(2)
    sc = PROB.where(elig & (mom3 > 0) & ((accel > 0) if accel_gate else True))
    rank = sc.rank(axis=1, ascending=False)
    didx = list(M)
    pos = {}; cash = 1.0; out = []
    for k, dt in enumerate(didx):
        px = me.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cpx = px.get(tk, np.nan)
            if not np.isfinite(cpx): pos.pop(tk); continue
            e["peak"] = max(e["peak"], cpx)
            if cpx / e["peak"] - 1 <= trail or cpx < ma10.loc[dt].get(tk, np.nan):
                cash += e["val"] * (1 - cost); pos.pop(tk)
        if dt in PROB.index:
            rk = rank.loc[dt]
            cands = [t for t in rk[rk <= N * 3].sort_values().index
                     if t not in pos and np.isfinite(px.get(t, np.nan))]
            need = N - len(pos)
            if need > 0 and cash > 1e-9 and cands:
                pick = cands[:need]
                for tk in pick:
                    a = cash / need
                    pos[tk] = {"px": px[tk], "peak": px[tk], "val": a * (1 - cost)}
                cash -= cash  # deploy all
        eq0 = cash + sum(e["val"] for e in pos.values())
        if k + 1 < len(didx):
            for tk in pos:
                r1 = ret.iloc[k + 1].get(tk, np.nan)
                pos[tk]["val"] *= (1 + (r1 if np.isfinite(r1) else -0.5))
        eq1 = cash + sum(e["val"] for e in pos.values())
        if dt >= idx[0] and dt <= idx[-1] and k + 1 < len(didx):
            out.append((didx[k + 1], eq1 / eq0 - 1 if eq0 > 0 else 0.0))
    return pd.Series(dict(out)).reindex(idx).fillna(0.0)

c, s, d = stats(qret); p(f"\n{'QQQ':58} {c:>7.1%} {s:>5.2f} {d:>7.1%}")
fwd3 = (me.shift(-3) / me - 1).clip(-0.9, 3.0)
for nm, PR in VARIANTS.items():
    PR = PR.reindex(columns=me.columns)
    # IC vs fwd3
    ics = []
    for dt in M[(M >= pd.Timestamp("2015-01-01")) & (M <= pd.Timestamp("2025-06-01"))][::2]:
        a = PR.loc[dt].dropna(); b = fwd3.loc[dt].reindex(a.index)
        dd = pd.concat([a, b], axis=1).dropna()
        if len(dd) > 100: ics.append(dd.iloc[:, 0].corr(dd.iloc[:, 1], method="spearman"))
    r = sim83(PR)
    c, s, d = stats(r)
    p(f"{nm:58} {c:>7.1%} {s:>5.2f} {d:>7.1%}  IC3 {np.nanmean(ics):+.3f}")
    r = sim83(PR, cost=0.002)
    c, s, d = stats(r)
    p(f"{'   + 20bps costs':58} {c:>7.1%} {s:>5.2f} {d:>7.1%}")
p(f"\nDONE t={time.time()-t0:.0f}s")
