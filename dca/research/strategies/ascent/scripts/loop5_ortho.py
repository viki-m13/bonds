"""LOOP iter 5 — is there ANY idiosyncratic (factor-orthogonal) skill in our
best signal? Per month, regress the honest ML score cross-sectionally on the
common factor exposures (mom12, mom6, size=log dv, vol6, roa, distHigh, trend)
and keep the RESIDUAL. Then:
  1. IC of raw score vs fwd12 rank; IC of residual score. (both 2015-2025)
  2. Precision: P(top-5 pick lands in true fwd-12m top decile) — raw vs
     residual vs random vs synthetic IC~0.05 (the error-structure comparison).
  3. Era backtests of the residual score through the mandate harness.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)
from engine import dca_benchmark, stats

t0 = time.time()
def p(*a): print(*a, flush=True)
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
qqq = bench["QQQ"].reindex(M)
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M).reindex(columns=cols)
dvr = dv.rank(axis=1, ascending=False)
top1500 = dvr <= 1500
fwd12 = (me.shift(-12) / me - 1)
FR = fwd12.where(liq & top1500).rank(axis=1, pct=True)

FACT = {
    "mom12": (me / me.shift(12) - 1), "mom6": (me / me.shift(6) - 1),
    "size": np.log(dv.clip(lower=1)), "vol6": FEAT["vol6"],
    "roa": FEAT["roa"], "distHigh": FEAT["distHigh"],
    "trend": FEAT["trend"],
}
FR_rank = {k: v.where(liq & top1500).rank(axis=1, pct=True) for k, v in FACT.items()}

# per-month cross-sectional residualization of the ML score
resid_rows = {}
for dt in M[(M >= pd.Timestamp("2015-01-01")) & (M <= pd.Timestamp("2026-06-01"))]:
    y = PROB.loc[dt].where(liq.loc[dt] & top1500.loc[dt]).dropna()
    if len(y) < 200: continue
    X = pd.DataFrame({k: FR_rank[k].loc[dt].reindex(y.index) for k in FR_rank}).fillna(0.5)
    X["const"] = 1.0
    Xv = X.values
    beta, *_ = np.linalg.lstsq(Xv, y.values, rcond=None)
    resid_rows[dt] = pd.Series(y.values - Xv @ beta, index=y.index)
RES = pd.DataFrame(resid_rows).T.reindex(M).reindex(columns=cols)
p(f"residualized {len(resid_rows)} months  t={time.time()-t0:.0f}s")

def ic_of(S, lo="2015-01-01", hi="2025-06-01"):
    ics = []
    for dt in M[(M >= lo) & (M <= hi)][::2]:
        if dt not in S.index: continue
        a = S.loc[dt].dropna(); b = FR.loc[dt].reindex(a.index)
        d = pd.concat([a, b], axis=1).dropna()
        if len(d) > 200: ics.append(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
    return np.nanmean(ics), np.nanstd(ics) / np.sqrt(len(ics)) if ics else np.nan

def precision5(S, lo="2015-01-01", hi="2025-06-01"):
    """P(a top-5 pick by S lands in the true fwd-12m top decile), eligible pond."""
    hits = tot = 0
    for dt in M[(M >= lo) & (M <= hi)]:
        if dt not in S.index: continue
        el = liq.loc[dt] & top1500.loc[dt] & (me.loc[dt] >= 3) & (me.loc[dt] > ma10.loc[dt])
        s = S.loc[dt].where(el).dropna()
        f = FR.loc[dt].reindex(s.index).dropna()
        s = s.reindex(f.index)
        if len(s) < 200: continue
        top5 = s.nlargest(5).index
        hits += (f.reindex(top5) >= 0.9).sum(); tot += len(top5)
    return hits / tot if tot else np.nan

icr, se = ic_of(PROB); p(f"ML raw IC(fwd12 rank): {icr:+.3f} (se {se:.3f})")
ico, se = ic_of(RES);  p(f"ML residual (factor-orthogonal) IC: {ico:+.3f} (se {se:.3f})")
rng = np.random.default_rng(5)
NOISE = pd.DataFrame(rng.random(me.shape), index=M, columns=cols).where(FR.notna())
S05 = 0.03 * FR + 0.97 * NOISE   # ~IC 0.05 random-error synthetic
p(f"\nPrecision P(top-5 pick in true top decile):")
p(f"  random:            {precision5(NOISE):.1%}")
p(f"  ML raw:            {precision5(PROB):.1%}")
p(f"  ML residual:       {precision5(RES):.1%}")
p(f"  synthetic IC~0.05: {precision5(S05):.1%}")

# era backtest of residual (2015+ only; PROB starts 2015)
def run_static(pick, start, end, k=5, contrib=1000.0, cost=0.002, trail=-0.30):
    dates = M[(M >= start) & (M <= end)]
    elig_all = (liq & top1500 & (me >= 3.0) & (me > ma10))
    pos = {}; qqq_units = 0.0; qqq_entry = None; cash = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        prow = me.loc[dt]; qp = qqq.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= 0.75; cash += e["val"] * (1 - cost); pos.pop(tk)
        srow = pick.loc[dt] if dt in pick.index else pd.Series(dtype=float)
        erow = elig_all.loc[dt]; marow = ma10.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < 30: continue
            cp = e["last_px"]
            if (cp / e["peak_px"] - 1) <= trail or (np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk]):
                cash += e["val"] * (1 - cost); pos.pop(tk)
        cash += contrib; contributed += contrib
        if len(srow):
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

p("\nEra backtests (top-5, mandate mechanics):")
for nm, S in [("ML raw", PROB), ("ML residual", RES)]:
    for st, en in [("2015-01", "2019-12"), ("2020-01", "2026-06"), ("2015-01", "2026-06")]:
        eq, dts = run_static(S, pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01"))
        b = dca_benchmark(bench["QQQ"], dts)
        p(f"  {nm:12} {st}..{en}: {stats(eq)['final']/b['V'].iloc[-1]:5.2f}x")
p(f"\nDONE t={time.time()-t0:.0f}s")
