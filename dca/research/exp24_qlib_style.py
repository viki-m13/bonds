"""Experiment 24 — QLIB's flagship method (Alpha158-style factors + LightGBM),
adapted to equities / QQQ. Universe: current Nasdaq-100 (survivorship-biased,
which INFLATES results -> a conservative test: if ML still can't beat QQQ with
that tailwind, the negative is robust). Walk-forward by year. Reports Qlib's
headline metric (cross-sectional rank IC) AND the practical one: does a
long-only top-decile portfolio, rebalanced monthly, beat QQQ buy&hold?"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import lightgbm as lgb
from scipy.stats import spearmanr
t0 = time.time()
tickers = [l.strip() for l in open("/tmp/wave/xuniverse_ndx.txt") if l.strip()]
tickers = sorted(set(tickers + ["QQQ"]))
raw = yf.download(tickers, start="2008-01-01", auto_adjust=True, progress=False)
C = raw["Close"]; H = raw["High"]; L = raw["Low"]; V = raw["Volume"]
tickers = [t for t in tickers if t in C.columns and C[t].notna().sum() > 1500]
C, H, L, V = C[tickers], H[tickers], L[tickers], V[tickers]
print(f"{len(tickers)} NDX names, {C.index[0].date()}->{C.index[-1].date()}"
      f"  t={time.time()-t0:.0f}s", flush=True)
r1 = C.pct_change()

# ---- Alpha158-style factor family ----
F = {}
for k in (1, 5, 10, 20, 40, 60):
    F[f"ret{k}"] = C / C.shift(k) - 1
for k in (5, 10, 20, 60):
    F[f"ma{k}"] = C / C.rolling(k).mean() - 1
    F[f"std{k}"] = r1.rolling(k).std()
for k in (5, 20):
    F[f"vma{k}"] = V / V.rolling(k).mean() - 1
    rng = H.rolling(k).max() - L.rolling(k).min()
    F[f"pos{k}"] = (C - L.rolling(k).min()) / rng.replace(0, np.nan)
F["rng"] = (H - L) / C
F["maxret20"] = r1.rolling(20).max()
F["minret20"] = r1.rolling(20).min()
up = r1.clip(lower=0).rolling(14).mean(); dn = (-r1.clip(upper=0)).rolling(14).mean()
F["rsi14"] = up / (up + dn + 1e-12)
q = C["QQQ"]
F["corrqqq"] = r1.rolling(60).corr(q.pct_change())
FEATS = list(F.keys())
fwd = (C.shift(-21) / C - 1)                          # forward ~1m return (target)

# month-end sampling
me = C.resample("ME").last().index
me = [d for d in me if d in C.index]
rows = []
for d in me:
    for t in tickers:
        if t == "QQQ":
            continue
        x = {f: F[f].at[d, t] for f in FEATS}
        x["y"] = fwd.at[d, t]; x["date"] = d; x["tk"] = t
        rows.append(x)
D = pd.DataFrame(rows).dropna(subset=FEATS)
# cross-sectional rank-normalize features and demean target per date (Qlib-style)
for f in FEATS:
    D[f] = D.groupby("date")[f].rank(pct=True)
D["yx"] = D["y"] - D.groupby("date")["y"].transform("mean")
print(f"panel {len(D)} rows  t={time.time()-t0:.0f}s", flush=True)

# ---- walk-forward by year: train < Y, test == Y ----
ics, strat_ret, qqq_ret = [], {}, {}
for Y in range(2015, 2026):
    tr = D[D.date < pd.Timestamp(Y, 1, 1)].dropna(subset=["yx"])
    te = D[(D.date >= pd.Timestamp(Y, 1, 1)) & (D.date < pd.Timestamp(Y + 1, 1, 1))]
    if len(tr) < 2000 or len(te) == 0:
        continue
    m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
                          verbose=-1)
    m.fit(tr[FEATS], tr["yx"])
    te = te.copy(); te["pred"] = m.predict(te[FEATS])
    for d, g in te.groupby("date"):
        if len(g) >= 20 and g["y"].notna().any():
            ic = spearmanr(g["pred"], g["y"]).correlation
            if np.isfinite(ic):
                ics.append(ic)
            # long-only top-decile, equal weight, hold 1 month
            k = max(5, len(g) // 10)
            top = g.sort_values("pred", ascending=False).head(k)
            strat_ret[d] = top["y"].mean()
            qqq_ret[d] = fwd.at[d, "QQQ"] if d in fwd.index else np.nan

ics = np.array(ics)
print(f"\nQlib-style LightGBM (Alpha158-like), walk-forward 2015-2025:", flush=True)
print(f"  cross-sectional rank IC: mean {ics.mean():+.4f}  ICIR "
      f"{ics.mean()/(ics.std()+1e-12):+.3f}  hit {100*(ics>0).mean():.0f}%  n={len(ics)}",
      flush=True)
s = pd.Series(strat_ret).dropna(); qd = pd.Series(qqq_ret).reindex(s.index)
# compound monthly top-decile vs QQQ over test period
def stats(x):
    eq = (1 + x).cumprod(); yrs = len(x) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    return eq.iloc[-1], cagr, mdd
te_eq, te_c, te_d = stats(s); qe, qc, qd2 = stats(qd)
print(f"\n  Long-only TOP-DECILE (ML-picked) vs QQQ, 2015+ (monthly rebal):", flush=True)
print(f"    top-decile: {te_eq:.2f}x  CAGR {te_c*100:.1f}%  maxDD {te_d*100:.0f}%", flush=True)
print(f"    QQQ       : {qe:.2f}x  CAGR {qc*100:.1f}%  maxDD {qd2*100:.0f}%", flush=True)
print(f"    monthly: top-decile beats QQQ {100*(s>qd).mean():.0f}% of months", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
