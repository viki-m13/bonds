"""Exp 41 — proper market-neutral StatArb (residual reversal, Avellaneda-Lee /
Da-Liu-Schaumburg), the canonical institutional StatArb. Hedge out market beta,
trade weekly mean-reversion of the idiosyncratic RESIDUAL. Long bottom-quintile
residual (oversold), short top (overbought), dollar/beta-neutral. Net of cost.
Key question: is it a REAL, ~0-corr-to-QQQ sleeve that lifts the ensemble
(exp40's ceiling was from equity-correlated sleeves)? Long-short = needs shorting.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
names = sorted(set(open("/tmp/wave/sp500_universe.txt").read().split()))
raw = yf.download(names + ["SPY", "QQQ"], start="2009-01-01", auto_adjust=True, progress=False)["Close"]
names = [t for t in names if t in raw.columns and raw[t].notna().sum() > 1500]
C = raw[names]; spy = raw["SPY"]
ret = C.pct_change()
print(f"{len(names)} names  t={time.time()-t0:.0f}s", flush=True)
# proper Avellaneda-Lee: top-K PCA eigen-factors, trade residual reversal
wk = C.index[::5]; K = 5; COST = float(__import__("os").environ.get("COST","0.001"))
rows = []; prevL = prevS = set()
for i in range(13, len(wk) - 1):
    d = wk[i]; nxt = wk[i + 1]
    win = ret.loc[:d].iloc[-60:]
    valid = [t for t in names if win[t].notna().all()
             and np.isfinite(C.loc[nxt, t]) and np.isfinite(C.loc[d, t])]
    if len(valid) < 50:
        continue
    R = win[valid].values
    Rz = (R - R.mean(0)) / (R.std(0) + 1e-9)                # standardize per stock
    U, S, Vt = np.linalg.svd(Rz, full_matrices=False)
    F = U[:, :K]                                            # orthonormal factor returns (60xK)
    resid = Rz - F @ (F.T @ Rz)                             # idiosyncratic residual
    score = resid[-5:].sum(0)                               # past-week residual (s-score proxy)
    s = pd.Series(score, index=valid)
    z = (s - s.mean()) / (s.std() + 1e-12)
    longs = list(z[z <= z.quantile(0.2)].index); shorts = list(z[z >= z.quantile(0.8)].index)  # reversal
    fr = (C.loc[nxt] / C.loc[d] - 1)
    rL = fr[longs].mean(); rS = fr[shorts].mean()
    turn = (1 - len(prevL & set(longs))/max(len(longs),1)) + (1 - len(prevS & set(shorts))/max(len(shorts),1))
    rows.append((nxt, (rL - rS) - turn * COST)); prevL, prevS = set(longs), set(shorts)
sa = pd.Series(dict(rows)).dropna()                         # weekly long-short return
def st(s, lo=None, hi=None, per=52):
    if lo: s = s[(s.index >= lo) & (s.index < hi)]
    s = s.dropna(); eq = (1 + s).cumprod(); yrs = len(s) / per
    return eq.iloc[-1] ** (1/yrs) - 1, s.mean()/(s.std()+1e-12)*np.sqrt(per), float((eq/eq.cummax()-1).min())
print("\nResidual-reversal StatArb (weekly, market-neutral, net 10bps/side):", flush=True)
for tag, lo, hi in (("FULL","2009-01-01","2026-12-31"),("2009-15","2009-01-01","2015-01-01"),
                    ("2015-19","2015-01-01","2019-01-01"),("2019-26","2019-01-01","2026-12-31")):
    c,sh,dd = st(sa,lo,hi); print(f"  {tag:8s} CAGR {c*100:5.1f}%  Sharpe {sh:.2f}  maxDD {dd*100:4.0f}%", flush=True)
# correlation to QQQ (weekly) and ensemble lift (monthly)
qqqw = qqq = raw["QQQ"]; qqq_wk = (qqqw.reindex(sa.index, method="ffill").pct_change())
print(f"\n  corr to QQQ (weekly): {sa.corr(qqq_wk):.2f}", flush=True)
sa_m = ((1+sa).groupby(sa.index.to_period("M")).prod()-1); sa_m.index = sa_m.index.to_timestamp()
# pull the exp40 ensemble sleeves quickly (monthly) to test the lift
import subprocess
try:
    D = pd.read_pickle("/tmp/wave/_ensemble_D.pkl")
    D2 = D.join(sa_m.rename("StatArb"), how="inner").dropna()
    print("\n  corr matrix with StatArb added:\n", D2.corr().round(2).to_string(), flush=True)
    iv = (1/D2.std())/(1/D2.std()).sum()
    for nm, s in (("prev best 3-sleeve", 0.6*D2.QQQ+0.2*D2.MeanRev+0.2*D2.Insider),
                  ("+StatArb inverse-vol", (D2*iv).sum(axis=1))):
        c,sh,dd = st(s, per=12); print(f"  {nm:24s} CAGR {c*100:5.1f}% Sharpe {sh:.2f} maxDD {dd*100:4.0f}%", flush=True)
    print(f"  weights: {dict((iv*100).round(0).astype(int))}", flush=True)
except Exception as e:
    print("  (ensemble panel not cached; standalone only)", e, flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
sa_m.to_pickle("/tmp/wave/_statarb_m.pkl")
