"""Exp 39 — THE real test of 'combine low-correlation sleeves for higher Sharpe'.
Combine the genuine OOS survivors: (1) QQQ beta, (2) IBS mean-reversion on QQQ
(real in-mkt Sharpe ~1.9, ~20% exposure), (3) insider-officer-buy tilt. Monthly,
net of cost. Show correlations + combined Sharpe/CAGR/DD vs QQQ, full + OOS.
Honest test of the diversification thesis (IC x sqrt(N)) with REAL sleeves."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
# --- sleeve 1+2: QQQ daily + IBS mean-reversion ---
raw = yf.download(["QQQ"], start="1999-03-01", auto_adjust=True, progress=False)
c, h, l = raw["Close"]["QQQ"].dropna(), raw["High"]["QQQ"], raw["Low"]["QQQ"]
h, l = h.reindex(c.index), l.reindex(c.index)
hl = h - l; ibs = (c - l) / hl.replace(0, np.nan)
lower = h.rolling(10).max() - 2.5 * hl.rolling(25).mean()
entry = (c < lower) & (ibs < 0.3); ex = c > h.shift(1)
pos = np.zeros(len(c)); inp = False
for i in range(len(c)):
    inp = True if (not inp and entry.iloc[i]) else (False if inp and ex.iloc[i] else inp)
    pos[i] = 1.0 if inp else 0.0
posl = pd.Series(pos, index=c.index).shift(1).fillna(0)
mr_d = posl * c.pct_change() - 0.0005 * posl.diff().abs().fillna(0)
def to_m(s):
    g = (1 + s).groupby(s.index.to_period("M")).prod() - 1
    g.index = g.index.to_timestamp(); return g
qqq_m = to_m(c.pct_change()); mr_m = to_m(mr_d)
# --- sleeve 3: insider-officer tilt (cached) ---
me = pd.read_pickle("/tmp/wave/_ins_px.pkl"); names = [x for x in me.columns if x not in ("SPY", "QQQ")]
P = pd.read_pickle("/tmp/wave/_insider_rich.pkl"); P = P[P.tk.isin(names)]
mret = me.pct_change()
off = (P.pivot_table(index="ym", columns="tk", values="off_buy", aggfunc="sum")
       .reindex(index=me.index, columns=names).fillna(0).rolling(3, min_periods=1).sum() > 0)
prev = set(); rr = []
for i in range(3, len(me.index) - 1):
    d = me.index[i]; nxt = me.index[i + 1]
    sel = [t for t in names if off.loc[d, t] and np.isfinite(mret.loc[nxt, t])]
    if len(sel) < 5: continue
    turn = 1 - len(prev & set(sel)) / max(len(sel), 1)
    rr.append((nxt, mret.loc[nxt, sel].mean() - turn * 0.002)); prev = set(sel)
ins_m = pd.Series(dict(rr)).dropna()

idx = qqq_m.index.intersection(mr_m.index).intersection(ins_m.index)
D = pd.DataFrame({"QQQ": qqq_m[idx], "MeanRev": mr_m[idx], "Insider": ins_m[idx]}).dropna()
print(f"combined monthly panel {len(D)} months {D.index[0].date()}->{D.index[-1].date()}", flush=True)
print("\ncorrelation matrix:\n", D.corr().round(2).to_string(), flush=True)

def stats(s, lo=None, hi=None):
    if lo: s = s[(s.index >= lo) & (s.index < hi)]
    s = s.dropna(); eq = (1 + s).cumprod(); yrs = len(s) / 12
    return eq.iloc[-1] ** (1/yrs) - 1, s.mean()/(s.std()+1e-12)*np.sqrt(12), float((eq/eq.cummax()-1).min())

ivol = 1 / D.std(); ivw = ivol / ivol.sum()
combos = {
    "100% QQQ": D.QQQ,
    "equal 1/3 each": D.mean(axis=1),
    "60 QQQ /20 MR /20 Ins": 0.6*D.QQQ + 0.2*D.MeanRev + 0.2*D.Insider,
    "inverse-vol weighted": (D * ivw).sum(axis=1),
}
print("\nCombined portfolios (monthly rebal):", flush=True)
for nm, s in combos.items():
    c_, sh, dd = stats(s)
    c1, sh1, _ = stats(s, "2010-01-01", "2018-01-01"); c2, sh2, _ = stats(s, "2018-01-01", "2026-12-31")
    print(f"  {nm:24s} CAGR {c_*100:5.1f}%  Sharpe {sh:.2f}  maxDD {dd*100:4.0f}%  "
          f"[Sh 10-17 {sh1:.2f} | 18-25 {sh2:.2f}]", flush=True)
print(f"  (inverse-vol weights: {dict(ivw.round(2))})", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
