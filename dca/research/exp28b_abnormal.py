"""Exp 28b — harden the FINRA short-vol signal. Use ABNORMAL short volume
(each stock's SVR z-scored vs its OWN trailing 12m history -> isolates the
informed time-varying component, removes structural MM-hedging differences).
Test sign STABILITY across 2-year sub-periods (is the edge robust or regime-
luck?). Raw vs abnormal, with random control."""
import warnings, time, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import spearmanr
t0 = time.time()
SVR = pd.read_pickle("/tmp/wave/_shortvol.pkl")          # month x symbol
names = list(SVR.columns)
pxf = "/tmp/wave/_shortvol_px.pkl"
if os.path.exists(pxf):
    me = pd.read_pickle(pxf)
else:
    px = yf.download(names + ["QQQ"], start="2016-06-01", auto_adjust=True,
                     progress=False)["Close"]
    me = px.resample("ME").last(); me.index = me.index.to_period("M").to_timestamp()
    me.to_pickle(pxf)
q = me["QQQ"]
print(f"SVR {SVR.shape}, prices {me.shape}  t={time.time()-t0:.0f}s", flush=True)

# abnormal SVR: per-stock time-series z-score vs trailing 12m
absvr = (SVR - SVR.rolling(12, min_periods=6).mean()) / SVR.rolling(12, min_periods=6).std()


def fwd(h): return (me.shift(-h) / me - 1).sub(q.shift(-h) / q - 1, axis=0)


def test(sig, lo, hi, h):
    f = fwd(h); ics, qs, rs = [], [], []
    rng = np.random.default_rng(0)
    for d in sig.index:
        if not (pd.Timestamp(lo) <= d < pd.Timestamp(hi)):
            continue
        s = sig.loc[d].dropna()
        common = [c for c in s.index if c in f.columns and np.isfinite(f.loc[d, c])]
        if len(common) < 40:
            continue
        x = s[common]; y = f.loc[d, common]
        ics.append(spearmanr(x, y).correlation)
        qs.append(y[x <= x.quantile(0.2)].mean() - y[x >= x.quantile(0.8)].mean())
        n5 = max(5, len(common) // 5)
        rs.append(np.mean([y.sample(n5, random_state=int(rng.integers(1e6))).mean()
                           - y.sample(n5, random_state=int(rng.integers(1e6))).mean()
                           for _ in range(8)]))
    if len(ics) < 5:
        return None
    ic = np.array(ics); qa = np.array(qs); ra = np.array(rs)
    return (ic.mean(), ic.mean()/(ic.std()+1e-12)*np.sqrt(len(ic)),
            qa.mean()*100, qa.mean()/(qa.std()+1e-12)*np.sqrt(len(qa)), ra.mean()*100, len(ic))


print("\nSign-stability across sub-periods (fwd-3m; low-minus-high quintile %/mo):",
      flush=True)
print(f"{'period':14s} | {'RAW SVR: IC / lo-hi% / t':>34s} | {'ABNORMAL SVR: IC / lo-hi% / t':>36s}",
      flush=True)
for lo, hi in (("2018-06-01", "2020-06-01"), ("2020-06-01", "2022-06-01"),
               ("2022-06-01", "2024-06-01"), ("2024-06-01", "2026-06-01")):
    rraw = test(SVR, lo, hi, 3); rab = test(absvr, lo, hi, 3)
    def fmt(r):
        return f"IC{r[0]:+.3f} {r[2]:+.2f}%/mo t{r[3]:+.1f}" if r else "n/a"
    print(f"  {lo[:7]}-{hi[:7]} | {fmt(rraw):>34s} | {fmt(rab):>36s}", flush=True)

print("\nFull TEST 2021-2026 (random control in parens):", flush=True)
for nm, sig in (("raw SVR", SVR), ("abnormal SVR", absvr)):
    for h in (1, 3):
        r = test(sig, "2021-01-01", "2026-06-01", h)
        if r:
            print(f"  {nm:13s} fwd{h}m: IC {r[0]:+.4f} (t {r[1]:+.1f})  "
                  f"low-minus-high {r[2]:+.2f}%/mo (t {r[3]:+.1f})  [random {r[4]:+.2f}%]  n={r[5]}",
                  flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
