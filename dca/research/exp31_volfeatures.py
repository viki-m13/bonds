"""Exp 31 — INVENT & test volume/volatility cross-sectional features for alpha.
Documented: low-vol, beta(BAB), MAX(lottery), idio-vol, Amihud illiquidity,
realized-skew. INVENTED: accum (vol-weighted range-close, Chaikin-style),
volvol_div (quiet accumulation = high volume + low vol), hv_lv_asym (return on
high-vol days minus low-vol days = informed-day asymmetry), obv_mom (signed-
volume momentum). Cross-sectional IC vs forward rel-QQQ return, with SUB-PERIOD
sign-stability + RANDOM-feature control. Universe: broad S&P (survivorship-
biased -> inflates, conservative for negatives)."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import spearmanr
t0 = time.time()
names = sorted(set(open("/tmp/wave/sp500_universe.txt").read().split()))
raw = yf.download(names + ["QQQ", "SPY"], start="2009-01-01", auto_adjust=True, progress=False)
C, H, L, V = raw["Close"], raw["High"], raw["Low"], raw["Volume"]
names = [t for t in names if t in C.columns and C[t].notna().sum() > 1500]
C, H, L, V = C[names], H[names], L[names], V[names]
print(f"{len(names)} names {C.index[0].date()}->{C.index[-1].date()}  t={time.time()-t0:.0f}s", flush=True)
r = C.pct_change(); spy = raw["Close"]["SPY"].pct_change(); q = raw["Close"]["QQQ"]
dv = C * V

F = {}
F["vol60"] = r.rolling(60).std()                                   # low-vol (expect IC<0)
F["beta60"] = r.rolling(60).cov(spy).div(spy.rolling(60).var(), axis=0)  # BAB (IC<0)
F["max21"] = r.rolling(21).max()                                   # lottery (IC<0)
F["amihud"] = (r.abs() / dv).rolling(21).mean()                    # illiquidity (IC>0)
F["skew21"] = r.rolling(42).skew()                                 # neg-skew pref (IC<0?)
# invented:
clv = (((C - L) - (H - C)) / (H - L).replace(0, np.nan))           # close location in range
F["accum"] = (clv * V).rolling(21).sum() / V.rolling(21).sum()     # accumulation (IC>0)
zv = (V / V.rolling(60).mean()); zvz = (zv - zv.rolling(120).mean()) / zv.rolling(120).std()
volz = (F["vol60"] - F["vol60"].rolling(120).mean()) / F["vol60"].rolling(120).std()
F["volvol_div"] = zvz * (-volz)                                    # high vol + low realized vol
F["obv_mom"] = (np.sign(r) * dv).rolling(21).sum() / dv.rolling(21).sum()  # signed-vol momentum
# informed-day asym (vectorized): volume-weighted return minus equal-weighted
# return over 60d -> positive if gains concentrate on high-volume days (accumulation)
F["vw_premium"] = ((r * V).rolling(60).sum() / V.rolling(60).sum()
                   - r.rolling(60).mean())                         # informed-day asym (IC>0)
print(f"features built t={time.time()-t0:.0f}s", flush=True)

me = C.resample("ME").last().index
me = [d for d in me if d in C.index]
def fwd(h):
    f = (C.shift(-h) / C - 1).sub(q.shift(-h) / q - 1, axis=0)
    return f

def ic_period(feat, h, lo, hi):
    f = fwd(h); ics = []
    for d in me:
        if not (pd.Timestamp(lo) <= d < pd.Timestamp(hi)):
            continue
        x = feat.loc[d].replace([np.inf, -np.inf], np.nan).dropna()
        common = [c for c in x.index if c in f.columns and np.isfinite(f.loc[d, c])]
        if len(common) < 50:
            continue
        ics.append(spearmanr(x[common], f.loc[d, common]).correlation)
    ics = np.array([i for i in ics if np.isfinite(i)])
    if len(ics) < 5:
        return None
    return ics.mean(), ics.mean() / (ics.std() + 1e-12) * np.sqrt(len(ics)), len(ics)

print("\nfwd-63d cross-sectional IC by sub-period (sign stability is the test):", flush=True)
print(f"{'feature':12s} {'2010-2017 IC/t':>16s} {'2018-2025 IC/t':>16s}  stable?", flush=True)
for name, feat in F.items():
    a = ic_period(feat, 63, "2010-01-01", "2018-01-01")
    b = ic_period(feat, 63, "2018-01-01", "2026-06-01")
    if a and b:
        stable = "YES" if (np.sign(a[0]) == np.sign(b[0]) and abs(a[1]) > 1.5 and abs(b[1]) > 1.5) else "no"
        print(f"{name:12s} {a[0]:+.3f}/{a[1]:+4.1f}      {b[0]:+.3f}/{b[1]:+4.1f}       {stable}", flush=True)
# random-feature control
rng = np.random.default_rng(0)
randf = pd.DataFrame(rng.standard_normal(C.shape), index=C.index, columns=C.columns)
rc = ic_period(randf, 63, "2018-01-01", "2026-06-01")
print(f"{'RANDOM ctrl':12s} {'--':>16s} {rc[0]:+.3f}/{rc[1]:+4.1f}       (should be ~0)", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
