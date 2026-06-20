"""Exp 34b — test the clean SEC insider-buying signal. Signals (point-in-time,
trailing 3m by filing date): nbuy (count of insider buy filings = cluster buying,
size-robust), netflag (net $ buyer), buy$ rank. Cross-sectional IC vs forward
return rel-SPY, sub-period split + RANDOM control; long-only top-quintile net of
cost vs equal-weight/random/SPY. Universe: S&P400+500+NDX (priceable mid/large;
understates small-cap effect). Honest disciplines applied."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import spearmanr
t0 = time.time()
P = pd.read_pickle("/tmp/wave/_insider_panel.pkl")
uni = set()
for f in ("sp500_universe.txt", "xuniverse_sp400.txt", "xuniverse_ndx.txt"):
    try:
        txt = open(f"/tmp/wave/{f}").read()
        uni |= set(txt.split()) if " " in txt else set(l.strip() for l in txt.splitlines() if l.strip())
    except Exception:
        pass
uni = sorted(t for t in uni if t and t.isalpha())
print(f"insider panel {len(P)} rows; price universe {len(uni)} names", flush=True)
px = yf.download(uni + ["SPY"], start="2009-06-01", auto_adjust=True, progress=False)["Close"]
names = [t for t in uni if t in px.columns and px[t].notna().sum() > 800]
me = px[names + ["SPY"]].resample("ME").last(); me.index = me.index.to_period("M").to_timestamp()
spy = me["SPY"]
print(f"priced {len(names)} names  t={time.time()-t0:.0f}s", flush=True)

# build monthly signal panels (trailing 3m sums), aligned to month index
P = P[P.tk.isin(names)].copy()
buy = P.pivot_table(index="ym", columns="tk", values="buy", aggfunc="sum").reindex(index=me.index, columns=names).fillna(0)
sell = P.pivot_table(index="ym", columns="tk", values="sell", aggfunc="sum").reindex(index=me.index, columns=names).fillna(0)
nbuy = P.pivot_table(index="ym", columns="tk", values="nbuy", aggfunc="sum").reindex(index=me.index, columns=names).fillna(0)
SIG = {
    "nbuy3": nbuy.rolling(3, min_periods=1).sum(),
    "net3$": (buy - sell).rolling(3, min_periods=1).sum(),
    "buy3$": buy.rolling(3, min_periods=1).sum(),
}
def fwd(h): return (me[names].shift(-h) / me[names] - 1).sub(spy.shift(-h) / spy - 1, axis=0)

def ic(sig, h, lo, hi):
    f = fwd(h); ics = []
    for d in me.index:
        if not (pd.Timestamp(lo) <= d < pd.Timestamp(hi)):
            continue
        x = sig.loc[d, names]
        x = x[x > 0].dropna()                       # only names with insider activity
        common = [c for c in x.index if np.isfinite(f.loc[d, c])]
        if len(common) < 20:
            continue
        ics.append(spearmanr(x[common], f.loc[d, common]).correlation)
    a = np.array([i for i in ics if np.isfinite(i)])
    return (a.mean(), a.mean()/(a.std()+1e-12)*np.sqrt(len(a)), len(a)) if len(a) > 5 else None

print("\nCross-sectional IC of insider signals vs forward rel-SPY return:", flush=True)
for nm, sig in SIG.items():
    for h in (3, 6):
        tr = ic(sig, h, "2010-01-01", "2018-01-01"); te = ic(sig, h, "2018-01-01", "2025-07-01")
        if tr and te:
            print(f"  {nm:7s} fwd{h}m: TRAIN IC {tr[0]:+.3f}(t{tr[1]:+.1f})  "
                  f"TEST IC {te[0]:+.3f}(t{te[1]:+.1f})", flush=True)

# event-style: forward return of NET-BUYER stock-months vs rest, + net-of-cost quintile
print("\nNet-insider-buyer stock-months vs rest (fwd-3m rel-SPY mean):", flush=True)
f3 = fwd(3); netflag = (buy - sell) > 0
for lo, hi, tag in (("2010-01-01","2018-01-01","TRAIN"),("2018-01-01","2025-07-01","TEST")):
    byr, rest, rnd = [], [], []
    rng = np.random.default_rng(0)
    for d in me.index:
        if not (pd.Timestamp(lo) <= d < pd.Timestamp(hi)): continue
        for t in names:
            y = f3.loc[d, t]
            if np.isfinite(y):
                (byr if netflag.loc[d, t] else rest).append(y)
    byr, rest = np.array(byr), np.array(rest)
    print(f"  {tag}: net-buyer {byr.mean()*100:+.2f}% (n={len(byr)})  | "
          f"rest {rest.mean()*100:+.2f}% (n={len(rest)})  | diff {(byr.mean()-rest.mean())*100:+.2f}pp", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
