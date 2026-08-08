"""Decay trajectory + VALIDATION for the survivors.

1. Era decomposition: z1-only and ladder daily books, gross Sharpe by 3-year
   era, 1996-2019 -> did the Sharpe-3 signal die, and when?
2. VAL (2015-2019) for the two surviving configs (first and only look):
   ladder ema3 lam0.3 REG and ema5 lam0.15 REG, at 2/5/10 bps.
3. Crisis-conditional variant: trade only when market vol percentile > 0.8
   (PIT), daily ladder at 2bps — does concentration-in-crisis rescue Sharpe?
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
idx = R.index
resid = pd.read_pickle("/tmp/sharpe3_work/_resid_daily.pkl")
E_d = ELIG.reindex(idx).ffill().fillna(False)[R.columns].astype(bool)
rvol = resid.rolling(63, min_periods=40).std().shift(1)
z1 = (resid/rvol).where(E_d).astype(np.float32)
z5 = (resid.rolling(5).sum()/(rvol*np.sqrt(5))).where(E_d).astype(np.float32)
z21 = (resid.rolling(21).sum()/(rvol*np.sqrt(21))).where(E_d).astype(np.float32)
a = np.array([0.455, 0.34, 0.205])
zC = (a[0]*(-z1) + a[1]*(-z5) + a[2]*(-z21)).clip(-3, 3)

mkt = R["SPY"] if "SPY" in R.columns else R.mean(axis=1)
v21 = mkt.rolling(21).std()
vpct = v21.expanding(252).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)

def prop_book(z):
    pos = z.clip(lower=0); neg = (-z).clip(lower=0)
    return 0.5*pos.div(pos.sum(axis=1), axis=0).fillna(0.0) - 0.5*neg.div(neg.sum(axis=1), axis=0).fillna(0.0)

def smooth(tgt, ema, lam, regime=False):
    z = tgt
    if regime:
        scale = (0.5 + vpct.clip(0, 1)).shift(1).fillna(1.0)
        z = z.mul(scale, axis=0)
    Tv = z.values; Wv = np.zeros_like(Tv); cur = np.zeros(Tv.shape[1])
    for ti in range(len(Tv)):
        cur = cur + lam*(Tv[ti]-cur); Wv[ti] = cur
    return pd.DataFrame(Wv, index=z.index, columns=z.columns)

# ---------- 1) era decomposition (gross, 0 fees) ----------
print("=== gross Sharpe by era, daily books (0 fees) ===")
Wz1 = prop_book((-z1).clip(-3, 3))
Wl = prop_book(zC.ewm(span=1, min_periods=1).mean())
for nm, W in [("z1 daily", Wz1), ("ladder daily", Wl)]:
    _, gross, _ = G.run(W, R, fee_bps=0)
    line = f"{nm:14}"
    for a0, b0 in [(1996, 1998), (1999, 2001), (2002, 2004), (2005, 2007), (2008, 2010),
                   (2011, 2013), (2014, 2016), (2017, 2019)]:
        s = G.sharpe(gross[f"{a0}":f"{b0}"])
        line += f"  {a0}-{str(b0)[2:]} {s:5.2f}"
    print(line, flush=True)

# ---------- 2) VAL for survivors ----------
print("=== VALIDATION 2015-2019 (first look) ===")
tgt = prop_book(zC.ewm(span=3, min_periods=1).mean())
for nm, ema, lam in [("ema3 lam0.3 REG", 3, 0.3), ("ema5 lam0.15 REG", 5, 0.15)]:
    base = prop_book(zC.ewm(span=ema, min_periods=1).mean())
    W = smooth(base, ema, lam, regime=True)
    line = f"{nm:18}"
    for fee in (2, 5, 10):
        net, gross, tno = G.run(W, R, fee_bps=fee)
        line += f"  {fee}bp DEV {G.sharpe(net['1996-06':'2014']):5.2f} VAL {G.sharpe(net['2015':'2019']):5.2f}"
    print(line, flush=True)

# ---------- 3) crisis-only daily ladder ----------
print("=== crisis-only (vpct>0.8) daily ladder ===")
on = (vpct.shift(1) > 0.8).astype(float)
Wc = Wl.mul(on, axis=0)
for fee in (0, 2, 5, 10):
    net, gross, tno = G.run(Wc, R, fee_bps=fee)
    dev = net["1996-06":"2014"]
    frac = float((Wc.abs().sum(axis=1)["1996-06":"2014"] > 0).mean())
    print(f"  {fee}bp: DEV full-period {G.sharpe(dev):.2f} (active {frac:.0%} of days, active-only {G.sharpe(dev[Wc.abs().sum(axis=1)['1996-06':'2014']>0]):.2f})", flush=True)
print(f"exp13 done t={time.time()-t0:.0f}s", flush=True)
