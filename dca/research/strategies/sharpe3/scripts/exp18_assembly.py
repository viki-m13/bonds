"""The final max-assembly: every sleeve with positive gross, one netted book.

zA = 0.5*ladder(z1,z5,z21) + 0.25*slowML-proxy(z63 ladder ext) + 0.15*peer-gap
     + 0.10*leadlag   (weights heuristic, declared BEFORE evaluation)
Machinery: EMA5 + lam0.15 partial adjustment (the surviving execution),
regime scale 0.5+volpct. Evaluated on DEV then DEV2/2015-19, fees 0/2/5/10.
This is the ceiling-assembly: if THIS can't approach 3, nothing in the space can.
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
z1 = (resid/rvol).where(E_d)
z5 = (resid.rolling(5).sum()/(rvol*np.sqrt(5))).where(E_d)
z21 = (resid.rolling(21).sum()/(rvol*np.sqrt(21))).where(E_d)
z63 = (resid.rolling(63).sum()/(rvol*np.sqrt(63))).where(E_d)
L = np.log1p(R)
r5 = np.expm1(L.rolling(5).sum())
DV63 = DV.rolling(63, min_periods=40).median()
lead100 = DV63.rank(axis=1, ascending=False) <= 100
lead_ret = (r5.where(lead100)).mean(axis=1)
zll_raw = -(r5.sub(lead_ret, axis=0))
zll = (zll_raw.sub(zll_raw.mean(axis=1), axis=0)).div(zll_raw.std(axis=1)+1e-12, axis=0).where(E_d)
r10 = np.expm1(L.rolling(10).sum())
zpg_raw = -(r10.sub(r10.where(E_d).mean(axis=1), axis=0))   # cheap peer-gap proxy: vs eligible mean
zpg = (zpg_raw.sub(zpg_raw.mean(axis=1), axis=0)).div(zpg_raw.std(axis=1)+1e-12, axis=0).where(E_d)

lad = (0.455*(-z1) + 0.34*(-z5) + 0.205*(-z21))
zA = (0.5*lad + 0.25*(-z63) + 0.15*zpg + 0.10*zll).clip(-3, 3)

mkt = R["SPY"]
vpct = mkt.rolling(21).std().rolling(756, min_periods=252).rank(pct=True)
scale = (0.5 + vpct.clip(0, 1)).shift(1).fillna(1.0)

z = zA.ewm(span=5, min_periods=1).mean()
pos = z.clip(lower=0); neg = (-z).clip(lower=0)
tgt = 0.5*pos.div(pos.sum(axis=1), axis=0).fillna(0.0) - 0.5*neg.div(neg.sum(axis=1), axis=0).fillna(0.0)
tgt = tgt.mul(scale, axis=0)
Tv = tgt.values; Wv = np.zeros_like(Tv); cur = np.zeros(Tv.shape[1])
for ti in range(len(Tv)):
    cur = cur + 0.15*(Tv[ti]-cur); Wv[ti] = cur
W = pd.DataFrame(Wv, index=idx, columns=R.columns)
for a0, b0 in [("1996-06", "2014"), ("2005", "2014"), ("2015", "2019")]:
    line = f"assembly {a0}-{b0}:"
    for fee in (0, 2, 5, 10):
        net, g_, t_ = G.run(W, R, fee_bps=fee)
        line += f"  {fee}bp {G.sharpe(net[a0:b0]):5.2f}"
    line += f"  (tno {t_[a0:b0].mean():.3f}, gross {G.sharpe(g_[a0:b0]):.2f})"
    print(line, flush=True)
print(f"exp18 done t={time.time()-t0:.0f}s", flush=True)
