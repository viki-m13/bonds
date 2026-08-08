"""Invention track J: the gross-ceiling push + the cost frontier.

Book: ALL eligible names, proportional to composite z (reversal ladder):
    zC = a1*(-z1) + a5*(-z5) + a21*(-z21)    (weights fit on 1996-2004 only,
                                              then FROZEN for 2005-2014)
Execution: EMA(span) smoothing + partial adjustment lambda, daily.
Overlays: vol-regime scaling (market 21d vol percentile, PIT) on gross.
Deliverable: DEV Sharpe at fees 0/2/5/10/20 bps -> the honest frontier.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
DEV_END = pd.Timestamp("2014-12-31")
idx = R.index
resid = pd.read_pickle("/tmp/sharpe3_work/_resid_daily.pkl")
E_d = ELIG.reindex(idx).ffill().fillna(False)[R.columns].astype(bool)
rvol = resid.rolling(63, min_periods=40).std().shift(1)
z1 = (resid/rvol).where(E_d).astype(np.float32)
z5 = (resid.rolling(5).sum()/(rvol*np.sqrt(5))).where(E_d).astype(np.float32)
z21 = (resid.rolling(21).sum()/(rvol*np.sqrt(21))).where(E_d).astype(np.float32)
print(f"signals t={time.time()-t0:.0f}s", flush=True)

# ---- fit ladder weights on 1996-2004 via per-sleeve gross Sharpe (no peeking past 2004)
fit_mask = (idx >= pd.Timestamp("1996-06-01")) & (idx <= pd.Timestamp("2004-12-31"))
def quick_gross(zmat):
    z = zmat.where(pd.DataFrame(fit_mask[:, None] if False else np.broadcast_to(fit_mask[:, None], zmat.shape), index=idx, columns=zmat.columns))
    z = zmat[fit_mask[0]:] if False else zmat
    # proportional book, daily, no costs, on fit window only
    zz = zmat.clip(-3, 3)
    pos = zz.clip(lower=0); neg = (-zz).clip(lower=0)
    Wp = 0.5*pos.div(pos.sum(axis=1), axis=0).fillna(0.0) - 0.5*neg.div(neg.sum(axis=1), axis=0).fillna(0.0)
    Wp = Wp[pd.Series(fit_mask, index=idx)]
    net, gross, tno = G.run(Wp, R, fee_bps=0.0)
    return G.sharpe(gross[gross.index <= pd.Timestamp("2004-12-31")])
s1 = quick_gross(-z1); s5 = quick_gross(-z5); s21 = quick_gross(-z21)
print(f"fit-window gross Sharpes: z1 {s1:.2f}  z5 {s5:.2f}  z21 {s21:.2f}", flush=True)
a = np.array([max(s1,0), max(s5,0), max(s21,0)]); a = a/a.sum() if a.sum()>0 else np.array([0,1,0])
print(f"frozen ladder weights: {a.round(3)}", flush=True)

zC = (a[0]*(-z1) + a[1]*(-z5) + a[2]*(-z21)).clip(-3, 3)

# market vol percentile (PIT, expanding)
mkt = R["SPY"] if "SPY" in R.columns else R.mean(axis=1)
v21 = mkt.rolling(21).std()
vpct = v21.expanding(252).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)  # slow but once
print(f"vol pct ready t={time.time()-t0:.0f}s", flush=True)

def build_book(ema, lam, regime):
    z = zC.ewm(span=ema, min_periods=1).mean() if ema > 1 else zC
    pos = z.clip(lower=0); neg = (-z).clip(lower=0)
    tgt = 0.5*pos.div(pos.sum(axis=1), axis=0).fillna(0.0) - 0.5*neg.div(neg.sum(axis=1), axis=0).fillna(0.0)
    if regime:
        scale = (0.5 + vpct.clip(0, 1)).shift(1).fillna(1.0)   # 0.5x..1.5x gross
        tgt = tgt.mul(scale, axis=0)
    # partial adjustment
    Tv = tgt.values; Wv = np.zeros_like(Tv)
    cur = np.zeros(Tv.shape[1])
    for ti in range(len(idx)):
        cur = cur + lam*(Tv[ti] - cur)
        Wv[ti] = cur
    return pd.DataFrame(Wv, index=idx, columns=zC.columns)

dev_a = pd.Timestamp("1996-06-01")
for ema, lam, regime in [(1, 1.0, False), (3, 0.3, False), (5, 0.2, False),
                          (3, 0.3, True), (5, 0.15, True), (10, 0.1, False)]:
    W = build_book(ema, lam, regime)
    W = W[(W.index >= dev_a) & (W.index <= DEV_END)]
    line = f"ladder ema{ema} lam{lam}{' REG' if regime else ''}:"
    for fee in [0, 2, 5, 10, 20]:
        net, gross, tno = G.run(W, R, fee_bps=fee)
        s = G.sharpe(net[dev_a:DEV_END])
        line += f"  {fee}bp {s:5.2f}"
    line += f"  (tno {tno[dev_a:DEV_END].mean():.3f}/d, gross {G.sharpe(gross[dev_a:DEV_END]):.2f})"
    print(line, flush=True)
    if regime and ema == 5:
        net, gross, tno = G.run(W, R, fee_bps=10)
        net[dev_a:DEV_END].to_pickle("/tmp/sharpe3_work/sleeve_ladder_best.pkl")
print(f"exp11 done t={time.time()-t0:.0f}s", flush=True)
