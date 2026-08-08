"""Turnover-matched null for the best DEV config (contract item 8, done right).

Same book machinery as the winning config (EMA5 smoothing, lam 0.15 partial
adjustment, proportional weights, regime scale OFF to isolate machinery), but
the signal is pure noise with the same cross-sectional footprint (random
normal on the same eligible names each day, EMA'd identically). K=30 runs,
DEV Sharpe distribution at 10 bps. The +0.51 headline is judged against this.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
idx = R.index
E_d = ELIG.reindex(idx).ffill().fillna(False)[R.columns].astype(bool)
Ev = E_d.values
dev_a, dev_b = pd.Timestamp("1996-06-01"), pd.Timestamp("2014-12-31")
rng = np.random.default_rng(11)

out = []
for k in range(30):
    Zn = np.where(Ev, rng.standard_normal(Ev.shape).astype(np.float32), np.nan)
    z = pd.DataFrame(Zn, index=idx, columns=R.columns).ewm(span=5, min_periods=1).mean().clip(-3, 3)
    pos = z.clip(lower=0); neg = (-z).clip(lower=0)
    tgt = 0.5*pos.div(pos.sum(axis=1), axis=0).fillna(0.0) - 0.5*neg.div(neg.sum(axis=1), axis=0).fillna(0.0)
    Tv = tgt.values; Wv = np.zeros_like(Tv); cur = np.zeros(Tv.shape[1])
    for ti in range(len(Tv)):
        cur = cur + 0.15*(Tv[ti]-cur); Wv[ti] = cur
    W = pd.DataFrame(Wv, index=idx, columns=R.columns)
    net, gross, tno = G.run(W, R, fee_bps=10)
    s = G.sharpe(net[dev_a:dev_b])
    out.append(s)
    print(f"null {k}: {s:5.2f} (tno {tno[dev_a:dev_b].mean():.3f})", flush=True)
a = np.array(out)
print(f"matched null: mean {a.mean():.2f} sd {a.std():.2f} p95 {np.percentile(a,95):.2f} "
      f"p99 {np.percentile(a,99):.2f} max {a.max():.2f}", flush=True)
print(f"+0.51 z-score vs matched null: {(0.51-a.mean())/a.std():.1f}", flush=True)
print(f"exp17 done t={time.time()-t0:.0f}s", flush=True)
