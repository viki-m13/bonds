"""Invention track G: tail-event dislocations.

Part A (diagnosis): event study — after a 1-day idiosyncratic move of z sigma
(residual vs 10-PC factor model, per-name vol-scaled), what is the mean
cumulative RESIDUAL return over the next 1..10 days, starting t+2 (honest lag)?
This measures the per-trade edge ceiling for tail reversal.

Part B (strategy): overlapping-tranche tail book. Each day, enter names whose
1-day resid z < -Z (long) / > +Z (short); hold H days; position 1/(2*N_active).
Uses the cached daily residual matrix from exp06.

DEV only.
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

# per-name residual vol (63d, PIT) -> z-scores
rvol = resid.rolling(63, min_periods=40).std()
Z = (resid / rvol.shift(1)).astype(np.float32)
Z = Z.where(E_d)
dev_mask = (idx >= pd.Timestamp("1996-01-01")) & (idx <= DEV_END)
print(f"Z ready t={time.time()-t0:.0f}s", flush=True)

# ---------- A) event study ----------
Zv = Z.values; Rres = resid.values
print("=== event study: mean cumulative residual return (bps) after 1-day resid z event, entry t+2 ===")
print(f"{'z<':>6} {'n':>8} {'d1':>6} {'d3':>6} {'d5':>6} {'d10':>6}   (long side)")
for th in [-2, -3, -4, -5]:
    ev = np.argwhere((Zv < th) & dev_mask[:, None])
    ev = ev[ev[:, 0] < len(idx) - 13]
    if len(ev) == 0: continue
    cums = {h: [] for h in (1, 3, 5, 10)}
    for ti, ni in ev:
        path = Rres[ti+2:ti+12, ni]
        if not np.isfinite(path[:1]).all(): continue
        c = np.nancumsum(path)
        for h in (1, 3, 5, 10): cums[h].append(c[h-1])
    print(f"{th:>6} {len(cums[1]):>8} " + " ".join(f"{np.nanmean(cums[h])*1e4:6.0f}" for h in (1, 3, 5, 10)), flush=True)
print(f"{'z>':>6}   (short side: mean residual AFTER up-spikes)")
for th in [2, 3, 4, 5]:
    ev = np.argwhere((Zv > th) & dev_mask[:, None])
    ev = ev[ev[:, 0] < len(idx) - 13]
    if len(ev) == 0: continue
    cums = {h: [] for h in (1, 3, 5, 10)}
    for ti, ni in ev:
        path = Rres[ti+2:ti+12, ni]
        if not np.isfinite(path[:1]).all(): continue
        c = np.nancumsum(path)
        for h in (1, 3, 5, 10): cums[h].append(c[h-1])
    print(f"{th:>6} {len(cums[1]):>8} " + " ".join(f"{np.nanmean(cums[h])*1e4:6.0f}" for h in (1, 3, 5, 10)), flush=True)

# ---------- B) tail book ----------
def tail_book(Zth, H, both=True):
    Wv = np.zeros((len(idx), len(R.columns)))
    for ti in range(len(idx)):
        if not dev_mask[ti]: continue
        z = Zv[ti]
        longs = np.where(z < -Zth)[0]
        shorts = np.where(z > Zth)[0] if both else np.array([], dtype=int)
        for arr, sgn in ((longs, 1.0), (shorts, -1.0)):
            for ni in arr:
                Wv[ti:ti+H, ni] += sgn
    # normalize each day to gross 1
    g = np.abs(Wv).sum(axis=1); g[g == 0] = 1
    Wn = Wv / g[:, None]
    W = pd.DataFrame(Wn, index=idx, columns=R.columns)
    net, gross, tno = G.run(W, R, lag=2)
    rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], f"tail z{Zth} H{H}{'' if both else ' Lonly'}")
    print(G.fmt(rep), flush=True)
    net[:DEV_END].to_pickle(f"/tmp/sharpe3_work/sleeve_tail_z{Zth}_H{H}{'LS' if both else 'L'}.pkl")
    return rep

for Zth, H in [(3, 5), (3, 10), (4, 5), (4, 10), (2.5, 5)]:
    tail_book(Zth, H)
tail_book(3, 10, both=False)
print(f"exp08 done t={time.time()-t0:.0f}s", flush=True)
