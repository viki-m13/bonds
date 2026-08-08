"""Invention track K: path-conditional entries + long-hedged book.

A) Event study refinement: after a z<-2 1-day residual crash, split the
   t+2..t+11 residual payoff by what happened on t+1 (the day we can't trade):
   bounced already (r_resid t+1 > 0) vs not yet. If the payoff concentrates in
   'not yet bounced', conditioning doubles per-trade edge at half the trades.
B) Book: long-only not-yet-bounced crashes (hold H days, overlapping tranches),
   hedged short SPY dollar-for-dollar. Borrow cost on SPY only. Frontier fees.
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
rvol = resid.rolling(63, min_periods=40).std().shift(1)
Z = (resid/rvol).where(E_d).astype(np.float32)
Zv = Z.values; Rres = resid.values
dev_mask = (idx >= pd.Timestamp("1996-06-01")) & (idx <= DEV_END)

# ---------- A) path-conditional event study ----------
print("=== crash z<-2, payoff t+2.. by t+1 behavior (bps of residual) ===")
ev = np.argwhere((Zv < -2) & dev_mask[:, None])
ev = ev[ev[:, 0] < len(idx) - 13]
groups = {"bounced t+1 (r1>+0.5s)": [], "flat t+1": [], "fell more t+1 (r1<-0.5s)": []}
rv = rvol.values
for ti, ni in ev:
    r1 = Rres[ti+1, ni]; s = rv[ti, ni]
    if not np.isfinite(r1) or not np.isfinite(s) or s == 0: continue
    c = np.nancumsum(Rres[ti+2:ti+12, ni])
    if len(c) < 10 or not np.isfinite(c[4]): continue
    key = "bounced t+1 (r1>+0.5s)" if r1 > 0.5*s else ("fell more t+1 (r1<-0.5s)" if r1 < -0.5*s else "flat t+1")
    groups[key].append((c[0], c[2], c[4], c[9]))
for k, v in groups.items():
    a = np.array(v)
    print(f"  {k:26} n={len(v):6}  d1 {a[:,0].mean()*1e4:5.0f}  d3 {a[:,1].mean()*1e4:5.0f}  d5 {a[:,2].mean()*1e4:5.0f}  d10 {a[:,3].mean()*1e4:5.0f}", flush=True)

# ---------- B) long-hedged not-yet-bounced book ----------
def book(th, H, cond, fee_list=(0, 5, 10)):
    Wv = np.zeros((len(idx), len(R.columns)))
    spy_col = R.columns.get_loc("SPY")
    for ti in range(1, len(idx)-1):
        if not dev_mask[ti]: continue
        # signal date = ti means: crash happened ti-1? No: use events where crash at ti-1, decide at ti (seeing t+1 path), trade at ti+1 via lag=1...
        # Simpler honest framing: signal at close ti = {crash at ti-1} AND {path condition on ti}; engine lag=1 executes at ti+1 close.
        z_y = Zv[ti-1]; r_t = Rres[ti]; s = rv[ti-1]
        with np.errstate(invalid="ignore"):
            crash = z_y < -th
            if cond == "nb":   sel = crash & (r_t < 0.5*s)
            elif cond == "all": sel = crash
            else:               sel = crash & (r_t > 0.5*s)
        sel &= np.isfinite(r_t)
        for ni in np.where(sel)[0]:
            Wv[ti:ti+H, ni] += 1.0
    g = np.abs(Wv).sum(axis=1); g[g == 0] = 1.0
    Wn = 0.5 * Wv / g[:, None]
    Wn[:, spy_col] = -0.5 * (np.abs(Wn).sum(axis=1) > 1e-9)
    W = pd.DataFrame(Wn, index=idx, columns=R.columns)
    line = f"longhedged z{th} H{H} {cond}:"
    for fee in fee_list:
        net, gross, tno = G.run(W, R, fee_bps=fee, lag=1)
        line += f"  {fee}bp {G.sharpe(net[dev_mask][net[dev_mask].index<=DEV_END]):5.2f}"
    net, gross, tno = G.run(W, R, fee_bps=10, lag=1)
    dm = net.index[dev_mask]
    line += f"  (tno {tno[dm].mean():.3f}, gross {G.sharpe(gross[dm]):.2f})"
    print(line, flush=True)

for th, H in [(2, 5), (2, 10), (3, 10)]:
    for cond in ["nb", "all"]:
        book(th, H, cond)
print(f"exp12 done t={time.time()-t0:.0f}s", flush=True)
