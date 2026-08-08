"""Modern-regime probes + null battery.

CONTRACT AMENDMENT (logged): the original DEV (1995-2014) selected strategies
whose alpha died by 2015 — every VAL look failed. For modern-regime ideas we
re-partition: DEV2 = 2005-2019 (selection window), TEST = 2020+ remains locked
and untouched. Everything tried on DEV2 is logged here.

1. Null battery for the best original-DEV config (+0.51): 50 matched random
   books -> where does +0.51 sit in the luck distribution?
2. Crisis-only ladder, fixed (vol-percentile via rolling 756d rank).
3. Modern probes on DEV2:
   a. spike-fade: short day-2 of >+3z 1-day resid up-spikes with volume>3x,
      hedged long SPY (the 'fade the pump' book), 2005-2019.
   b. quiet-drift: long minimal-|z5| names (no-news drift), short SPY.
   c. Friday->Monday reversal of weekly losers (weekend liquidity).
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
DEV2_A, DEV2_B = pd.Timestamp("2005-01-01"), pd.Timestamp("2019-12-31")
spy_col = R.columns.get_loc("SPY")

# ---------- 1) null battery ----------
wkall = [d for d in G.week_ends(idx) if pd.Timestamp("1996-06-01") < d <= pd.Timestamp("2014-12-31")]
nulls = G.null_sharpes(wkall, ELIG, R, 450, 450, K=50, seed=7)
print(f"null battery (450/450 weekly, 96-2014, 10bp): mean {np.nanmean(nulls):.2f} "
      f"sd {np.nanstd(nulls):.2f} p99 {np.nanpercentile(nulls,99):.2f} max {np.nanmax(nulls):.2f}", flush=True)
print(f"  -> best original-DEV net +0.51 is {(0.51-np.nanmean(nulls))/np.nanstd(nulls):.1f} sigma above null mean; "
      f"~60 configs were tried", flush=True)

# ---------- 2) crisis-only ladder (fixed percentile) ----------
mkt = R["SPY"]
v21 = mkt.rolling(21).std()
vpct = v21.rolling(756, min_periods=252).rank(pct=True)
a = np.array([0.455, 0.34, 0.205])
z21m = (resid.rolling(21).sum()/(rvol*np.sqrt(21))).where(E_d).astype(np.float32)
zC = (a[0]*(-z1) + a[1]*(-z5) + a[2]*(-z21m)).clip(-3, 3)
pos = zC.clip(lower=0); neg = (-zC).clip(lower=0)
Wl = 0.5*pos.div(pos.sum(axis=1), axis=0).fillna(0.0) - 0.5*neg.div(neg.sum(axis=1), axis=0).fillna(0.0)
on = (vpct.shift(1) > 0.8).astype(float)
Wc = Wl.mul(on, axis=0)
for fee in (0, 2, 5):
    net, gross, tno = G.run(Wc, R, fee_bps=fee)
    dev = net["1996-06":"2014"]
    act = Wc.abs().sum(axis=1)["1996-06":"2014"] > 0
    print(f"crisis-only ladder {fee}bp: full {G.sharpe(dev):5.2f}  active-only {G.sharpe(dev[act]):5.2f}  (active {act.mean():.0%})", flush=True)

# ---------- 3) modern probes on DEV2 ----------
def hedged_book(sel_fn, name, H=3, side=-1.0):
    Wv = np.zeros((len(idx), len(R.columns)))
    for ti in range(1, len(idx)-1):
        d = idx[ti]
        if not (DEV2_A <= d <= DEV2_B): continue
        sel = sel_fn(ti)
        for ni in sel:
            Wv[ti:ti+H, ni] += side
    g = np.abs(Wv).sum(axis=1); g[g == 0] = 1.0
    Wn = 0.5*Wv/g[:, None]
    Wn[:, spy_col] = -np.sign(Wn.sum(axis=1))*0.5*(np.abs(Wn).sum(axis=1) > 1e-9)
    W = pd.DataFrame(Wn, index=idx, columns=R.columns)
    line = f"{name:26}"
    for fee in (0, 5, 10):
        net, gross, tno = G.run(W, R, fee_bps=fee, lag=1)
        line += f"  {fee}bp {G.sharpe(net[DEV2_A:DEV2_B]):5.2f}"
    net, gross, tno = G.run(W, R, fee_bps=10, lag=1)
    line += f"  (tno {tno[DEV2_A:DEV2_B].mean():.3f}, gross {G.sharpe(gross[DEV2_A:DEV2_B]):.2f})"
    print(line, flush=True)

Zv = z1.values
dvr = (DV.rolling(2).mean() / DV.rolling(63, min_periods=40).mean()).values
def spike_sel(ti):
    with np.errstate(invalid="ignore"):
        return np.where((Zv[ti-1] > 3) & (dvr[ti-1] > 3))[0]
hedged_book(spike_sel, "spike-fade short d2 H3", H=3, side=-1.0)
hedged_book(spike_sel, "spike-MOMO long d2 H3", H=3, side=+1.0)

z5v = z5.values
def quiet_sel(ti):
    z = np.abs(z5v[ti])
    ok = np.where(np.isfinite(z))[0]
    if len(ok) < 100: return []
    thr = np.percentile(z[ok], 20)
    return ok[z[ok] <= thr]
hedged_book(quiet_sel, "quiet-drift long H5", H=5, side=+1.0)

# Friday losers -> Monday
days = pd.Series(idx.dayofweek, index=idx)
def friday_losers(ti):
    if days.iloc[ti] != 4: return []
    z = z5v[ti]
    ok = np.where(np.isfinite(z))[0]
    if len(ok) < 100: return []
    thr = np.percentile(z[ok], 5)
    return ok[z[ok] <= thr]
hedged_book(friday_losers, "friday-losers long H2", H=2, side=+1.0)
print(f"exp14 done t={time.time()-t0:.0f}s", flush=True)
